#!/usr/bin/env bash
# Route P3: the observation-assimilation lag sweep, with the training-time
# injection fixed.
#
# The previous sweep of this family measured nothing. The training loop
# passed the observed history to the objective but never to the model, so
# all four arms trained on identical inputs and their losses agreed to a
# thousandth. What was measured was the price of handing a model an input
# distribution it had never seen, not the value of the observation.
#
# The training loop now publishes the observation on the model, exactly as
# the evaluation does, so the arms differ in the one thing they are
# supposed to differ in.
#
# The lag sweep asks an operational question: if the ground rainfall is
# late by twelve minutes, or by twenty-four, or never arrives, how much of
# the benefit is left? The answer is the curve.
#
# The arms use qpe_weight = 0. With the injection in place, keeping the
# history supervision would mix two variables -- what the model is given,
# and what the inverse branch is trained against -- and the sweep would
# measure their combination rather than the lag. Route R1 owns the
# supervision question.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p3}"
EPOCHS="${EPOCHS:-20}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"

mkdir -p "${OUT_DIR}"

# name  model  gpu  lag
ARMS=(
    "assim_lag0 simvp_v2_assimilated 0 0"
    "assim_lag2 simvp_v2_assimilated 1 2"
    "assim_lag4 simvp_v2_assimilated 2 4"
    "assim_lag11 simvp_v2_assimilated 3 11"
)

echo "=========================================================="
echo " P3: the observation-assimilation lag sweep"
echo " qpe_weight = 0 (R1 owns the supervision question)"
echo "=========================================================="

for entry in "${ARMS[@]}"; do
    read -r name model gpu lag <<< "${entry}"
    log="${OUT_DIR}/${name}.log"
    echo "  -> ${name} (lag ${lag}) on GPU ${gpu}"
    (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS=4 \
        "${PYTHON}" scripts/train.py \
            --model "${model}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split train \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --device cuda \
            --out-dir "${OUT_DIR}/${name}" \
            --loss balanced \
            --alpha-csi "${ALPHA_CSI}" \
            --tier-thresholds ${TIERS} \
            --qpe-weight 0.0 \
            --model-kwargs "{\"obs_lag_steps\":${lag}}" \
            > "${log}" 2>&1
    ) &
done
wait

echo "training done; scoring on the clean test split"

for entry in "${ARMS[@]}"; do
    read -r name model gpu lag <<< "${entry}"
    run_dir="$(ls -dt "${OUT_DIR}/${name}"/*/ 2>/dev/null | head -1)"
    if [[ -z "${run_dir}" ]]; then
        echo "  ${name}: no run directory, skipped"
        continue
    fi
    ck="${run_dir}last.pt"
    [[ -f "${ck}" ]] || continue
    (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON}" scripts/evaluate.py \
            --model "${model}" \
            --checkpoint "${ck}" \
            --data-root "${DATA_ROOT}" \
            --packed --split test \
            --batch-size "${BATCH_SIZE}" --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" --slide-stride "${SLIDE_STRIDE}" \
            --thresholds ${TIERS} \
            --device cuda \
            --model-kwargs "{\"obs_lag_steps\":${lag}}" \
            --out-json "${OUT_DIR}/${name}.eval.json" \
            > "${OUT_DIR}/${name}.eval.log" 2>&1
    ) &
done
wait

# ---------------------------------------------------------------------------
# The summary, and the check that the fix did what it was supposed to.
#
# The four arms must now train differently. If their losses still agree to
# a thousandth, the injection is still lazy and the whole sweep is void
# again -- so the check is printed next to the scores, not buried.
# ---------------------------------------------------------------------------
"${PYTHON}" - "${OUT_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
ARMS = [("assim_lag0", "no delay"),
        ("assim_lag2", "12 min late"),
        ("assim_lag4", "24 min late"),
        ("assim_lag11", "no observation")]

HEADER = (f"{'arm':16s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
          f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}{'n':>7s}")

print("\n" + "=" * 88)
print("the lag sweep")
print("=" * 88)
print(HEADER)
print("-" * len(HEADER))
for name, label in ARMS:
    path = out_dir / f"{name}.eval.json"
    if not path.exists():
        print(f"{label:16s}(not finished)")
        continue
    report = json.loads(path.read_text())
    t = report["thresholds"]
    g = lambda th, key: t.get(th, {}).get(key, float("nan"))
    print(f"{label:16s}"
          + "".join(f"{g(th, 'csi'):9.3f}" for th in ("0.1", "0.5", "2.0", "4.0"))
          + f"{g('0.5', 'far'):9.3f}{g('0.5', 'bias'):9.3f}"
          + f"{report.get('n_samples', 0):7d}")

print("\n" + "=" * 88)
print("did the injection reach the training? (epoch-0 and epoch-19 losses)")
print("=" * 88)
losses = {}
for name, _ in ARMS:
    log = out_dir / f"{name}.log"
    if not log.exists():
        continue
    picked = {}
    for line in log.read_text(errors="ignore").splitlines():
        if line.startswith("epoch ") and ": loss=" in line:
            try:
                epoch = int(line.split()[1].rstrip(":"))
                picked[epoch] = float(line.split("loss=")[1].split()[0])
            except (ValueError, IndexError):
                continue
    losses[name] = picked
    if picked:
        first = picked.get(min(picked))
        last = picked.get(max(picked))
        print(f"  {name:16s} ep{min(picked)}={first:.4f}  ep{max(picked)}={last:.4f}")

ep0 = [v.get(0) for v in losses.values() if v.get(0) is not None]
if len(ep0) >= 2:
    spread = max(ep0) - min(ep0)
    if spread < 1e-3:
        print("\n  FAIL: the epoch-0 losses still agree to a thousandth.")
        print("  The observation still does not reach the training, so this")
        print("  sweep measures nothing again.")
    else:
        print(f"\n  PASS: the epoch-0 losses differ by {spread:.4f}, so the")
        print("  lag now changes what the model trains on.")
print("\nThese arms consume ground observations at inference, so their")
print("scores are not comparable with the pure radar baselines.")
PYEOF

echo "=========================================================="
echo " P3 done."
echo "=========================================================="
