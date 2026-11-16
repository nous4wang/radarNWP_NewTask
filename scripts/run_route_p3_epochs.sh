#!/usr/bin/env bash
# Route P3, re-measured at the epoch the score actually peaks.
#
# Why this exists
# ---------------
# ``scripts/run_route_p3.sh`` scores ``last.pt``, which ``scripts/train.py``
# rewrites at the end of every epoch.  Route P2 then measured what that
# epoch is worth: every arm peaks at **epoch 0** and loses 2.7x (qpe_05)
# to 8.5x (base_concat) by epoch 19.  So the P3 lag curve was read off
# the collapsed end of every curve, where the differences between arms are
# small relative to the collapse.
#
# The lag curve is not monotone there either: 12 minutes late (0.0334)
# scores below no observation at all (0.0399), which no mechanism
# explains.  That is the signature of a measurement taken where the
# signal has already decayed into noise, not a property of the
# assimilation lag.
#
# What this script does
# ----------------------
# The same four arms, the same hyper-parameters, the same seed (0), the
# same ``qpe_weight = 0`` -- but trained with ``train_keep_epochs.py``,
# which keeps a copy of every epoch.  Two epochs are then scored:
#
#   ep00  the peak, which is the quantity the comparison is about;
#   ep19  the epoch ``last.pt`` would have held, which reproduces the
#         original P3 run exactly if the pipeline is unchanged.
#
# The ep19 arm is the check on the whole thing.  The original P3 run
# used ``scripts/train.py``; this one uses a wrapper that patches
# ``torch.save``.  If the wrapper changed anything -- the random stream,
# the training loop, anything -- the ep19 scores would move.  They must
# come back bit-identical, and they must match the original
# ``runs/route_p3/*.eval.json`` to the digit.  If they do not, the ep00
# numbers are not comparable with the original and the run is void.
#
# The judge is the file count: eight ``.eval.json`` files under
# ``runs/route_p3_epochs``.  The original ``runs/route_p3/`` is left
# alone -- it is the record of what the first run produced, and this
# script is a second measurement, not a replacement.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p3_epochs}"
EPOCHS="${EPOCHS:-20}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
SCORE_EPOCHS="${SCORE_EPOCHS:-00 19}"
SEED="${SEED:-0}"

mkdir -p "${OUT_DIR}"

# name  model  gpu  lag
ARMS=(
    "assim_lag0 simvp_v2_assimilated 0 0"
    "assim_lag2 simvp_v2_assimilated 1 2"
    "assim_lag4 simvp_v2_assimilated 2 4"
    "assim_lag11 simvp_v2_assimilated 3 11"
)

echo "=========================================================="
echo " P3 re-measured: the lag sweep at the peak epoch"
echo " seed ${SEED}, qpe_weight = 0, scoring epochs: ${SCORE_EPOCHS}"
echo " the ep19 scores must reproduce runs/route_p3 exactly"
echo "=========================================================="

for entry in "${ARMS[@]}"; do
    read -r name model gpu lag <<< "${entry}"
    log="${OUT_DIR}/${name}.log"
    echo "  -> ${name} (lag ${lag}) on GPU ${gpu}"
    (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS=4 \
        "${PYTHON}" scripts/train_keep_epochs.py \
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
            --seed "${SEED}" \
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

echo "training done; scoring the peak and the last epoch"

for entry in "${ARMS[@]}"; do
    read -r name model gpu lag <<< "${entry}"
    run_dir="$(ls -dt "${OUT_DIR}/${name}"/*/ 2>/dev/null | head -1)"
    if [[ -z "${run_dir}" ]]; then
        echo "  ${name}: no run directory, skipped"
        continue
    fi
    for ep in ${SCORE_EPOCHS}; do
        ck="${run_dir}epoch_${ep}.pt"
        [[ -f "${ck}" ]] || continue
        out="${OUT_DIR}/${name}_ep${ep}.eval.json"
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
                --out-json "${out}" \
                > "${out%.json}.log" 2>&1
        ) &
    done
done
wait

# ---------------------------------------------------------------------------
# The check that goes with the numbers: the ep19 scores must come back
# identical to the original P3 run.  They are the same weights, scored by
# the same code, so any difference means the pipeline moved and the ep00
# numbers cannot be compared with the original.
# ---------------------------------------------------------------------------
"${PYTHON}" - "${OUT_DIR}" "${REPO_ROOT}/runs/route_p3" <<'PYEOF'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
orig_dir = Path(sys.argv[2])
ARMS = ["assim_lag0", "assim_lag2", "assim_lag4", "assim_lag11"]
TIERS = ["0.1", "0.5", "2.0", "4.0"]


def csi(path):
    t = json.loads(path.read_text())["thresholds"]
    return {th: t[th]["csi"] for th in TIERS}


print("\n" + "=" * 88)
print("the peak-epoch lag curve")
print("=" * 88)
header = f"{'arm':16s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}{'CSI@4.0':>9s}"
print(header)
print("-" * len(header))
for name in ARMS:
    path = out_dir / f"{name}_ep00.eval.json"
    if not path.exists():
        print(f"{name:16s}(not finished)")
        continue
    c = csi(path)
    print(f"{name:16s}" + "".join(f"{c[th]:9.4f}" for th in TIERS))

print("\n" + "=" * 88)
print("the reproducibility check: ep19 here against the original P3 run")
print("=" * 88)
all_same = True
for name in ARMS:
    new = out_dir / f"{name}_ep19.eval.json"
    old = orig_dir / f"{name}.eval.json"
    if not new.exists() or not old.exists():
        print(f"  {name:16s}(missing: new={new.exists()} old={old.exists()})")
        all_same = False
        continue
    a, b = csi(new), csi(old)
    same = all(abs(a[th] - b[th]) < 1e-12 for th in TIERS)
    all_same = all_same and same
    print(f"  {name:16s} new ep19 CSI@2.0 = {a['2.0']:.6f}   "
          f"original = {b['2.0']:.6f}   {'identical' if same else 'DIFFERENT'}")
if all_same:
    print("\n  PASS: the ep19 scores are identical to the original run, so")
    print("        the peak-epoch scores are comparable with it.")
else:
    print("\n  FAIL: the ep19 scores moved. The peak-epoch numbers are")
    print("        not comparable with the original run.")
PYEOF

echo "=========================================================="
echo " P3 peak re-measurement done."
echo "=========================================================="
