#!/usr/bin/env bash
# Does the collapse of the heavy-rain score belong to the loss or to
# the models?
#
# The matrix said that every method on this benchmark collapses on the
# heavy rain, and that the collapse is a property of the method family:
# the frequency bias is 0.08 to 0.33 and the small-scale energy is
# flattened. That was measured with the benchmark's own objective, the
# log-space regression.
#
# The fix for the three defects changed the heavy-rain score of one
# model from zero to 0.123, which is above the physical extrapolation
# baseline. So the collapse is not a property of the family. The
# question this scan answers is what it is a property of.
#
# The arm that answers it is the plain backbone trained with the
# score-aware objective. It has no hierarchical head, so the whole
# effect has to come from the objective. If it also reaches the heavy
# rain, then the collapse belongs to the log-space regression and
# every model in the matrix was limited by the loss rather than by its
# architecture.
#
#   D1  simvp_v2, the score-aware loss        the decisive arm
#   D2  simvp_v2_nwp_concat, the score-aware loss
#   D3  simvp_v2_quantum, the score-aware loss
#   D4  simvp_v2, the benchmark's own loss    the control
#
# Four runs on four cards.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/hqma_scan4}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
GPUS="${GPUS:-0 1 2 3}"

TIERS="0.1 0.5 2.0 4.0"

mkdir -p "${OUT_DIR}"

# The runs: name, GPU, model, loss.
RUNS=(
    "scan_D1 0 simvp_v2 balanced"
    "scan_D2 1 simvp_v2_nwp_concat balanced"
    "scan_D3 2 simvp_v2_quantum balanced"
    "scan_D4 3 simvp_v2 standard"
)

echo "=== phase 1: training ==="
for entry in "${RUNS[@]}"; do
    read -r run gpu model loss <<< "${entry}"
    rundir="${OUT_DIR}/${run}"
    log="${OUT_DIR}/${run}.log"
    mkdir -p "${rundir}"

    echo "  -> ${run} on GPU ${gpu} (model=${model}, loss=${loss})"
    (
        CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS=4 \
        PYTHONUNBUFFERED=1 \
        "${PYTHON}" "${REPO_ROOT}/scripts/train.py" \
            --model "${model}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split train \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --lr "${LR}" \
            --loss "${loss}" \
            --alpha-csi 0.5 \
            --tier-thresholds ${TIERS} \
            --log-every 200 \
            --device cuda \
            --out-dir "${rundir}" \
            > "${log}" 2>&1
        echo "  [${run}] $(grep -E '^epoch [0-9]+:' "${log}" | tail -1)"
    ) &
    sleep 4
done
wait
echo "all training finished"

echo
echo "=== phase 2: evaluation (clean test split) ==="
for entry in "${RUNS[@]}"; do
    read -r run gpu model loss <<< "${entry}"
    ckpt="$(ls -dt "${OUT_DIR}/${run}"/*/last.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${run}] no checkpoint -- skipped"
        continue
    fi
    (
        CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
            --model "${model}" \
            --checkpoint "${ckpt}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split test \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --thresholds ${TIERS} \
            --device cuda \
            --out-json "${OUT_DIR}/${run}.eval.json" \
            > "${OUT_DIR}/${run}.eval.log" 2>&1
        echo "  [${run}] evaluated"
    ) &
    sleep 3
done
wait

echo
echo "=== the summary ==="
"${PYTHON}" - "${OUT_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

CONTROLS = {
    "simvp_v2 (benchmark loss, old matrix)":
        "/home/wf/radarNWP/runs/hqma/simvp_v2.eval.json",
    "pysteps_adv (physical extrapolation)":
        "/home/wf/radarNWP/runs/tgrs/pysteps_adv.eval.json",
    "scan_C1 (HQMA, fixed)":
        "/home/wf/radarNWP/runs/hqma_scan3/scan_C1.eval.json",
}

SCAN = {
    "scan_D1": "simvp_v2, score-aware loss",
    "scan_D2": "M1 concat, score-aware loss",
    "scan_D3": "quantum plugin, score-aware loss",
    "scan_D4": "simvp_v2, benchmark loss",
}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def row(name, report):
    t = report["thresholds"]

    def g(thr, key):
        return t.get(thr, {}).get(key, float("nan"))

    return (
        name,
        g("0.1", "csi"), g("0.5", "csi"), g("2.0", "csi"), g("4.0", "csi"),
        g("0.5", "far"), g("0.5", "bias"), g("2.0", "hits"),
    )


rows = []
for name, path in CONTROLS.items():
    report = read(path)
    if report is not None:
        rows.append(row(name, report))
for run, label in SCAN.items():
    report = read(out / f"{run}.eval.json")
    if report is not None:
        rows.append(row(f"{run} ({label})", report))

print(f"{'run':46s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
      f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}{'hits@2.0':>12s}")
print("-" * 121)
for r in rows:
    print(f"{r[0]:46s}" + "".join(f"{v:9.3f}" for v in r[1:7])
          + f"{int(r[7]):12d}")

print()
print("D1 is the decisive arm: the plain backbone, no hierarchical head,")
print("no NWP, trained with the score-aware objective. If it reaches")
print("the heavy rain, the collapse belonged to the log-space")
print("regression and not to the method family.")
PYEOF

echo
echo "=== done: reports in ${OUT_DIR} ==="
