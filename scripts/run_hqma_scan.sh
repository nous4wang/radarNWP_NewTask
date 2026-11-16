#!/usr/bin/env bash
# The scan that separates the three effects.
#
# The hierarchical loss supervises the exceedance probabilities of the
# head, and the head has to be built so that those probabilities answer
# the questions the thresholds ask. The six-bin head does; the five-bin
# head does not, and the difference between them is what the alignment
# is worth. So the scan has three things to separate:
#
#   A1  six bins, a light probability term
#   A2  six bins, the main setting
#   A3  six bins, the main setting with the focal modulation
#   A4  five bins, the misaligned head, everything else as A2
#
# A4 is the control. Without it, a difference between A2 and the earlier
# runs would be unreadable: the head, the loss and the alignment all
# changed at once.
#
# Two controls from outside the scan are needed to read the results at
# all: the backbone (simvp_v2) and the plain concatenation (M1), both
# under the benchmark's own loss. They are in runs/hqma from the
# previous round and are quoted in the summary.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/hqma_scan}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
GPUS="${GPUS:-0 1 2 3}"

mkdir -p "${OUT_DIR}"

# The four runs: name, GPU, alpha, bins, focal.
RUNS=(
    "scan_A1 0 0.1 6 0"
    "scan_A2 1 0.3 6 0"
    "scan_A3 2 0.3 6 1"
    "scan_A4 3 0.3 5 0"
)

echo "=== phase 1: training ==="
for entry in "${RUNS[@]}"; do
    read -r run gpu alpha bins focal <<< "${entry}"
    rundir="${OUT_DIR}/${run}"
    log="${OUT_DIR}/${run}.log"
    mkdir -p "${rundir}"

    focal_flag="--no-focal"
    if [[ "${focal}" == "1" ]]; then
        focal_flag="--use-focal"
    fi

    echo "  -> ${run} on GPU ${gpu} (alpha=${alpha}, bins=${bins}, ${focal_flag})"
    (
        CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS=4 \
        PYTHONUNBUFFERED=1 \
        "${PYTHON}" "${REPO_ROOT}/scripts/train.py" \
            --model simvp_v2_hqma \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split train \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --lr "${LR}" \
            --loss hierarchical \
            --alpha-prob "${alpha}" \
            --head-bins "${bins}" \
            "${focal_flag}" \
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
i=0
for entry in "${RUNS[@]}"; do
    read -r run gpu alpha bins focal <<< "${entry}"
    ckpt="$(ls -dt "${OUT_DIR}/${run}"/*/last.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${run}] no checkpoint -- skipped"
        continue
    fi
    (
        CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
            --model simvp_v2_hqma \
            --checkpoint "${ckpt}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split test \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --device cuda \
            --out-json "${OUT_DIR}/${run}.eval.json" \
            > "${OUT_DIR}/${run}.eval.log" 2>&1
        echo "  [${run}] evaluated"
    ) &
    i=$(( i + 1 ))
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

# The controls from the previous round, under the benchmark's own loss.
CONTROLS = {
    "simvp_v2 (backbone)": "/home/wf/radarNWP/runs/hqma/simvp_v2.eval.json",
    "M1 concat": "/home/wf/radarNWP/runs/hqma/simvp_v2_nwp_concat.eval.json",
    "hqma_balanced": "/home/wf/radarNWP/runs/hqma/hqma_balanced.eval.json",
}

SCAN = {
    "scan_A1": "6 bins, alpha 0.1",
    "scan_A2": "6 bins, alpha 0.3",
    "scan_A3": "6 bins, alpha 0.3, focal",
    "scan_A4": "5 bins, alpha 0.3",
}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def row(name, report):
    t = report["thresholds"]
    return (
        name,
        t["0.1"]["csi"], t["0.5"]["csi"], t["2.0"]["csi"], t["4.0"]["csi"],
        t["0.5"]["far"], t["0.5"]["pod"], t["0.5"]["bias"],
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

print(f"{'run':44s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
      f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'POD@0.5':>9s}{'BIAS@0.5':>9s}")
print("-" * 107)
for r in rows:
    print(f"{r[0]:44s}" + "".join(f"{v:9.3f}" for v in r[1:]))

print()
print("the two numbers the design turns on:")
print("  BIAS@0.5  has to rise above the backbone's, or the loss is not")
print("             correcting the under-prediction")
print("  CSI@0.1   has not to collapse, or the correction is being paid")
print("             for with false alarms")
print()
print("A2 against A4 is the alignment: same loss, same strength, the only")
print("difference is whether the probabilities answer the questions the")
print("thresholds ask.")
PYEOF

echo
echo "=== done: reports in ${OUT_DIR} ==="
