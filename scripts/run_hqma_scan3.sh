#!/usr/bin/env bash
# The scan that decides whether the three defects are the reason the
# heavy-rain score is zero.
#
# Three defects were found by measurement on the archive, not by reading
# the code:
#
#   1. the head reconstructed the rain rate as the expectation over the
#      bins, with each bin represented by its lower edge. The bin that
#      begins at 2.0 mm was therefore represented by 2.0 mm, and the
#      expectation of a distribution that sits on a bin edge and leaks
#      slightly below it is below the edge. The measured maximum
#      prediction was 1.999797 mm: 1.51 % of the pixels sat within a
#      hundredth of a millimetre below the 2.0 mm threshold and none
#      crossed it, so the score at 2.0 mm was exactly zero however much
#      probability mass the head put on the heavy bins.
#
#   2. the tier weights were looked up by position, so a run scoring the
#      business thresholds (0.1, 0.5, 2.0, 4.0) gave the 2.0 mm tier
#      the weights meant for 1.0 mm and never used the heaviest tier's
#      weights at all.
#
#   3. the probability channels were matched to the thresholds by
#      position as well. The head's channels answer (0.1, 0.5, 1.0, 2.0,
#      4.0), so the run was supervising P(R >= 1.0) against the truth's
#      "at or above 2 mm" indicator, which is what drove P(R >= 2 mm) to
#      almost zero.
#
# The arms are a two by two: the head's construction against the loss.
#
#   C1  midpoint head, field-wide tier loss   the candidate
#   C2  floor head,    field-wide tier loss   the control for C1
#   C3  midpoint head, soft-CSI loss          the head under the loss
#                                             that measured best before
#   C4  floor head,    soft-CSI loss          the control for C3
#
# C2 and C4 also serve as the reproduction of what the previous scan
# measured, so the difference between the two columns is the head and
# nothing else.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/hqma_scan3}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
GPUS="${GPUS:-0 1 2 3 4 5}"

# The thresholds the runs are trained against and scored at. 2.0 mm per
# 6-minute accumulation is 20 mm/h, the short-duration heavy-rain warning
# line. 4.0 mm is 40 mm/h. 10.0 mm is deliberately absent: it holds
# 0.002 % of the field, and a score on a few hundred pixels cannot tell
# two models apart.
TIERS="0.1 0.5 2.0 4.0"

mkdir -p "${OUT_DIR}"

# The runs: name, GPU, loss, level mode.
RUNS=(
    "scan_C1 0 field_csi midpoint"
    "scan_C2 1 field_csi floor"
    "scan_C3 2 balanced midpoint"
    "scan_C4 3 balanced floor"
)

echo "=== phase 1: training ==="
for entry in "${RUNS[@]}"; do
    read -r run gpu loss level <<< "${entry}"
    rundir="${OUT_DIR}/${run}"
    log="${OUT_DIR}/${run}.log"
    mkdir -p "${rundir}"

    echo "  -> ${run} on GPU ${gpu} (loss=${loss}, level=${level})"
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
            --loss "${loss}" \
            --alpha-prob 0.3 \
            --tier-thresholds ${TIERS} \
            --head-bins 6 \
            --level-mode "${level}" \
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
    read -r run gpu loss level <<< "${entry}"
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

# The controls from the previous rounds. The soft-CSI row is quoted from
# the run that used the same architecture, so the comparison is not
# against a different model.
CONTROLS = {
    "simvp_v2 (backbone)": "/home/wf/radarNWP/runs/hqma/simvp_v2.eval.json",
    "M1 concat": "/home/wf/radarNWP/runs/hqma/simvp_v2_nwp_concat.eval.json",
    "scan_B2 (floor head, old scan)":
        "/home/wf/radarNWP/runs/hqma_scan2/scan_B2.eval.json",
}

SCAN = {
    "scan_C1": "midpoint head, field_csi",
    "scan_C2": "floor head, field_csi",
    "scan_C3": "midpoint head, soft-CSI",
    "scan_C4": "floor head, soft-CSI",
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
        g("0.5", "far"), g("0.5", "pod"), g("0.5", "bias"),
        g("2.0", "hits"),
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
      f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'POD@0.5':>9s}{'BIAS@0.5':>9s}"
      f"{'hits@2.0':>12s}")
print("-" * 119)
for r in rows:
    print(f"{r[0]:44s}" + "".join(f"{v:9.3f}" for v in r[1:8])
          + f"{int(r[8]):12d}")

print()
print("C1 against C3 is the midpoint head against the floor head under")
print("the same loss, on the same cards, at the same time. C3 reproduces")
print("what the previous scan measured, so the difference between them")
print("is the head and nothing else.")
print()
print("The number to read is hits@2.0. The previous scan's best was")
print("zero: the reconstruction sat on the threshold and never crossed")
print("it. A run that puts a heavy pixel above the threshold has to")
print("show hits there.")
PYEOF

echo
echo "=== done: reports in ${OUT_DIR} ==="
