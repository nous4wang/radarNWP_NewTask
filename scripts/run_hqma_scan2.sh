#!/usr/bin/env bash
# The scan that decides whether the field-wide tier loss is the way.
#
# Two variables have to be separated, because both changed at once in the
# previous round:
#
#   the objective   the per-pixel tier loss measured worse than the plain
#                   soft-CSI loss, and the field-wide form is the fix;
#   the thresholds  2.0 mm per 6-minute accumulation is 20 mm/h, which is
#                   the short-duration heavy-rain warning line, so the
#                   score has to be computed there;
#
# and one thing has to be kept as a control:
#
#   the alignment   the five-bin head puts the exceedance probabilities
#                   one tier out of step with the thresholds, and it
#                   measured badly enough to be worth keeping the arm.
#
# So five runs on five cards:
#
#   B1  field-wide tier loss, alpha 0.1
#   B2  field-wide tier loss, alpha 0.3
#   B3  field-wide tier loss, alpha 0.3, per-pixel (the form that failed)
#   B4  field-wide tier loss, alpha 0.5
#   B5  the five-bin head with the field-wide loss (the alignment control)
#
# The controls from the earlier rounds -- the backbone, the plain
# concatenation, and the soft-CSI loss -- are quoted in the summary, and
# the soft-CSI one is the bar to beat.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/hqma_scan2}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
GPUS="${GPUS:-0 1 2 3 4}"

# The thresholds the runs are trained against and scored at. 2.0 mm per
# 6-minute accumulation is 20 mm/h, the short-duration heavy-rain warning
# line; 4.0 is 40 mm/h. 10.0 is deliberately absent: it holds 0.002 % of
# the field, and a score on a few hundred pixels cannot tell two models
# apart.
TIERS="0.1 0.5 2.0 4.0"

mkdir -p "${OUT_DIR}"

# The five runs: name, GPU, alpha, bins, extra flag.
RUNS=(
    "scan_B1 0 0.1 6 "
    "scan_B2 1 0.3 6 "
    "scan_B3 2 0.3 6 --per-pixel"
    "scan_B4 3 0.5 6 "
    "scan_B5 4 0.3 5 "
)

echo "=== phase 1: training ==="
for entry in "${RUNS[@]}"; do
    read -r run gpu alpha bins extra <<< "${entry}"
    rundir="${OUT_DIR}/${run}"
    log="${OUT_DIR}/${run}.log"
    mkdir -p "${rundir}"

    echo "  -> ${run} on GPU ${gpu} (alpha=${alpha}, bins=${bins}, ${extra:-plain})"
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
            --loss field_csi \
            --alpha-prob "${alpha}" \
            --tier-thresholds ${TIERS} \
            --head-bins "${bins}" \
            ${extra} \
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
    read -r run gpu alpha bins extra <<< "${entry}"
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

CONTROLS = {
    "simvp_v2 (backbone)": "/home/wf/radarNWP/runs/hqma/simvp_v2.eval.json",
    "M1 concat": "/home/wf/radarNWP/runs/hqma/simvp_v2_nwp_concat.eval.json",
    "hqma_balanced (soft-CSI)": "/home/wf/radarNWP/runs/hqma/hqma_balanced.eval.json",
}

SCAN = {
    "scan_B1": "field-wide, alpha 0.1",
    "scan_B2": "field-wide, alpha 0.3",
    "scan_B3": "field-wide, alpha 0.3, per-pixel",
    "scan_B4": "field-wide, alpha 0.5",
    "scan_B5": "5 bins (alignment control)",
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
print("the bar to beat is the soft-CSI row: BIAS 0.211, CSI@2.0 0.019,")
print("CSI@0.1 0.251, FAR@0.5 0.585. The question is whether the")
print("field-wide tier loss reaches the bias and the heavy-rain score")
print("without giving the false alarms away.")
print()
print("B2 against B3 is the field-wide form against the per-pixel one.")
print("B2 against B5 is the alignment, again.")
PYEOF

echo
echo "=== done: reports in ${OUT_DIR} ==="
