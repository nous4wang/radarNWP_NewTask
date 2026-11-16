#!/usr/bin/env bash
# The spectral diagnostic for the corrected runs.
#
# The chain the matrix suggested is: under-prediction flattens the field,
# a flattened field cannot place the convective core, and the score at
# the heavy thresholds collapses. The correlation between the small-scale
# energy a forecast keeps and the score it gets was +0.950 over
# eighteen models -- all of them trained with the benchmark's own
# objective.
#
# These runs break that constraint: they differ in the objective and in
# the head. If the chain holds when the objective changes, it describes
# a relation between the objective and the output rather than a
# property of the architectures.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/rapsd_scan3}"
MAX_CASES="${MAX_CASES:-4}"
GPU="${GPU:-4}"

mkdir -p "${OUT_DIR}"

for model in scan_C1 scan_C2; do
    ckpt="$(ls -dt "${REPO_ROOT}/runs/hqma_scan3/${model}"/*/last.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        continue
    fi
    echo "  -> ${model}: ${ckpt}"
    mkdir -p "${OUT_DIR}/${model}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PYTHON}" "${REPO_ROOT}/scripts/rapsd.py" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split test \
            --model simvp_v2_hqma \
            --checkpoint "${ckpt}" \
            --out-dir "${OUT_DIR}/${model}" \
            --max-cases "${MAX_CASES}" \
            --device cuda \
            > "${OUT_DIR}/${model}.log" 2>&1
    echo "  [${model}] $(grep -E 'high-k band' "${OUT_DIR}/${model}.log" | tail -1)"
done

echo "done"
