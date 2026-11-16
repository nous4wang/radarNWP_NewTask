#!/usr/bin/env bash
# Evaluate the runs of a scan that has already been trained.
#
# The sweep script trains and then evaluates, and re-running it would
# train again. This is the evaluation half on its own, for the case
# where the training is finished and the evaluation needs to be redone.
#
#   scripts/evaluate_scan.sh <scan-directory> [model-per-run-file]
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${1:?the scan directory is required}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
GPUS="${GPUS:-0 1 2 3}"

# Which model each run uses. A file with one "run model" pair per line
# overrides the guess from the run's own name.
MODEL_FILE="${2:-}"

declare -A MODELS
if [[ -n "${MODEL_FILE}" && -f "${MODEL_FILE}" ]]; then
    while read -r run model; do
        [[ -z "${run}" ]] && continue
        MODELS["${run}"]="${model}"
    done < "${MODEL_FILE}"
fi

mkdir -p "${OUT_DIR}"

index=0
for rundir in "${OUT_DIR}"/scan_*; do
    [[ -d "${rundir}" ]] || continue
    run="$(basename "${rundir}")"
    ckpt="$(ls -dt "${rundir}"/*/last.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${run}] no checkpoint -- skipped"
        continue
    fi
    model="${MODELS[${run}]:-}"
    if [[ -z "${model}" ]]; then
        # The run directory is named after the run, not the model, so the
        # model is read from the training record.
        model="$(/home/wf/.conda/envs/radar_nwp/bin/python - "${ckpt}" <<'PYEOF'
import json
import sys
from pathlib import Path

cfg = json.loads((Path(sys.argv[1]).parent / "config.json").read_text())
print(cfg.get("model", ""))
PYEOF
)"
    fi
    gpu="${GPUS[${index}]:-0}"
    index=$((index + 1))
    echo "  -> ${run} (${model}) on GPU ${gpu}"
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
        echo "  [${run}] $(grep -cE 'CSI|threshold' "${OUT_DIR}/${run}.eval.log" >/dev/null && echo evaluated || echo FAILED)"
    ) &
    sleep 3
done
wait

echo "=== done ==="
