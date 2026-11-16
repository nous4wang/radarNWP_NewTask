#!/usr/bin/env bash
# The A/B that the paper's central claim rests on, re-run on the corrected
# pipeline.
#
# Two things changed since the previous run and both of them matter:
#
#   * the NWP normalisation constants were re-measured (norm-v2). The
#     divergence fields, which the plugin reads, had their standard
#     deviation overstated by a factor of 2.7 to 4.6, so the plugin had
#     been reading a third of the signal it was built to read.
#   * the history window is built from timestamps rather than positions, so
#     the lead times mean what they say.
#
# Both models run on the same cards, at the same time, from the same code:
# the only difference is whether the coupling plugin is mounted.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/ab_v2}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

mkdir -p "${OUT_DIR}"

train_one() {
    local model="$1" gpu="$2"
    local rundir="${OUT_DIR}/${model}"
    local log="${OUT_DIR}/${model}.log"

    mkdir -p "${rundir}"
    echo "  -> ${model} on GPU ${gpu} (log ${log})"
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
        --log-every 200 \
        --device cuda \
        --out-dir "${rundir}" \
        > "${log}" 2>&1
    echo "  [${model}] finished: $(grep -E '^epoch [0-9]+:' "${log}" | tail -1)"
}

echo "=== training (${EPOCHS} epochs each, in parallel) ==="
train_one simvp_v2 "${GPU_A}" &
A=$!
train_one simvp_v2_quantum "${GPU_B}" &
B=$!
wait $A $B

echo
echo "=== evaluating on the clean test split ==="
for model in simvp_v2 simvp_v2_quantum; do
    ckpt="$(ls -dt "${OUT_DIR}/${model}"/*/last.pt 2>/dev/null | head -1)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU_A}" \
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
        --device cuda \
        --out-json "${OUT_DIR}/${model}.eval.json" \
        > "${OUT_DIR}/${model}.eval.log" 2>&1
    echo "  [${model}] evaluated"
done

echo
echo "=== done ==="
