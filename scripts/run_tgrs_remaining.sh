#!/usr/bin/env bash
# Train the models the main sweep did not reach, in parallel on the free
# cards.
#
# The main sweep trains in one batch and then waits for the whole batch
# before it looks at the next model, so a slow model (predrnn takes ten
# minutes an epoch) holds up eight models that could have been training
# meanwhile. This script takes the list of models that are still missing a
# checkpoint and trains them across whatever cards are free, which is what
# the sweep's own dispatcher does within a batch but not across batches.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/tgrs}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

read -r -a CARDS <<< "${GPUS}"

MODELS=(
    metnet3 diffcast cascast_nwp mocast
    pw_foucast fusioncast mmf_rnn furecast
)

gpu_is_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1 || true)"
    [[ -z "${used}" ]] && return 1
    [[ "${used}" -lt 100 ]]
}

latest_checkpoint() {
    ls -dt "$1"/*/last.pt 2>/dev/null | head -1 || true
}

mkdir -p "${OUT_DIR}"

echo "=== training the remaining models ==="
i=0
for model in "${MODELS[@]}"; do
    ckpt="$(latest_checkpoint "${OUT_DIR}/${model}")"
    if [[ -n "${ckpt}" ]]; then
        echo "  -> ${model}: already trained, skipped"
        continue
    fi

    gpu=""
    for offset in "${!CARDS[@]}"; do
        candidate="${CARDS[$(( (i + offset) % ${#CARDS[@]} ))]}"
        if gpu_is_free "${candidate}"; then
            gpu="${candidate}"
            break
        fi
    done
    if [[ -z "${gpu}" ]]; then
        echo "  -> ${model}: no free card, skipped"
        continue
    fi

    rundir="${OUT_DIR}/${model}"
    log="${OUT_DIR}/${model}.log"
    mkdir -p "${rundir}"
    echo "  -> ${model} on GPU ${gpu}"
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
            --log-every 200 \
            --device cuda \
            --out-dir "${rundir}" \
            > "${log}" 2>&1
        echo "  [${model}] $(grep -E '^epoch [0-9]+:' "${log}" | tail -1)"
    ) &
    i=$(( i + 1 ))
    sleep 4
done
wait
echo "all remaining training finished"
