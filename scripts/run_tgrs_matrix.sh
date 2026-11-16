#!/usr/bin/env bash
# The full comparison matrix for the paper.
#
# Four tiers, run on the clean test split with the tiles stitched back into
# the full 301x301 field:
#
#   tier 1  physics extrapolation      pysteps_adv
#   tier 2  classic spatio-temporal    convlstm predrnn simvp_v2 earthformer
#   tier 3  multimodal SOTA            metnet3 diffcast cascast_nwp mocast
#                                     pw_foucast fusioncast mmf_rnn furecast
#   tier 4  the Q-NWP group            the same backbones with the coupling
#                                     plugin mounted
#
# The training runs are spread over the free cards; the evaluation runs one
# model per card as well, because the evaluation is the same kind of work
# as the training and doing them one at a time leaves seven cards idle.
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

# The models this sweep covers. The plugin-mounted twins are named
# explicitly: they are the same backbone with one extra line, which is what
# makes the comparison a controlled one.
MODELS=(
    pysteps_adv
    convlstm
    predrnn
    simvp_v2
    earthformer
    simvp_v2_quantum
    simvp_v2_nwp_concat
    simvp_v2_nwp_xattn
    simvp_v2_nwp_delta
    metnet3
    diffcast
    cascast_nwp
    mocast
    pw_foucast
    fusioncast
    mmf_rnn
    furecast
)

mkdir -p "${OUT_DIR}"

gpu_is_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1)"
    [[ -z "${used}" ]] && return 1
    [[ "${used}" -lt 100 ]]
}

latest_checkpoint() {
    # The glob does not match when the model has not been trained yet, and
    # under "set -e" a failed command substitution would take the whole
    # script down. The empty result is the answer, not an error.
    ls -dt "$1"/*/last.pt 2>/dev/null | head -1 || true
}

train_one() {
    local model="$1" gpu="$2"
    local rundir="${OUT_DIR}/${model}"
    local log="${OUT_DIR}/${model}.log"

    mkdir -p "${rundir}"
    echo "  -> ${model} on GPU ${gpu}"
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
}

# --------------------------------------------------------------------------
# phase 1: training, spread over the free cards
# --------------------------------------------------------------------------
echo "=== phase 1: training ==="
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
        echo "  -> ${model}: no free card, skipped (re-run to pick it up)"
        continue
    fi

    train_one "${model}" "${gpu}" &
    i=$(( i + 1 ))
    sleep 3
done
wait
echo
echo "all training finished"

# --------------------------------------------------------------------------
# phase 2: evaluation on the clean test split, one model per card
# --------------------------------------------------------------------------
echo
echo "=== phase 2: evaluation (clean test split) ==="
i=0
for model in "${MODELS[@]}"; do
    ckpt="$(latest_checkpoint "${OUT_DIR}/${model}")"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        continue
    fi
    gpu="${CARDS[$(( i % ${#CARDS[@]} ))]}"
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
            --device cuda \
            --out-json "${OUT_DIR}/${model}.eval.json" \
            > "${OUT_DIR}/${model}.eval.log" 2>&1
        echo "  [${model}] evaluated"
    ) &
    i=$(( i + 1 ))
    sleep 2
done
wait

# --------------------------------------------------------------------------
# phase 3: the robustness split -- the cases with a hole in their radar
# --------------------------------------------------------------------------
echo
echo "=== phase 3: evaluation (robustness_missing split) ==="
i=0
for model in "${MODELS[@]}"; do
    ckpt="$(latest_checkpoint "${OUT_DIR}/${model}")"
    if [[ -z "${ckpt}" ]]; then
        continue
    fi
    gpu="${CARDS[$(( i % ${#CARDS[@]} ))]}"
    (
        CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
            --model "${model}" \
            --checkpoint "${ckpt}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split robustness_missing \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --device cuda \
            --out-json "${OUT_DIR}/${model}.robustness.json" \
            > "${OUT_DIR}/${model}.robustness.log" 2>&1
        echo "  [${model}] robustness evaluated"
    ) &
    i=$(( i + 1 ))
    sleep 2
done
wait

echo
echo "=== done: reports in ${OUT_DIR} ==="
