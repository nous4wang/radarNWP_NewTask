#!/usr/bin/env bash
# Retrain the baselines whose earlier runs were invalid.
#
# Two defects made the earlier runs of these six meaningless:
#
#   * their output head ended in a hard nn.ReLU(), which killed the
#     gradient at every pixel and left the model predicting a constant
#     field (convlstm scored exactly 0.000);
#   * the NWP fields were not normalised, so the large variables
#     swamped the first convolution.
#
# The runs are redone with the fixed pipeline, on the GPUs that are free.
set -uo pipefail

cd /home/wf/radarNWP
PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
OUT=${OUT:-/home/wf/radarNWP/runs/fixed}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-16}
GPUS=${GPUS:-"2 3 4 5 6 7"}
MODELS=${MODELS:-"convlstm earthformer predrnn simvp_v2 simvp_early_concat simvp_radar_only"}

mkdir -p "${OUT}"

GPU_LIST=(${GPUS})

gpu_is_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "${1}" 2>/dev/null || echo 0)"
    [ "${used:-0}" -lt 100 ]
}

echo "=============================================================="
echo " 重训：修复了输出头与 NWP 归一化的基线"
echo "=============================================================="
echo "  models : ${MODELS}"
echo "  gpus   : ${GPU_LIST[*]}"
echo "  epochs : ${EPOCHS}"
echo

i=0
for model in ${MODELS}; do
    gpu=""
    for _ in "${!GPU_LIST[@]}"; do
        candidate="${GPU_LIST[$(( (i + ${_}) % ${#GPU_LIST[@]} ))]}"
        if gpu_is_free "${candidate}"; then
            gpu="${candidate}"
            break
        fi
    done
    if [ -z "${gpu}" ]; then
        echo "  -> ${model}: no free GPU, skipped"
        continue
    fi

    mkdir -p "${OUT}/${model}"
    echo "  -> ${model} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
    "${PYTHON}" scripts/train.py \
        --model "${model}" \
        --data-root "${DATA}" \
        --packed \
        --split train \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH}" \
        --num-workers 6 \
        --patch-size 64 \
        --slide-stride 64 \
        --lr 3e-4 \
        --log-every 200 \
        --device cuda \
        --out-dir "${OUT}/${model}" \
        > "${OUT}/${model}.log" 2>&1 &
    i=$(( i + 1 ))
    sleep 3
done
wait

echo
echo "all retrains finished"
for model in ${MODELS}; do
    echo "--- ${model} ---"
    grep -E '^epoch [0-9]+:' "${OUT}/${model}.log" 2>/dev/null | tail -5
done
