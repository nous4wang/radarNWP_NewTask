#!/usr/bin/env bash
# Stage three: the semi-Lagrangian advection, with its three ablation arms.
#
# The arms share the weights and differ only in which parts are switched
# on:
#   full   the advection and the residual, with the gate between them
#   adv    the advection alone
#   res    the residual alone
#
# The comparison that matters is against the inversion baseline, not
# against the plain concatenation: the advection model shares the
# backbone of the inversion baseline and adds three small heads, so the
# two differ in the extrapolation and in almost nothing else. The
# concatenation baseline has three times the parameters and is a
# different model.
#
# The gate weight and the mass ratio are reported by the model itself on
# every forward pass, so "the model uses the physics" is a measurement.
# A model that sets the gate to zero has become the residual arm and its
# score has to be read as such.
#
# Every run keeps three epochs: the weights of an epoch do not depend on
# the training budget, so this costs a sixth of twenty and gives the
# same epoch_00, epoch_01 and epoch_02.
set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-runs/four_tiers/r4_lagrangian}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-3}"
GPU_BASE="${GPU_BASE:-0}"
GPU_COUNT="${GPU_COUNT:-8}"

mkdir -p "${OUT_DIR}"

run_arm() {
    local tag="$1"; shift
    local kwargs="$1"; shift
    local seed="$1"; shift
    local out="${OUT_DIR}/${tag}_s${seed}"
    if [ -f "${out}/.done" ]; then
        echo "skip ${tag} seed ${seed}: already done"
        return 0
    fi
    local gpu=$(( (seed + ${#tag} + GPU_BASE) % GPU_COUNT ))
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_keep_epochs.py \
        --model simvp_v2_lagrangian \
        --model-kwargs "${kwargs}" \
        --data-root "${DATA_ROOT}" --packed --split train \
        --epochs "${EPOCHS}" --batch-size 16 --num-workers 4 \
        --patch-size 64 --slide-stride 64 \
        --loss balanced --alpha-csi 0.5 \
        --seed "${seed}" \
        --out-dir "${out}" > "${OUT_DIR}/${tag}_s${seed}.log" 2>&1
    local code=$?
    if [ "${code}" -eq 0 ]; then
        touch "${out}/.done"
        echo "done ${tag} seed ${seed}"
    else
        echo "FAILED ${tag} seed ${seed} (exit ${code})"
        tail -5 "${OUT_DIR}/${tag}_s${seed}.log"
    fi
    return 0
}

echo "=========================================================="
echo " stage three: the advection model and its ablation arms"
echo "=========================================================="
for seed in ${SEEDS}; do
    run_arm "full" '{"use_advection": true, "use_source_sink": true}' "${seed}" &
    while [ "$(jobs -rp | wc -l)" -ge "${GPU_COUNT}" ]; do wait -n; done
done
wait
for seed in ${SEEDS}; do
    run_arm "adv" '{"use_advection": true, "use_source_sink": false}' "${seed}" &
    while [ "$(jobs -rp | wc -l)" -ge "${GPU_COUNT}" ]; do wait -n; done
done
wait
for seed in ${SEEDS}; do
    run_arm "res" '{"use_advection": false, "use_source_sink": true}' "${seed}" &
    while [ "$(jobs -rp | wc -l)" -ge "${GPU_COUNT}" ]; do wait -n; done
done
wait

echo "=========================================================="
echo " stage three: training done, scoring"
echo "=========================================================="
"${PYTHON}" tools/_eval_four_tiers.py "${OUT_DIR}" --epochs "${SCORE_EPOCHS:-1}"
echo "STAGE3_EXIT=$?"
