#!/usr/bin/env bash
# Stage two: the neighbourhood fractions term, measured over ten seeds.
#
# The arms differ from the baseline by exactly one thing: the objective.
# The baseline is the score-aware loss that every other result in the
# project was measured under, and the new arm is that same objective with
# the fractions term added on top.
#
# The control arm -- the fractions term with a weight of zero -- is the
# score-aware loss exactly, so it is not trained again: the ten seeds of
# the baseline already exist and are read from the earlier run.
#
# The scale arms exist to show which neighbourhood size carries the
# effect. They get three seeds rather than ten: they are a mechanism
# check, not a performance claim, and the performance claim has to come
# from the main arm.
#
# Every run keeps three epochs. The weights of an epoch do not depend on
# the training budget -- there is no learning-rate schedule and the batch
# order is fixed by the seed -- so three epochs cost a sixth of twenty
# and give the same epoch_00, epoch_01 and epoch_02.
#
# The exit code of each job is collected rather than left to `set -e`:
# one arm failing must not discard the others.
set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-runs/four_tiers/r5_soft_fss}"
MODEL="${MODEL:-simvp_v2_nwp_concat}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2 3 4 5 6 7 8 9}"
SEEDS_SCALE="${SEEDS_SCALE:-0 1 2}"

mkdir -p "${OUT_DIR}"

train_one() {
    local tag="$1"; shift
    local seed="$1"; shift
    local out="${OUT_DIR}/${tag}_s${seed}"
    if [ -f "${out}/.done" ]; then
        echo "skip ${tag} seed ${seed}: already done"
        return 0
    fi
    local gpu
    gpu=$(( (seed + ${#tag}) % 8 ))
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_keep_epochs.py \
        --model "${MODEL}" \
        --data-root "${DATA_ROOT}" --packed --split train \
        --epochs 3 --batch-size 16 --num-workers 4 \
        --patch-size 64 --slide-stride 64 \
        --loss soft_fss --alpha-fss 0.4 \
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
echo " stage two: the fractions term, ten seeds on the main arm"
echo "=========================================================="
for seed in ${SEEDS_MAIN}; do
    train_one "fss35" "${seed}" &
    while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n; done
done
wait

echo "=========================================================="
echo " stage two: the scale arms, three seeds each"
echo "=========================================================="
for seed in ${SEEDS_SCALE}; do
    CUDA_VISIBLE_DEVICES=$(( seed % 8 )) "${PYTHON}" scripts/train_keep_epochs.py \
        --model "${MODEL}" \
        --data-root "${DATA_ROOT}" --packed --split train \
        --epochs 3 --batch-size 16 --num-workers 4 \
        --patch-size 64 --slide-stride 64 \
        --loss soft_fss --alpha-fss 0.4 --fss-scales 3 \
        --seed "${seed}" \
        --out-dir "${OUT_DIR}/scale3_s${seed}" > "${OUT_DIR}/scale3_s${seed}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n; done
done
for seed in ${SEEDS_SCALE}; do
    CUDA_VISIBLE_DEVICES=$(( seed % 8 )) "${PYTHON}" scripts/train_keep_epochs.py \
        --model "${MODEL}" \
        --data-root "${DATA_ROOT}" --packed --split train \
        --epochs 3 --batch-size 16 --num-workers 4 \
        --patch-size 64 --slide-stride 64 \
        --loss soft_fss --alpha-fss 0.4 --fss-scales 5 \
        --seed "${seed}" \
        --out-dir "${OUT_DIR}/scale5_s${seed}" > "${OUT_DIR}/scale5_s${seed}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n; done
done
wait

echo "=========================================================="
echo " stage two: training done, scoring"
echo "=========================================================="
# Only the peak epoch is scored first. The criterion for this stage is
# about the peak -- section 4.12 measured that the peak of every arm is
# at epoch zero -- and scoring three epochs of sixteen runs would take
# three times as long for two epochs that are not the criterion.
SCORE_EPOCHS="${SCORE_EPOCHS:-1}"
echo " scoring epoch 0 only (SCORE_EPOCHS=${SCORE_EPOCHS})"
"${PYTHON}" tools/_eval_four_tiers.py "${OUT_DIR}" --epochs "${SCORE_EPOCHS}"
echo "STAGE2_EXIT=$?"
