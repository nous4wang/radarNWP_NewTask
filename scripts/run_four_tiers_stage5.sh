#!/usr/bin/env bash
# Stage five: the lag sweep with the decoupled arm, and the missing-frame
# pairing.
#
# The lag sweep is re-run because the no-observation arm was a copy of
# the inversion arm: with a lag of eleven frames and a history of eleven,
# the observation branch was unreachable, so that arm's weights were bit
# for bit identical to the inversion arm's. A sweep with one arm
# duplicated has three distinct arms in it, and it cannot say what an
# observation is worth.
#
# The criterion is not monotonicity. The evidence so far points the other
# way -- twelve minutes of delay beating twenty-four holds on four
# epochs and on two of three seeds -- so requiring a strict order would
# be fitting the criterion to the answer. The criterion is that the
# paired interval excludes zero, and that the order of the arms is the
# same across seeds. If the order moves with the seed, that is reported
# as the answer.
#
# The missing-frame pairing uses one checkpoint and switches only the
# interpolation, so the two runs differ by the fault tolerance and by
# nothing else. The split holds the twenty-three cases with a hole in
# the record, which is a small sample, so the per-lead differences and
# their intervals are reported rather than a single total.
set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-runs/four_tiers/r2_decoupled}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-3}"
ROBUST_DIR="${ROBUST_DIR:-runs/four_tiers/robustness_pairing}"
PAIR_CKPT="${PAIR_CKPT:-}"

mkdir -p "${OUT_DIR}" "${ROBUST_DIR}"

run_lag() {
    local lag="$1"; shift
    local seed="$1"; shift
    local out="${OUT_DIR}/lag${lag}_s${seed}"
    if [ -f "${out}/.done" ]; then
        echo "skip lag ${lag} seed ${seed}: already done"
        return 0
    fi
    local gpu=$(( (lag + seed) % 8 ))
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_keep_epochs.py \
        --model simvp_v2_assimilated \
        --model-kwargs "{\"obs_lag_steps\": ${lag}}" \
        --data-root "${DATA_ROOT}" --packed --split train \
        --epochs "${EPOCHS}" --batch-size 16 --num-workers 4 \
        --patch-size 64 --slide-stride 64 \
        --loss balanced --alpha-csi 0.5 --qpe-weight 0.0 \
        --seed "${seed}" \
        --out-dir "${out}" > "${OUT_DIR}/lag${lag}_s${seed}.log" 2>&1
    local code=$?
    if [ "${code}" -eq 0 ]; then
        touch "${out}/.done"
        echo "done lag ${lag} seed ${seed}"
    else
        echo "FAILED lag ${lag} seed ${seed} (exit ${code})"
        tail -5 "${OUT_DIR}/lag${lag}_s${seed}.log"
    fi
    return 0
}

echo "=========================================================="
echo " stage five: the lag sweep with the decoupled arm"
echo "=========================================================="
for seed in ${SEEDS}; do
    for lag in 0 2 4 11; do
        run_lag "${lag}" "${seed}" &
        while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n; done
    done
done
wait

echo "=========================================================="
echo " stage five: scoring the lag sweep"
echo "=========================================================="
"${PYTHON}" tools/_eval_four_tiers.py "${OUT_DIR}" --epochs "${SCORE_EPOCHS:-1}"
echo "LAGEVAL_EXIT=$?"

echo "=========================================================="
echo " stage five: the missing-frame pairing, one checkpoint, two switches"
echo "=========================================================="
if [ -z "${PAIR_CKPT}" ]; then
    echo "  PAIR_CKPT is not set, looking for one in the earlier runs"
    PAIR_CKPT="$(ls -dt "${OUT_DIR}"/lag0_s*/*/epoch_00.pt 2>/dev/null | head -1)"
fi
if [ -z "${PAIR_CKPT}" ]; then
    echo "  no checkpoint found, the pairing is skipped"
else
    echo "  using ${PAIR_CKPT}"
    for mode in with_interp no_interp; do
        flag=""
        if [ "${mode}" = "no_interp" ]; then
            flag="--no-interpolate"
        fi
        CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/evaluate.py \
            --model simvp_v2_assimilated \
            --model-kwargs '{"obs_lag_steps": 0}' \
            --checkpoint "${PAIR_CKPT}" \
            --data-root "${DATA_ROOT}" --packed \
            --split robustness_missing \
            --batch-size 16 --num-workers 4 \
            --patch-size 64 --slide-stride 64 \
            --thresholds 0.1 0.5 2.0 4.0 \
            ${flag} --device cuda \
            --out-json "${ROBUST_DIR}/robustness_${mode}.eval.json" \
            > "${ROBUST_DIR}/robustness_${mode}.log" 2>&1
        echo "  ${mode}: exit $?"
    done
fi
echo "STAGE5_EXIT=$?"
