#!/usr/bin/env bash
# Route P2: the epoch curve and the seed spread of the QPE-QPF arms.
#
# Two questions, two scripts' worth of work.
#
# 1. Where does the score peak?  The single-model backbone peaks at epoch 2
#    (CSI@2.0 = 0.153) and decays afterwards, and the QPE-QPF split holds
#    its score across the lead times rather than losing it. If the two
#    properties compose, the split should have an early peak that the
#    backbone does not, and the epoch at which it sits is a property of the
#    architecture rather than of the training budget.
#
# 2. Is the difference larger than the run-to-run noise?  The same
#    configuration trained twice has been measured at 1.78x on CSI@2.0, so
#    a single pair of numbers cannot decide anything. Three seeds per
#    configuration, with a bootstrap interval over the evaluation windows,
#    is what decides it.
#
# The acceptance criterion is written before the run, per the protocol:
#
#   The criterion is the persistence floor, CSI@2.0 = 0.0924.  A
#   configuration that reaches it at any epoch has learned to place the
#   heavy rain; one that does not has not. The number 0.16 to 0.20 is
#   recorded as an expectation, not as a criterion: it is not derived from
#   anything, and a criterion that is fitted to the result afterwards is
#   not a criterion.
#
# Everything trains on the remote; this script only dispatches.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p2}"
EPOCHS="${EPOCHS:-20}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Part 1: the epoch curve.
#
# One seed, three arms, every epoch kept. The wrapper keeps a copy of each
# epoch's weights, so the curve is a measurement of the same run rather
# than of three separate runs.
# ---------------------------------------------------------------------------
CURVE_ARMS=(
    "qpe_0 simvp_v2_qpe_guided 0"
    "qpe_05 simvp_v2_qpe_guided 1"
    "base_concat simvp_v2_nwp_concat 2"
)

echo "=========================================================="
echo " P2 part 1: the epoch curve (seed 0, every epoch kept)"
echo " criterion: CSI@2.0 > 0.0924 (the persistence floor)"
echo "=========================================================="

CURVE_DIR="${OUT_DIR}/curve"
mkdir -p "${CURVE_DIR}"

for entry in "${CURVE_ARMS[@]}"; do
    read -r name model gpu <<< "${entry}"
    log="${CURVE_DIR}/${name}.log"
    echo "  -> ${name} (${model}) on GPU ${gpu}"
    (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS=4 \
        "${PYTHON}" scripts/train_keep_epochs.py \
            --model "${model}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split train \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --device cuda \
            --out-dir "${CURVE_DIR}/${name}" \
            --loss balanced \
            --alpha-csi "${ALPHA_CSI}" \
            --tier-thresholds ${TIERS} \
            --qpe-weight "$([ "${name}" = "qpe_05" ] && echo 0.5 || echo 0.0)" \
            > "${log}" 2>&1
    ) &
done
wait

echo "training done; evaluating every kept epoch"

# The evaluation is the expensive part: each epoch is a full pass over the
# stitched field, and the arms run in parallel across the free cards.
for entry in "${CURVE_ARMS[@]}"; do
    read -r name model gpu <<< "${entry}"
    run_dir="$(ls -dt "${CURVE_DIR}/${name}"/*/ 2>/dev/null | head -1)"
    if [[ -z "${run_dir}" ]]; then
        echo "  ${name}: no run directory, skipped"
        continue
    fi
    mkdir -p "${CURVE_DIR}/${name}_eval"
    for ck in "${run_dir}"epoch_*.pt; do
        [[ -f "${ck}" ]] || continue
        base="$(basename "${ck}" .pt)"
        (
            cd "${REPO_ROOT}"
            CUDA_VISIBLE_DEVICES="${gpu}" \
            "${PYTHON}" scripts/evaluate.py \
                --model "${model}" \
                --checkpoint "${ck}" \
                --data-root "${DATA_ROOT}" \
                --packed --split test \
                --batch-size "${BATCH_SIZE}" --num-workers "${WORKERS}" \
                --patch-size "${PATCH_SIZE}" --slide-stride "${SLIDE_STRIDE}" \
                --thresholds ${TIERS} \
                --device cuda \
                --out-json "${CURVE_DIR}/${name}_eval/${base}.eval.json" \
                > "${CURVE_DIR}/${name}_eval/${base}.eval.log" 2>&1
        )
    done
done

# ---------------------------------------------------------------------------
# Part 2: the seed spread.
#
# Three seeds for the two configurations the paper compares: the QPE-QPF
# split and the plain concatenation it has to beat. The per-window tables
# are written as well, because that is what the bootstrap resamples.
# ---------------------------------------------------------------------------
SEED_ARMS=(
    "qpe_0 simvp_v2_qpe_guided"
    "base_concat simvp_v2_nwp_concat"
)

echo "=========================================================="
echo " P2 part 2: ${SEEDS} seeds per configuration"
echo "=========================================================="

SEED_DIR="${OUT_DIR}/seeds"
mkdir -p "${SEED_DIR}"

for entry in "${SEED_ARMS[@]}"; do
    read -r name model <<< "${entry}"
    for seed in ${SEEDS}; do
        gpu=$(( (seed + 2) % 8 ))
        log="${SEED_DIR}/${name}_s${seed}.log"
        echo "  -> ${name} seed ${seed} on GPU ${gpu}"
        (
            cd "${REPO_ROOT}"
            CUDA_VISIBLE_DEVICES="${gpu}" \
            OMP_NUM_THREADS=4 \
            "${PYTHON}" scripts/train.py \
                --model "${model}" \
                --data-root "${DATA_ROOT}" \
                --packed \
                --split train \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --num-workers "${WORKERS}" \
                --patch-size "${PATCH_SIZE}" \
                --slide-stride "${SLIDE_STRIDE}" \
                --device cuda \
                --seed "${seed}" \
                --out-dir "${SEED_DIR}/${name}_s${seed}" \
                --loss balanced \
                --alpha-csi "${ALPHA_CSI}" \
                --tier-thresholds ${TIERS} \
                > "${log}" 2>&1
        ) &
        # Three at a time: eight concurrent runs on eight cards would put
        # three per card and the epoch time would triple.
        while [[ $(jobs -rp | wc -l) -ge 3 ]]; do sleep 5; done
    done
done
wait

echo "training done; scoring every seed with the per-window tables"

for entry in "${SEED_ARMS[@]}"; do
    read -r name model <<< "${entry}"
    for seed in ${SEEDS}; do
        run_dir="$(ls -dt "${SEED_DIR}/${name}_s${seed}"/*/ 2>/dev/null | head -1)"
        [[ -n "${run_dir}" ]] || continue
        ck="${run_dir}last.pt"
        [[ -f "${ck}" ]] || continue
        (
            cd "${REPO_ROOT}"
            CUDA_VISIBLE_DEVICES=0 \
            "${PYTHON}" scripts/evaluate.py \
                --model "${model}" \
                --checkpoint "${ck}" \
                --data-root "${DATA_ROOT}" \
                --packed --split test \
                --batch-size "${BATCH_SIZE}" --num-workers "${WORKERS}" \
                --patch-size "${PATCH_SIZE}" --slide-stride "${SLIDE_STRIDE}" \
                --thresholds ${TIERS} \
                --device cuda \
                --out-json "${SEED_DIR}/${name}_s${seed}.eval.json" \
                --per-window-out "${SEED_DIR}/${name}_s${seed}.windows.json" \
                > "${SEED_DIR}/${name}_s${seed}.eval.log" 2>&1
        ) &
        while [[ $(jobs -rp | wc -l) -ge 4 ]]; do sleep 5; done
    done
done
wait

echo "=========================================================="
echo " P2 done. Next: tools/_analyze_p2.py"
echo "=========================================================="
