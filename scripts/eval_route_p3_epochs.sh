#!/usr/bin/env bash
# Score more epochs of the route P3 peak re-measurement.
#
# ``scripts/run_route_p3_epochs.sh`` trained the four arms and scored
# epoch 0 and epoch 19.  Epoch 0 was assumed to be the peak because
# that is where route P2 found its arms to peak -- but the assumption
# has to be checked on these four arms, not carried over.  A lag curve
# read at the wrong epoch is the mistake this whole re-measurement
# exists to undo.
#
# This script scores whatever is already on disk and changes nothing
# else.  The checkpoints are the ones ``train_keep_epochs.py`` wrote.
#
# The judge is the file count under ``runs/route_p3_epochs``.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p3_epochs}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
SCORE_EPOCHS="${SCORE_EPOCHS:-01 02}"
PARALLEL="${PARALLEL:-8}"

# name  model  gpu  lag
ARMS=(
    "assim_lag0 simvp_v2_assimilated 0 0"
    "assim_lag2 simvp_v2_assimilated 1 2"
    "assim_lag4 simvp_v2_assimilated 2 4"
    "assim_lag11 simvp_v2_assimilated 3 11"
)

mkdir -p "${OUT_DIR}"

running=0
for entry in "${ARMS[@]}"; do
    read -r name model gpu lag <<< "${entry}"
    run_dir="$(ls -dt "${OUT_DIR}/${name}"/*/ 2>/dev/null | head -1)"
    [[ -n "${run_dir}" ]] || continue
    for ep in ${SCORE_EPOCHS}; do
        ck="${run_dir}epoch_${ep}.pt"
        [[ -f "${ck}" ]] || continue
        out="${OUT_DIR}/${name}_ep${ep}.eval.json"
        if [[ -f "${out}" ]] && "${PYTHON}" -c \
            "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null; then
            echo "  ${name} epoch ${ep}: already scored"
            continue
        fi
        (
            cd "${REPO_ROOT}"
            CUDA_VISIBLE_DEVICES="${gpu}" \
            "${PYTHON}" scripts/evaluate.py \
                --model "${model}" \
                --checkpoint "${ck}" \
                --data-root "${DATA_ROOT}" \
                --packed --split test \
                --batch-size "${BATCH_SIZE}" --num-workers "${WORKERS}" \
                --patch-size 64 --slide-stride 64 \
                --thresholds ${TIERS} \
                --device cuda \
                --model-kwargs "{\"obs_lag_steps\":${lag}}" \
                --out-json "${out}" \
                > "${out%.json}.log" 2>&1
        ) &
        running=$(( running + 1 ))
        while [[ $(jobs -rp | wc -l) -ge ${PARALLEL} ]]; do sleep 5; done
    done
done
wait

echo "the judge is the file count:"
for entry in "${ARMS[@]}"; do
    read -r name model gpu lag <<< "${entry}"
    for ep in 00 01 02 19; do
        n=0
        [[ -f "${OUT_DIR}/${name}_ep${ep}.eval.json" ]] && n=$(( n + 1 ))
        echo "  ${name} epoch ${ep}: ${n}"
    done
done
