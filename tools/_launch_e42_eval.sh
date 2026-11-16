#!/usr/bin/env bash
# E42 -- score the fixed A2' arm and reuse E41's B0 weights and tables.
#
# The criteria are frozen (E42 section 2, 2026-11-09 14:22).  The
# evaluator, the windows, the thresholds and the seed set are exactly
# E41's, so A2' is paired with the SAME B0 that E41 scored: the paired
# bootstrap compares A2' against B0 on the same 538 windows.
#
# B0's weights are NOT retrained: E41's batch 1 used the same
# scripts/train.py code path, and this round changed only
# radar_nwp_bench/metrics/struct_losses.py, which the B0 arm (standard
# loss) never touches.  Reusing them keeps the batch same-batch, which
# is what S1/S3 require.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP

PY=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
E41=/home/wf/radarNWP/runs/e41
OUT=/home/wf/radarNWP/runs/e42
EVAL_DIR="${OUT}/evals"
FIELDS="${OUT}/fields"
STAGE="${OUT}/eval_stage.log"
BATCH=16
WORKERS=6
PAR=3
SEEDS="0 1 2"
EPOCH="00"

mkdir -p "${EVAL_DIR}" "${FIELDS}"
: > "${STAGE}"
log() { echo "$1 $(date '+%F %T')" | tee -a "${STAGE}"; }

score_one() {
    local arm="$1" model="$2" kwargs="$3" seed="$4" gpu="$5" train_root="$6" dump="$7"
    local run_dir ck tag out win
    run_dir="$(ls -d "${train_root}/${arm}_s${seed}"/*/ 2>/dev/null | head -1)"
    [ -n "${run_dir}" ] || { log "MISS ${arm} s${seed}: no run dir"; return 0; }
    ck="${run_dir}epoch_${EPOCH}.pt"
    [ -f "${ck}" ] || { log "MISS ${arm} s${seed}: no ${EPOCH} checkpoint"; return 0; }
    tag="${arm}_s${seed}_ep${EPOCH}"
    out="${EVAL_DIR}/${tag}.eval.json"
    win="${EVAL_DIR}/${tag}.windows.json"
    if [ -f "${out}" ] && "${PY}" -c \
        "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null; then
        log "SKIP ${tag}: already scored"
        return 0
    fi
    local extra=()
    if [ "${dump}" = "yes" ]; then
        extra=(--dump-fields "${FIELDS}/${tag}")
    fi
    (
        cd /home/wf/radarNWP || exit 1
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 \
        "${PY}" scripts/evaluate.py \
            --model "${model}" \
            --checkpoint "${ck}" \
            --data-root "${DATA}" \
            --packed --split test \
            --batch-size "${BATCH}" --num-workers "${WORKERS}" \
            --patch-size 64 --slide-stride 64 \
            --thresholds 0.1 0.5 2.0 4.0 \
            --device cuda \
            --model-kwargs "${kwargs}" \
            --out-json "${out}" \
            --per-window-out "${win}" \
            "${extra[@]}" \
            > "${OUT}/logs/${tag}.eval.log" 2>&1
    )
    if [ -f "${out}" ]; then
        log "DONE ${tag}"
    else
        log "FAILED ${tag}"
        tail -n 12 "${OUT}/logs/${tag}.eval.log" >> "${STAGE}"
    fi
}

# Copy E41's B0 tables and fields so that the S3 control and the
# structure metrics compare A2' against the SAME scored B0.
for seed in ${SEEDS}; do
    for src in "${E41}/evals/B0_s${seed}_ep${EPOCH}.eval.json" \
               "${E41}/evals/B0_s${seed}_ep${EPOCH}.windows.json"; do
        [ -f "${src}" ] && cp -p "${src}" "${EVAL_DIR}/" && log "COPY $(basename "${src}")"
    done
    for src in "${E41}/fields/B0_s${seed}_ep${EPOCH}.pred.npy" \
               "${E41}/fields/B0_s${seed}_ep${EPOCH}.truth.npy"; do
        [ -f "${src}" ] && cp -p "${src}" "${FIELDS}/" && log "COPY $(basename "${src}")"
    done
done

slot=0
for seed in ${SEEDS}; do
    gpu=$(( slot % 8 ))
    slot=$(( slot + 1 ))
    log "LAUNCH A2p s${seed} gpu=${gpu}"
    score_one "A2p" "struct_hurdle" '{}' "${seed}" "${gpu}" "${OUT}/train" "yes" &
    while [ "$(jobs -rp | wc -l)" -ge "${PAR}" ]; do sleep 5; done
done
wait

ok=0; bad=0
for seed in ${SEEDS}; do
    for arm in B0 A2p; do
        f="${EVAL_DIR}/${arm}_s${seed}_ep${EPOCH}.eval.json"
        if [ -f "${f}" ]; then ok=$(( ok + 1 )); else bad=$(( bad + 1 )); fi
    done
done
log "E42 EVAL DONE ok=${ok} failed=${bad} epoch=${EPOCH}"
echo "evals: ${EVAL_DIR}; fields: ${FIELDS}; stage: ${STAGE}"
