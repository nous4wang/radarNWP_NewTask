#!/usr/bin/env bash
# Fill the two models that the paper matrix never reached.
#
# run_tgrs_matrix.sh and run_tgrs_remaining.sh both walk their model list
# once, in a single pass, and skip a model when every card happens to be
# busy at that moment. mmf_rnn and furecast were skipped by both, so
# nothing on the server was going to train them: the matrix would have
# stopped at 15 of the 17, and the paper's multimodal-SOTA tier would have
# had a hole in it. A model that is silently not run is a hole in the
# final table, so this script runs the two that fell through.
#
# They are trained into a staging directory rather than straight into
# runs/tgrs/, and the reason matters. train.py writes last.pt after every
# epoch, so a model that is still training already has a checkpoint, and
# the matrix driver's evaluation phase would happily pick that up and
# score it -- half a model, reported as if it were the whole one.
# Staging keeps runs/tgrs/mmf_rnn and runs/tgrs/furecast empty until the
# two are actually finished; the checkpoints are moved into place
# afterwards and the evaluation runs against the moved ones.
#
# The evaluation is the one run_tgrs_matrix.sh would have run for these
# two models had it not skipped them: the clean test split for the main
# table and the robustness_missing split for the fault-tolerance table,
# both with the tiles stitched back into the 301x301 field.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
STAGE_DIR="${STAGE_DIR:-${REPO_ROOT}/runs/tgrs_gap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/tgrs}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"

# The three models, and the card each one goes on.
#
# mmf_rnn and furecast fell through the matrix's one-pass model loop
# because every card was busy when the loop reached them. simvp_radar_only
# was never in that loop at all, but paper_tables.py prints it as the M0
# rung of the ablation ladder -- the radar-only twin of the SimVP-v2 core
# -- and without it that row is empty and the ladder has no anchor.
#
# The driver's evaluation phase assigns cards round-robin over the models
# that have a checkpoint, so it puts one or two evaluations on every card
# while these are training. A card holds a training job and two
# evaluations without trouble on a 24 GB card.
MODELS="${MODELS:-mmf_rnn furecast simvp_radar_only}"
MODEL_GPUS="${MODEL_GPUS:-mmf_rnn:0 furecast:7 simvp_radar_only:1}"

mkdir -p "${STAGE_DIR}"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

gpu_is_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "$1" 2>/dev/null || echo 0)"
    [ "${used:-0}" -lt 100 ]
}

latest_checkpoint() {
    ls -dt "$1"/*/last.pt 2>/dev/null | head -1 || true
}

train_one() {
    local model="$1" gpu="$2"
    local rundir="${STAGE_DIR}/${model}"
    local log_file="${STAGE_DIR}/${model}.log"

    if [ -n "$(latest_checkpoint "${rundir}")" ]; then
        log "  ${model}: already trained in the staging area, skipped"
        return 0
    fi

    # Wait for the card rather than skipping it.
    local waited=0
    while ! gpu_is_free "${gpu}"; do
        if [ "${waited}" -eq 0 ]; then
            log "  ${model}: GPU ${gpu} is busy, waiting ..."
        fi
        sleep 60
        waited=$(( waited + 1 ))
        if [ "${waited}" -ge 120 ]; then
            log "  ${model}: GPU ${gpu} still busy after 2 h -- skipped"
            return 1
        fi
    done

    mkdir -p "${rundir}"
    log "  ${model}: training on GPU ${gpu} (log: ${log_file})"
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
        > "${log_file}" 2>&1
    local status=$?
    log "  ${model}: training finished with status ${status}"
    if [ "${status}" -ne 0 ]; then
        tail -5 "${log_file}" | sed 's/^/      /'
        return 1
    fi
    log "  ${model}: $(grep -E '^epoch [0-9]+:' "${log_file}" | tail -1)"
    return 0
}

# --------------------------------------------------------------------------
# phase 1: train the two models, one per card
# --------------------------------------------------------------------------
log "=== phase 1: training the models the matrix skipped ==="
train_pids=()
for pair in ${MODEL_GPUS}; do
    model="${pair%%:*}"
    gpu="${pair##*:}"
    train_one "${model}" "${gpu}" &
    train_pids+=( $! )
    sleep 5
done

train_failed=0
for pid in "${train_pids[@]}"; do
    wait "${pid}" || train_failed=1
done
log "=== training done (failures: ${train_failed}) ==="

# --------------------------------------------------------------------------
# phase 2: wait for the matrix's own driver to exit
#
# Its evaluation phase walks the same model list and picks up whatever
# checkpoint it finds. Promoting into runs/tgrs/ while that is still
# running would put the two models into the driver's loop, where they
# would be evaluated twice and the two runs would race on the same file.
# --------------------------------------------------------------------------
log "=== phase 2: waiting for the matrix driver to exit ==="
waited=0
while pgrep -f 'scripts/run_tgrs_matrix.sh' >/dev/null 2>&1 \
   || pgrep -f 'scripts/run_tgrs_remaining.sh' >/dev/null 2>&1; do
    if [ "${waited}" -eq 0 ]; then
        log "  the matrix driver is still running, waiting ..."
    fi
    sleep 60
    waited=$(( waited + 1 ))
    if [ "${waited}" -ge 360 ]; then
        log "  the driver has not exited after 6 h -- continuing anyway"
        break
    fi
done
log "=== the matrix driver has exited ==="

# --------------------------------------------------------------------------
# phase 3: move the finished runs into runs/tgrs/
# --------------------------------------------------------------------------
log "=== phase 3: promoting the trained runs into ${OUT_DIR} ==="
for model in ${MODELS}; do
    ckpt="$(latest_checkpoint "${STAGE_DIR}/${model}")"
    if [ -z "${ckpt}" ]; then
        log "  ${model}: no checkpoint in the staging area -- skipped"
        continue
    fi
    rundir="$(dirname "${ckpt}")"
    epochs_done=0
    if [ -f "${rundir}/history.json" ]; then
        epochs_done="$(grep -c '"epoch":' "${rundir}/history.json" || true)"
    fi
    if [ "${epochs_done}" -lt "${EPOCHS}" ]; then
        log "  ${model}: history holds ${epochs_done} epochs, expected ${EPOCHS} -- skipped"
        continue
    fi
    mkdir -p "${OUT_DIR}/${model}"
    target="${OUT_DIR}/${model}/$(basename "${rundir}")"
    if [ -e "${target}" ]; then
        log "  ${model}: already promoted to ${target} -- skipped"
        continue
    fi
    mv "${rundir}" "${target}"
    log "  ${model}: promoted to ${target}"
done

# --------------------------------------------------------------------------
# phase 4: evaluate the two models, on both splits
# --------------------------------------------------------------------------
log "=== phase 4: evaluating ==="
eval_one() {
    local model="$1" split="$2" suffix="$3" gpu="$4"
    local ckpt
    ckpt="$(latest_checkpoint "${OUT_DIR}/${model}")"
    if [ -z "${ckpt}" ]; then
        log "  ${model}: no checkpoint -- ${split} skipped"
        return 1
    fi
    local out="${OUT_DIR}/${model}${suffix}.json"
    if [ -e "${out}" ]; then
        log "  ${model}: ${out} already exists -- skipped"
        return 0
    fi

    local waited=0
    while ! gpu_is_free "${gpu}"; do
        sleep 30
        waited=$(( waited + 1 ))
        if [ "${waited}" -ge 60 ]; then
            log "  ${model}: GPU ${gpu} still busy before the ${split} evaluation -- skipped"
            return 1
        fi
    done

    log "  ${model}: ${split} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split "${split}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        --slide-stride "${SLIDE_STRIDE}" \
        --device cuda \
        --out-json "${out}" \
        > "${OUT_DIR}/${model}${suffix}.log" 2>&1
    local status=$?
    if [ "${status}" -ne 0 ]; then
        log "  ${model}: ${split} failed with status ${status}"
        tail -5 "${OUT_DIR}/${model}${suffix}.log" | sed 's/^/      /'
        return 1
    fi
    log "  ${model}: ${split} done -- $(grep -E '^\s+0\.5 mm' "${OUT_DIR}/${model}${suffix}.log" | head -1)"
    return 0
}

clean_pids=()
slot=0
for model in ${MODELS}; do
    eval_one "${model}" test ".eval" "${slot}" &
    clean_pids+=( $! )
    slot=$(( (slot + 1) % 8 ))
    sleep 3
done
for pid in "${clean_pids[@]}"; do
    wait "${pid}" || log "  one of the clean evaluations failed"
done

robust_pids=()
slot=0
for model in ${MODELS}; do
    eval_one "${model}" robustness_missing ".robustness" "${slot}" &
    robust_pids+=( $! )
    slot=$(( (slot + 1) % 8 ))
    sleep 3
done
for pid in "${robust_pids[@]}"; do
    wait "${pid}" || log "  one of the robustness evaluations failed"
done

# --------------------------------------------------------------------------
# phase 5: the table, and a count of what exists
# --------------------------------------------------------------------------
log "=== phase 5: building the paper tables ==="
"${PYTHON}" "${REPO_ROOT}/scripts/paper_tables.py" \
    --runs-dir "${OUT_DIR}" \
    --markdown "${REPO_ROOT}/docs/TGRS_TABLES.md" \
    > "${REPO_ROOT}/runs/tgrs_gap_tables.txt" 2>&1
log "  tables written to ${REPO_ROOT}/docs/TGRS_TABLES.md"

log "=== counts ==="
log "  eval reports:       $(ls "${OUT_DIR}"/*.eval.json 2>/dev/null | wc -l)"
log "  robustness reports: $(ls "${OUT_DIR}"/*.robustness.json 2>/dev/null | wc -l)"
log "=== done ==="
