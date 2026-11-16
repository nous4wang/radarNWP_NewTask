#!/usr/bin/env bash
# Evaluate the score-aware matrix, one model per free card.
#
# Why this exists.  scripts/run_score_matrix.sh evaluates every model in
# phase 2 with CUDA_VISIBLE_DEVICES=0 and all eighteen processes alive at
# the same time.  One evaluation holds about 1.7 GB and the card has
# 24.5 GB, so the tail of the list runs out of memory and never writes
# its report.  The training that produced the checkpoints is the
# expensive part; the report is not worth losing.
#
# The script does the same work spread over whatever cards are free.  It
# keeps going until every model has a report, so a model that finishes
# training late is picked up as soon as its twentieth epoch is on disk
# rather than waiting for the whole matrix to be done.
#
#   bash scripts/run_score_matrix_eval.sh [model ...]
#
# With no arguments it covers every model of the matrix.  It is safe to
# run while the driver is still training.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/score_matrix}"
DRIVER_LOG="${DRIVER_LOG:-${REPO_ROOT}/runs/score_matrix_driver.log}"
LOCK_DIR="${OUT_DIR}/.eval_lock"
CLAIM_DIR="${OUT_DIR}/.eval_claim"

BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
POLL_SECONDS="${POLL_SECONDS:-180}"
MAX_ROUNDS="${MAX_ROUNDS:-80}"

MODELS=(
    simvp_v2 simvp_v2_nwp_concat simvp_v2_quantum simvp_v2_nwp_xattn
    simvp_v2_nwp_delta simvp_early_concat simvp_radar_only convlstm
    predrnn earthformer metnet3 diffcast cascast_nwp mocast pw_foucast
    fusioncast mmf_rnn furecast
)

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
fi

mkdir -p "${OUT_DIR}"

# Only one copy of this script may run at a time.  The lock also makes
# the script safe if the launch is executed twice.
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "another evaluation is already running; leaving it alone"
    exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

free_gpu() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "$1" 2>/dev/null | head -1)"
    [[ -z "${used}" ]] && return 1
    (( used < 100 ))
}

free_gpus() {
    local gpu
    for gpu in 0 1 2 3 4 5 6 7; do
        if free_gpu "${gpu}"; then
            echo "${gpu}"
        fi
    done
}

# A model counts as trained once its log carries the twentieth epoch.  The
# driver writes last.pt after every epoch, so a checkpoint alone does not
# mean the model is finished.
trained() {
    grep -qE '^epoch 19:' "${OUT_DIR}/$1.log" 2>/dev/null
}

report_ok() {
    [[ -s "$1" ]] || return 1
    "${PYTHON}" - "$1" <<'PY' >/dev/null 2>&1
import json, sys
thresholds = json.load(open(sys.argv[1]))["thresholds"]
assert all(k in thresholds for k in ("0.1", "0.5", "2.0", "4.0"))
PY
}

eval_model() {
    local model="$1" gpu="$2"
    local ckpt out log rc
    ckpt="$(ls -dt "${OUT_DIR}/${model}"/*/last.pt 2>/dev/null | head -1)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        return 1
    fi
    out="${OUT_DIR}/${model}.eval.json"
    log="${OUT_DIR}/${model}.eval.log"
    echo "  [${model}] evaluating on GPU ${gpu}"
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
        --thresholds ${TIERS} \
        --device cuda \
        --out-json "${out}" \
        > "${log}" 2>&1
    rc=$?
    if report_ok "${out}"; then
        echo "  [${model}] done"
        return 0
    fi
    echo "  [${model}] FAILED (exit ${rc}); see ${log}"
    tail -3 "${log}" 2>/dev/null
    return 1
}

# One worker per free card.  A model is claimed with mkdir, which is
# atomic, so no two workers ever evaluate the same model.
run_queue() {
    local -a queue=("$@")
    local -a cards=()
    cards=($(free_gpus))
    if [[ ${#cards[@]} -eq 0 ]]; then
        echo "no free card; waiting 60s"
        sleep 60
        cards=($(free_gpus))
        [[ ${#cards[@]} -eq 0 ]] && return 1
    fi

    local card model candidate
    local -a pids=()
    for card in "${cards[@]}"; do
        (
            while true; do
                model=""
                for candidate in "${queue[@]}"; do
                    if mkdir "${CLAIM_DIR}/${candidate}" 2>/dev/null; then
                        model="${candidate}"
                        break
                    fi
                done
                [[ -z "${model}" ]] && break
                eval_model "${model}" "${card}"
            done
        ) &
        pids+=("$!")
    done
    wait "${pids[@]}"
}

# ---------------------------------------------------------------------
# keep going until every model has a report
# ---------------------------------------------------------------------
for round in $(seq 1 "${MAX_ROUNDS}"); do
    rm -rf "${CLAIM_DIR}"
    mkdir -p "${CLAIM_DIR}"

    pending=()
    deferred=()
    for model in "${MODELS[@]}"; do
        if report_ok "${OUT_DIR}/${model}.eval.json"; then
            continue
        fi
        if trained "${model}"; then
            pending+=("${model}")
        else
            deferred+=("${model}")
        fi
    done

    if [[ ${#pending[@]} -eq 0 && ${#deferred[@]} -eq 0 ]]; then
        echo "=== all ${#MODELS[@]} models have a report ==="
        break
    fi

    if [[ ${#pending[@]} -gt 0 ]]; then
        echo "=== round ${round}: evaluating ${#pending[@]} model(s): ${pending[*]} ==="
        run_queue "${pending[@]}"
    else
        echo "=== round ${round}: waiting for ${#deferred[@]} model(s) to finish training ==="
        sleep "${POLL_SECONDS}"
        continue
    fi

    still_missing=0
    for model in "${MODELS[@]}"; do
        if ! report_ok "${OUT_DIR}/${model}.eval.json"; then
            still_missing=$(( still_missing + 1 ))
        fi
    done
    echo "=== round ${round}: ${still_missing} model(s) still without a report ==="
    [[ ${still_missing} -eq 0 ]] && break
    [[ ${#deferred[@]} -gt 0 ]] && sleep "${POLL_SECONDS}"
done

# ---------------------------------------------------------------------
# the driver's own phase 2 may still be running and may overwrite what
# was written here.  Wait for it, then repair whatever it left broken.
# ---------------------------------------------------------------------
echo "=== waiting for the driver's own evaluation to finish ==="
for _ in $(seq 1 360); do
    if grep -q 'done: reports in' "${DRIVER_LOG}" 2>/dev/null; then
        break
    fi
    sleep 60
done

echo "=== final repair ==="
rm -rf "${CLAIM_DIR}"
mkdir -p "${CLAIM_DIR}"
missing=()
for model in "${MODELS[@]}"; do
    if ! report_ok "${OUT_DIR}/${model}.eval.json"; then
        missing+=("${model}")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "  missing: ${missing[*]}"
    run_queue "${missing[@]}"
else
    echo "  nothing to repair"
fi

echo "=== evaluation sweep done ==="
