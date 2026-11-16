#!/usr/bin/env bash
# E41 batch 1 -- EVALUATION.  The criteria are frozen (E41 section 5,
# 2026-11-09 09:29) and are NOT touched here.
#
# What this scores:
#   B0  simvp_v2      + standard   (same-batch control)
#   A0  struct_hurdle head=rain   + standard   (implementation equivalence)
#   A1  struct_hurdle head=hurdle + hurdle    (output parameterisation)
#   A2  struct_hurdle head=rain   + uot        (loss geometry)
#
# One epoch is scored: epoch_00.  The master document fixes the peak of
# every arm at the first epoch, and section 11.9 binds the pass rate to
# ep00, so ep00 is the point the criterion is read at.  ep01/ep02 are
# NOT scored in this batch: they are the collapse end, and scoring them
# would multiply the evaluation cost without changing the decision.
#
# The evaluation windows are the same for every arm: --split test with
# --patch-size 64 --slide-stride 64 and the stitched whole-field
# verification, i.e. the same 538 windows the other tables in the master
# document use.
#
# --dump-fields writes the stitched float32 forecast field and the truth
# field per window.  That is what the section 2 structure metrics and the
# section 5 S3 equal-bias control are computed from, with the SAME
# implementation the section 2 probe used.  The evaluator is not modified.
#
# A0 is scored without --dump-fields: its role is the equivalence check
# (its training loss was already bit-identical to B0's), and its fields
# would be a duplicate of B0's.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP

PY=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
OUT=/home/wf/radarNWP/runs/e41
TRAIN="${OUT}/train"
EVAL_DIR="${OUT}/evals"
FIELDS="${OUT}/fields"
STAGE="${OUT}/eval_stage.log"
BATCH=16
WORKERS=6
PAR=8
SEEDS="0 1 2"
EPOCH="00"

mkdir -p "${EVAL_DIR}" "${FIELDS}" "${OUT}/logs/eval"
: > "${STAGE}"
log() { echo "$1 $(date '+%F %T')" | tee -a "${STAGE}"; }

run_dir_of() { ls -d "${TRAIN}/$1_s$2"/*/ 2>/dev/null | head -1; }

score_one() {
    local arm="$1" model="$2" kwargs="$3" seed="$4" gpu="$5" dump="$6"
    local run_dir ck tag out win
    run_dir="$(run_dir_of "${arm}" "${seed}")"
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
            > "${OUT}/logs/eval/${tag}.eval.log" 2>&1
    )
    if [ -f "${out}" ]; then
        log "DONE ${tag}"
    else
        log "FAILED ${tag}"
        tail -n 12 "${OUT}/logs/eval/${tag}.eval.log" >> "${STAGE}"
    fi
}

slot=0
for seed in ${SEEDS}; do
    for spec in \
        "B0 simvp_v2 {} yes" \
        "A0 struct_hurdle {} no" \
        "A1 struct_hurdle {\"head_mode\":\"hurdle\"} yes" \
        "A2 struct_hurdle {} yes" ; do
        read -r arm model kwargs dump <<< "${spec}"
        gpu=$(( slot % 8 ))
        slot=$(( slot + 1 ))
        log "LAUNCH ${arm} s${seed} gpu=${gpu} dump=${dump}"
        score_one "${arm}" "${model}" "${kwargs}" "${seed}" "${gpu}" "${dump}" &
        while [ "$(jobs -rp | wc -l)" -ge "${PAR}" ]; do sleep 5; done
    done
done
wait

ok=0
bad=0
for seed in ${SEEDS}; do
    for arm in B0 A0 A1 A2; do
        f="${EVAL_DIR}/${arm}_s${seed}_ep${EPOCH}.eval.json"
        if [ -f "${f}" ]; then ok=$(( ok + 1 )); else bad=$(( bad + 1 )); fi
    done
done
log "E41 EVAL DONE ok=${ok} failed=${bad} epoch=${EPOCH}"
echo "evals: ${EVAL_DIR}; fields: ${FIELDS}; stage: ${STAGE}"
