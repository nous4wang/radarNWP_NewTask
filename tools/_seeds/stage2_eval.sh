#!/usr/bin/env bash
# Score the extra route P2/P3 seed runs.
#
# Two studies are scored here, and they are scored the same way: the
# same evaluator, the same thresholds, the same test split, and the same
# per-window tables, because the bootstrap resamples the windows.
#
# 1. Route P2's pass rate at the peak.  Ten seeds per configuration, each
#    scored at the epochs where its peak can be.  The criterion is the
#    persistence floor, CSI@2.0 > 0.0924, and the question is how often
#    each configuration reaches it.
#
# 2. Route P3's lag sweep, three seeds per arm.  The master document says
#    the lag curve cannot support a quantitative conclusion because one
#    seed's spread reaches 2.87x, and that four epochs of one run are not
#    four independent samples.  Three seeds are what the master document
#    asks for.
#
# The judge is the file count: one ``.eval.json`` and one
# ``.windows.json`` per (arm, seed, epoch).  A report that exists and
# parses is left alone, which is what makes the script safe to run again
# while the training is still going.
#
# The concurrency is eight.  The master document measured that eight is
# what this machine does best: the verification is bound by memory
# bandwidth, not by the number of cards, so more concurrency does not go
# faster.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_seeds}"
TRAIN_DIR="${OUT_DIR}/train"
EVAL_DIR="${OUT_DIR}/evals"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
PARALLEL="${PARALLEL:-8}"
LOCK_DIR="${OUT_DIR}/.eval_locks"

mkdir -p "${EVAL_DIR}" "${LOCK_DIR}"

# The epochs to score.  The peak of every arm is in the first three
# epochs (master document 4.12.1), and the last epoch is kept as the
# counterpart of the earlier collapsed-end numbers.
SCORE_EPOCHS="${SCORE_EPOCHS:-00 01 02}"

# The card is chosen here, in the parent, and passed in.  Choosing it
# inside the background job would give every job the same card, because
# a background job runs in a subshell and cannot change the parent's
# counter.  The same is true of a command substitution: the increment
# has to happen in the parent loop, not inside a $(...) that is run in a
# subshell of its own.
score_one() {
    local name="$1" model="$2" kwargs="$3" seed="$4" ep="$5" run_dir="$6" gpu="$7"
    local ck out win lock
    ck="${run_dir}epoch_${ep}.pt"
    [[ -f "${ck}" ]] || return 0
    out="${EVAL_DIR}/${name}_s${seed}_ep${ep}.eval.json"
    win="${EVAL_DIR}/${name}_s${seed}_ep${ep}.windows.json"
    if [[ -f "${out}" ]] && "${PYTHON}" -c \
        "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null; then
        echo "  ${name} seed ${seed} epoch ${ep}: already scored"
        return 0
    fi
    lock="${LOCK_DIR}/${name}_s${seed}_ep${ep}"
    if ! mkdir "${lock}" 2>/dev/null; then
        echo "  ${name} seed ${seed} epoch ${ep}: another scorer has it"
        return 0
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
            --model-kwargs "${kwargs}" \
            --out-json "${out}" \
            --per-window-out "${win}" \
            > "${EVAL_DIR}/${name}_s${seed}_ep${ep}.eval.log" 2>&1
    )
    if [[ -f "${out}" ]]; then
        echo "  done: ${name} seed ${seed} epoch ${ep}"
    else
        echo "  FAILED: ${name} seed ${seed} epoch ${ep}"
    fi
    rmdir "${lock}" 2>/dev/null || true
}

slot=0

# ---------------------------------------------------------------------------
# Route P3: the lag sweep.  Seed 0 is already trained (runs/route_p3_epochs);
# it is re-scored here with the per-window tables, which the earlier run
# did not write.
# ---------------------------------------------------------------------------
echo "=========================================================="
echo " route P3: the lag sweep, three seeds per arm"
echo "=========================================================="

for entry in "assim_lag0 0" "assim_lag2 2" "assim_lag4 4" "assim_lag11 11"; do
    read -r name lag <<< "${entry}"
    for seed in 0 1 2; do
        if [[ "${seed}" -eq 0 ]]; then
            run_dir="$(ls -dt "/home/wf/radarNWP/runs/route_p3_epochs/${name}"/*/ 2>/dev/null | head -1)"
        else
            run_dir="$(ls -dt "${TRAIN_DIR}/${name}_s${seed}"/*/ 2>/dev/null | head -1)"
        fi
        [[ -n "${run_dir}" ]] || continue
        for ep in ${SCORE_EPOCHS}; do
            gpu=$(( slot % 8 ))
            slot=$(( slot + 1 ))
            score_one "${name}" simvp_v2_assimilated \
                "{\"obs_lag_steps\":${lag}}" "${seed}" "${ep}" "${run_dir}" "${gpu}" &
            while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]]; do sleep 5; done
        done
    done
done
wait

# ---------------------------------------------------------------------------
# Route P2: the pass rate at the peak, ten seeds per configuration.
#
# These two models take no ``obs_lag_steps``; the keyword would be an
# unexpected argument and the evaluator would refuse to build the model.
# ---------------------------------------------------------------------------
echo
echo "=========================================================="
echo " route P2: the pass rate at the peak, ten seeds each"
echo "=========================================================="

for entry in "qpe_0 simvp_v2_qpe_guided" "base_concat simvp_v2_nwp_concat"; do
    read -r name model <<< "${entry}"
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        if [[ "${seed}" -le 2 ]]; then
            run_dir="$(ls -dt "/home/wf/radarNWP/runs/route_p2/seeds/${name}_s${seed}"/*/ 2>/dev/null | head -1)"
        else
            run_dir="$(ls -dt "${TRAIN_DIR}/${name}_s${seed}"/*/ 2>/dev/null | head -1)"
        fi
        [[ -n "${run_dir}" ]] || continue
        for ep in ${SCORE_EPOCHS}; do
            gpu=$(( slot % 8 ))
            slot=$(( slot + 1 ))
            score_one "${name}" "${model}" "{}" "${seed}" "${ep}" "${run_dir}" "${gpu}" &
            while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]]; do sleep 5; done
        done
    done
done
wait

echo
echo "the judge is the file count:"
for name in assim_lag0 assim_lag2 assim_lag4 assim_lag11 qpe_0 base_concat; do
    n="$(ls "${EVAL_DIR}/${name}"_s*_ep*.eval.json 2>/dev/null | wc -l)"
    w="$(ls "${EVAL_DIR}/${name}"_s*_ep*.windows.json 2>/dev/null | wc -l)"
    echo "  ${name}: ${n} evaluations, ${w} window tables"
done

echo "=========================================================="
echo " route P2/P3 extra seeds scored."
echo "=========================================================="
