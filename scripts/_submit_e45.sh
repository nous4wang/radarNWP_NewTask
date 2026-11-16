#!/usr/bin/env bash
# E45 -- submit the seven-arm, three-seed ablation to Slurm.
#
# The order is fixed by the platform rules: the log directory has to
# exist before the submission (the #SBATCH output path is relative to
# the submission directory and is not expanded by the shell), and the
# evaluation must not start before the training finished, so the
# submissions are chained with --dependency=afterok.
#
# The submission allowance is the binding constraint and it is shared
# with another experiment on the same account: the QoS allows 12
# submitted jobs and an ARRAY COUNTS ITS SUB-TASKS.  Twenty-one
# training sub-tasks plus three evaluation jobs would need
# twenty-four slots, so the training is submitted in THREE batches of
# seven -- one batch per seed -- and the batches are chained with
# afterok.  At any moment only seven sub-tasks plus one job are
# submitted, which fits the allowance.
#
# The evaluation is likewise split per seed: E44's single stage-2 job
# walked five evaluations in 1 h 07 min, so twenty-one in a row would
# exceed a sane job length and a failure would cost all of it.
#
# The submission therefore RETRIES until the allowance frees up,
# rather than being cut down: a cut would have meant dropping an arm
# or a seed, and the arms and the seeds are the whole point.
#
# Usage:
#   bash scripts/_submit_e45.sh
set -uo pipefail
cd /home/wf/radarNWP || exit 1

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

MAX_WAIT="${MAX_WAIT:-14400}"          # 4 h of waiting for the allowance
RETRY_WAIT="${RETRY_WAIT:-120}"        # 2 min between attempts
DEADLINE=$(( $(date +%s) + MAX_WAIT ))

echo "=== E45 submission $(date -Is) ==="
echo "--- queue before ---"
squeue -u "$USER" 2>&1 | head -12

# submit_job <dependency-or-empty> <sbatch args...>
# Echoes the job id on stdout.  Retries until the allowance frees up.
submit_job() {
    local dep="$1"; shift
    local job=""
    while :; do
        if [[ -n "${dep}" ]]; then
            job=$(sbatch --parsable --dependency="${dep}" "$@" 2>&1)
        else
            job=$(sbatch --parsable "$@" 2>&1)
        fi
        rc=$?
        if [[ ${rc} -eq 0 && -n "${job}" ]]; then
            echo "${job%%;*}"
            return 0
        fi
        echo "the submission was refused: ${job}"
        if (( $(date +%s) > DEADLINE )); then
            echo "FATAL: the submission allowance did not free up in time" >&2
            return 1
        fi
        echo "retrying in ${RETRY_WAIT}s (the account's submitted-job allowance is full)"
        sleep "${RETRY_WAIT}"
    done
}

# --- 1. the training batches, one per seed ----------------------------
# SEED is exported into the array's environment, so the three seeds of
# an arm are three sub-tasks of the same array.
PREV=""
for SEED in 0 1 2; do
    echo
    echo "--- training batch, seed ${SEED} ---"
    TRAIN_JOB=$(submit_job "${PREV}" \
        --export=ALL,SEED="${SEED}" scripts/e45_train.sbatch)
    if [[ -z "${TRAIN_JOB}" ]]; then
        echo "FATAL: the training batch (seed ${SEED}) was not submitted" >&2
        exit 1
    fi
    echo "training job id (seed ${SEED}) : ${TRAIN_JOB}"
    PREV="afterok:${TRAIN_JOB}"
done

# --- 2. the evaluation batches, one per seed --------------------------
# Each evaluation job scores its own seed's seven arms and runs the S3
# equal-bias control against that seed's control arm.
for SEED in 0 1 2; do
    echo
    echo "--- evaluation batch, seed ${SEED} ---"
    STAGE2_JOB=$(submit_job "${PREV}" \
        --export=ALL,SEED="${SEED}" scripts/e45_stage2.sbatch)
    if [[ -z "${STAGE2_JOB}" ]]; then
        echo "FATAL: the evaluation batch (seed ${SEED}) was not submitted" >&2
        exit 1
    fi
    echo "evaluation job id (seed ${SEED}) : ${STAGE2_JOB}"
    PREV="afterok:${STAGE2_JOB}"
done

echo
echo "--- queue after ---"
squeue -u "$USER" 2>&1 | head -15
echo
echo "watch:  squeue -u \$USER"
echo "logs :  /home/wf/radarNWP/${LOG_DIR}/"
echo "state: sacct -u \$USER -S today --format=JobID,JobName%20,State%18,Elapsed,ExitCode"
