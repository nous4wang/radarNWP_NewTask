#!/usr/bin/env bash
# E44 -- submit the four-branch ablation to Slurm.
#
# The order is fixed by the platform rules: the log directory has to
# exist before the submission (the #SBATCH output path is relative to
# the submission directory and is not expanded by the shell), and the
# evaluation must not start before the training finished, so the two
# submissions are chained with --dependency=afterok.
#
# The submission allowance is the binding constraint and it is shared
# with another experiment on the same account: the QoS allows 12
# submitted jobs and an ARRAY COUNTS ITS SUB-TASKS.  Measured at
# submission time on 2026-11-09 the account already held 8 submitted
# sub-tasks, so the five training tasks plus the stage-2 job need six
# slots and only four were free.  The submission therefore RETRIES
# until the allowance frees up, rather than being cut down: a cut
# would have meant dropping an arm, and the arms are the whole point.
#
# The stage-2 job is submitted AFTER the training job, so it takes one
# slot at a time rather than both at once.  That also means the
# evaluation only enters the queue when the training is actually done.
#
# Usage:
#   bash scripts/_submit_e44.sh
set -uo pipefail
cd /home/wf/radarNWP || exit 1

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

MAX_WAIT="${MAX_WAIT:-7200}"          # 2 h of waiting for the allowance
RETRY_WAIT="${RETRY_WAIT:-120}"        # 2 min between attempts
DEADLINE=$(( $(date +%s) + MAX_WAIT ))

echo "=== E44 submission $(date -Is) ==="
echo "--- queue before ---"
squeue -u "$USER" 2>&1 | head -12

# --- 1. the training array ------------------------------------------------
TRAIN_JOB=""
while :; do
    TRAIN_JOB=$(sbatch --parsable scripts/e44_train.sbatch 2>&1)
    rc=$?
    if [[ ${rc} -eq 0 && -n "${TRAIN_JOB}" ]]; then
        TRAIN_JOB="${TRAIN_JOB%%;*}"
        break
    fi
    echo "the training array was refused: ${TRAIN_JOB}"
    if (( $(date +%s) > DEADLINE )); then
        echo "FATAL: the submission allowance did not free up in time" >&2
        exit 1
    fi
    echo "retrying in ${RETRY_WAIT}s (the account's submitted-job allowance is full)"
    sleep "${RETRY_WAIT}"
done
echo
echo "training job id : ${TRAIN_JOB}"

# --- 2. the second stage --------------------------------------------------
# It depends on the training, so it cannot be submitted before the
# training has drained anyway; submitting it now means Slurm holds it
# and starts it automatically.
STAGE2_JOB=""
while :; do
    STAGE2_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
        scripts/e44_stage2.sbatch 2>&1)
    rc=$?
    if [[ ${rc} -eq 0 && -n "${STAGE2_JOB}" ]]; then
        STAGE2_JOB="${STAGE2_JOB%%;*}"
        break
    fi
    echo "the stage-2 job was refused: ${STAGE2_JOB}"
    if (( $(date +%s) > DEADLINE )); then
        echo "FATAL: the stage-2 job could not be submitted in time" >&2
        echo "       the training job ${TRAIN_JOB} is still running;" >&2
        echo "       submit it later with:" >&2
        echo "         bash scripts/_submit_e44_stage2.sh ${TRAIN_JOB}" >&2
        exit 1
    fi
    echo "retrying in ${RETRY_WAIT}s"
    sleep "${RETRY_WAIT}"
done
echo "stage-2 job id  : ${STAGE2_JOB}  (depends on ${TRAIN_JOB})"

echo
echo "--- queue after ---"
squeue -u "$USER" 2>&1 | head -15
echo
echo "watch:  squeue -u \$USER"
echo "logs :  /home/wf/radarNWP/${LOG_DIR}/"
echo "state: sacct -u \$USER -S today --format=JobID,JobName%20,State%18,Elapsed,ExitCode"
