#!/usr/bin/env bash
# E43 -- submit the second stage: score the four arms and run the analyses.
#
# The second stage is ONE job, not two.  The account's QoS allows 12
# submitted jobs and the training array already took the last of the
# allowance, so an evaluation array plus a separate analysis job could
# not be submitted at all.  A single job that walks the four arms in
# turn needs one submitted slot and one GPU, and the queue accepts it
# where the array was refused.  The order inside the job is fixed by the
# data flow: the structure metrics (S1) and the equal-bias control (S3)
# read the fields that the scoring dumps.
#
# The dependency is expressed to Slurm rather than to a poller: the
# afterok dependency on the training array fires only when EVERY task of
# that array completed successfully, and each training task runs the S0
# gate as its first step.  So the evaluation cannot start on a broken
# implementation, and no background process on the login session is
# needed to submit it.
#
# Usage:
#   bash scripts/_submit_e43_stage2.sh [train_job_id]
set -uo pipefail
cd /home/wf/radarNWP || exit 1

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

TRAIN_JOB="${1:-${E43_TRAIN_JOB:-}}"
if [[ -z "${TRAIN_JOB}" ]]; then
    TRAIN_JOB=$(squeue -u "$USER" -h -n -o "%i %j" 2>/dev/null \
        | awk '$2 ~ /^e43-train/ {print $1}' | head -1)
    # An array shows up as "<id>_[<first>-<last>]"; take the base id.
    TRAIN_JOB="${TRAIN_JOB%%_*}"
fi
if [[ -z "${TRAIN_JOB}" ]]; then
    echo "FATAL: no training job id given and none found in the queue" >&2
    echo "       usage: bash scripts/_submit_e43_stage2.sh <train_job_id>" >&2
    exit 1
fi

echo "=== E43 stage-2 submission $(date -Is) ==="
echo "training job id : ${TRAIN_JOB}"

# Idempotency: if the stage-2 job is already queued, do not submit a
# second copy.
existing=$(squeue -u "$USER" -h -n -o "%i %j" 2>/dev/null \
    | awk '$2 == "e43-stage2" {print $1}' | head -1)
if [[ -n "${existing}" ]]; then
    echo "a stage-2 job is already queued (${existing}) -- nothing to do"
    exit 0
fi

STAGE2_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    scripts/e43_stage2.sbatch 2>&1)
rc=$?
if [[ ${rc} -ne 0 || -z "${STAGE2_JOB}" ]]; then
    echo "FATAL: the stage-2 job was not submitted: ${STAGE2_JOB}" >&2
    exit 1
fi
STAGE2_JOB="${STAGE2_JOB%%;*}"

echo
echo "stage-2 job id  : ${STAGE2_JOB}  (depends on ${TRAIN_JOB})"
echo
echo "--- queue after ---"
squeue -u "$USER" 2>&1 | head -15
echo
echo "watch:  squeue -u \$USER"
echo "logs :  /home/wf/radarNWP/${LOG_DIR}/"
echo "state: sacct -u \$USER -S today --format=JobID,JobName%20,State%18,Elapsed,ExitCode"
