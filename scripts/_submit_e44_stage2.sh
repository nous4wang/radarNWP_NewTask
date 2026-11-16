#!/usr/bin/env bash
# E44 -- submit the second stage on its own, after the training.
#
# The submitter job (scripts/e44_submit.sbatch) normally submits both
# stages in one go.  This script is the fallback for the case where the
# allowance stayed full for the whole of the submitter's lifetime: the
# training job is running or done, and the second stage still has to be
# submitted by hand.
#
# Usage:
#   bash scripts/_submit_e44_stage2.sh [train_job_id]
set -uo pipefail
cd /home/wf/radarNWP || exit 1

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

TRAIN_JOB="${1:-${E43_TRAIN_JOB:-}}"
if [[ -z "${TRAIN_JOB}" ]]; then
    TRAIN_JOB=$(squeue -u "$USER" -h -n -o "%i %j" 2>/dev/null \
        | awk '$2 ~ /^e43-train/ || $2 ~ /^e44-train/ {print $1}' | head -1)
    TRAIN_JOB="${TRAIN_JOB%%_*}"
fi
if [[ -z "${TRAIN_JOB}" ]]; then
    echo "FATAL: no training job id given and none found in the queue" >&2
    echo "       usage: bash scripts/_submit_e44_stage2.sh <train_job_id>" >&2
    exit 1
fi

echo "=== E44 stage-2 submission $(date -Is) ==="
echo "training job id : ${TRAIN_JOB}"

existing=$(squeue -u "$USER" -h -n -o "%i %j" 2>/dev/null \
    | awk '$2 == "e44-stage2" {print $1}' | head -1)
if [[ -n "${existing}" ]]; then
    echo "a stage-2 job is already queued (${existing}) -- nothing to do"
    exit 0
fi

STAGE2_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    scripts/e44_stage2.sbatch 2>&1)
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
