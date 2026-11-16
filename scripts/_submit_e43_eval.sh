#!/usr/bin/env bash
# E43 -- submit the evaluation and the analysis, once the training array
# has drained.
#
# The account's QoS allows 12 submitted jobs at a time and the training
# array used the last of the allowance, so the evaluation could not be
# submitted in the same breath (it was rejected with
# QOSMaxSubmitJobPerUserLimit).  Submitting it afterwards is the same
# two-stage chain the rules describe, and it also means the evaluation
# only enters the queue when the training is actually done.
#
# The training job id is taken as the first argument, or from
# E43_TRAIN_JOB, or taken as the most recent e43-train job of this
# account.
#
# Usage:
#   bash scripts/_submit_e43_eval.sh [train_job_id]
set -uo pipefail
cd /home/wf/radarNWP || exit 1

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

TRAIN_JOB="${1:-${E43_TRAIN_JOB:-}}"
if [[ -z "${TRAIN_JOB}" ]]; then
    TRAIN_JOB=$(squeue -u "$USER" -h -n -o "%i %j" 2>/dev/null \
        | awk '$2 ~ /^e43-train/ {print $1}' | tail -1)
fi
if [[ -z "${TRAIN_JOB}" ]]; then
    echo "FATAL: no training job id given and none found in the queue" >&2
    echo "       usage: bash scripts/_submit_e43_eval.sh <train_job_id>" >&2
    exit 1
fi

echo "=== E43 evaluation submission $(date -Is) ==="
echo "training job id : ${TRAIN_JOB}"

# The evaluation waits for the training array to finish successfully.
# The S0 gate runs inside every training task, so a task that failed the
# gate fails the array, and this dependency then never fires.
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    scripts/e43_eval.sbatch)
if [[ -z "${EVAL_JOB}" ]]; then
    echo "FATAL: the evaluation job was not submitted" >&2
    exit 1
fi
echo "evaluation job id: ${EVAL_JOB}  (depends on ${TRAIN_JOB})"

# The structure metrics and the equal-bias control read the fields the
# evaluation dumped, so they come after it.  One CPU job, not an array:
# they are cheap next to the evaluation and they share one result file.
ANALYSE_JOB=$(sbatch --parsable --dependency="afterok:${EVAL_JOB}" \
    scripts/e43_analyse.sbatch)
if [[ -z "${ANALYSE_JOB}" ]]; then
    echo "FATAL: the analysis job was not submitted" >&2
    exit 1
fi
echo "analysis job id  : ${ANALYSE_JOB}  (depends on ${EVAL_JOB})"

echo
echo "--- queue after ---"
squeue -u "$USER" 2>&1 | head -15
echo
echo "watch:  squeue -u \$USER"
echo "logs :  /home/wf/radarNWP/${LOG_DIR}/"
