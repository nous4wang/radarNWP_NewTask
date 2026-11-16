#!/usr/bin/env bash
# E43 -- submit the low-rank + sparse ablation to Slurm.
#
# The order is fixed by the platform rules: the log directory has to
# exist before the submission (the #SBATCH output path is relative to
# the submission directory and is not expanded by the shell).
#
# The array is 6 tasks (2 arms x 3 seeds), each with one GPU, so the
# account's total of 6 GPUs sits exactly at the documented ceiling
# (section 2.2: 6 GPUs for the whole account, 4 running jobs, 12
# submitted jobs).  The array counts as 6 submissions, which is inside
# the limit of 12.
#
# Why the evaluation is NOT submitted here: the QoS allows 12 submitted
# jobs per account, and this account already held 8 when the training
# array went in, so the training array consumed the last of the
# allowance and the evaluation was rejected with
# QOSMaxSubmitJobPerUserLimit.  The evaluation is therefore submitted
# once the training array has drained; scripts/_submit_e43_eval.sh does
# that, and it is the same two-stage chain the rules describe.
#
# Usage:
#   bash scripts/_submit_e43.sh          # training
#   bash scripts/_submit_e43_eval.sh     # evaluation + analysis
set -uo pipefail
cd /home/wf/radarNWP || exit 1

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

echo "=== E43 training submission $(date -Is) ==="
echo "--- queue before ---"
squeue -u "$USER" 2>&1 | head -12

TRAIN_JOB=$(sbatch --parsable scripts/e43_train.sbatch)
if [[ -z "${TRAIN_JOB}" ]]; then
    echo "FATAL: the training job was not submitted" >&2
    exit 1
fi
echo
echo "training job id : ${TRAIN_JOB}"

echo
echo "--- queue after ---"
squeue -u "$USER" 2>&1 | head -15
echo
echo "The evaluation is submitted separately once the training array has"
echo "drained (the account's 12-job submission allowance was already"
echo "close to its limit).  Submit it with:"
echo
echo "    bash scripts/_submit_e43_eval.sh ${TRAIN_JOB}"
echo
echo "watch:  squeue -u \$USER"
echo "logs :  /home/wf/radarNWP/${LOG_DIR}/"
echo "state: sacct -u \$USER -S today --format=JobID,JobName%20,State%18,Elapsed,ExitCode"
