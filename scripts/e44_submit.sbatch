#!/bin/bash
# E44 -- the submitter, as a Slurm job.
#
# The account's submitted-job allowance is shared with another
# experiment, and at the time of writing it held 8 of the 12 allowed
# sub-tasks, while this batch needs 6.  A submitter that waits for the
# allowance has to live somewhere that outlives an SSH session: the
# login slice reaps the processes of a session when it closes, which
# was observed twice (the nohup+setsid form and the detached form both
# died with the connection).
#
# So the submitter is itself a Slurm job.  It asks for no GPU and a
# couple of cores: it only calls sbatch, and the heavy compute stays
# inside the jobs it submits.  The platform rules forbid using
# background processes to run HEAVY COMPUTE outside Slurm; running the
# submission machinery inside Slurm is the opposite of that.
#
# The job's own lifetime is bounded by --time, so the submitter cannot
# outlive the allowance wait.

#SBATCH --job-name=e44-submit
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

RETRY_WAIT="${RETRY_WAIT:-120}"        # 2 min between attempts
DEADLINE=$(( $(date +%s) + 6600 ))     # 1 h 50 min, inside --time

echo "=========================================================="
echo " E44 submitter   $(date -Is)"
echo "   job id : ${SLURM_JOB_ID}"
echo "=========================================================="
squeue -u "$USER" 2>&1 | head -12

# --- the training array --------------------------------------------------
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
        echo "FATAL: the allowance did not free up in time" >&2
        exit 1
    fi
    echo "retrying in ${RETRY_WAIT}s"
    sleep "${RETRY_WAIT}"
done
echo
echo "training job id : ${TRAIN_JOB}"

# --- the second stage ----------------------------------------------------
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
        echo "       submit the second stage later with:" >&2
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
echo "=========================================================="
echo " submitter finished $(date -Is)"
echo "=========================================================="
