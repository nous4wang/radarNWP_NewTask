#!/usr/bin/env bash
# E45 -- submit the seven-arm, three-seed ablation to Slurm.
#
# WHY THIS RUNS AS A SLURM JOB AND NOT FROM A LOGIN SESSION
# ----------------------------------------------------------
# E44 tried this twice from a login session: once with `nohup setsid`
# and once with `remote.py --script --detached`.  BOTH were reaped when
# the SSH session ended, so the second and third batches were never
# submitted and the batch had to be rescued by hand.  The lesson is
# recorded in MASTER v2.39: the submitter itself must be a Slurm job.
# It takes no GPU -- it only calls sbatch -- so it costs one CPU slot
# and nothing else.
#
# WHY THE SEEDS ARE SUBMITTED ONE AT A TIME
# ------------------------------------------
# The platform's limits are FOUR RUNNING JOBS for the whole account and
# TWELVE SUBMITTED JOBS, and an array's SUB-TASKS count against the
# twelve.  One seed's training array is seven sub-tasks, so two seeds
# would already be fourteen.  The chain therefore submits one seed,
# waits for its evaluation to finish, and only then submits the next.
#
# The first attempt submitted everything at once and the submission was
# refused with QOSMaxSubmitJobPerUserLimit; the retry loop then kept
# asking for the same job until the deadline, and the last evaluation
# batch was never submitted at all.  A refusal is now a hard stop with
# a clear message rather than a silent retry loop.
#
# WHY NOTHING BUT THE SEED GOES THROUGH --export
# ----------------------------------------------
# Slurm separates the variables of an --export list with COMMAS.  So
# "--export=ALL,ARM_LIST=0,1,2" is parsed as three variables and the
# list arrives as the single value "0" -- the other two entries become
# variables named "1" and "2".  The first attempt passed the arm list
# that way and every job trained exactly one arm, so four arms per seed
# never ran at all.  Only SEED (a single integer) is exported now; the
# arm is the array task id, and the arm table lives in
# scripts/_e45_arms.sh.
#
# The order is fixed by the platform rules: the log directory has to
# exist before the submission (the #SBATCH output path is relative to
# the submission directory and is not expanded by the shell), and the
# evaluation must not start before the training finished, so the
# submissions are chained with --dependency=afterok.
#
# The evaluation is one job per seed: E44's single stage-2 job walked
# five evaluations in 1 h 07 min, so twenty-one in a row would exceed
# a sane job length and a failure would cost all of it.  One job per
# seed walks seven, which is about 1 h 30 min.
#
# Usage:
#   sbatch --export=ALL,START_SEED=0 scripts/_submit_e45.sbatch

#SBATCH --job-name=e45-submit
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

set -uo pipefail

LOG_DIR="run_logs/slurm"
mkdir -p "${LOG_DIR}"

START_SEED="${START_SEED:-0}"
#: Where the runs live.  The submitter reads it to decide whether a
#: seed's training is already on disk, so it has to agree with the
#: scripts it submits.
RUN_ROOT="${RUN_ROOT:-/home/wf/radarNWP/runs/e45}"
#: How long to wait for one seed's evaluation before giving up on it.
SEED_TIMEOUT="${SEED_TIMEOUT:-21600}"          # 6 h
POLL_INTERVAL="${POLL_INTERVAL:-120}"

echo "=== E45 submission $(date -Is) ==="
echo "--- queue before ---"
squeue -u "$USER" 2>&1 | head -12

# submit_job <dependency-or-empty> <sbatch args...>
# Echoes the job id on stdout.  A refusal is reported and NOT retried:
# the twelve-job allowance frees up when the earlier jobs finish, and
# the caller decides whether to wait.
submit_job() {
    local dep="$1"; shift
    local out=""
    if [[ -n "${dep}" ]]; then
        out="$(sbatch --parsable --dependency="${dep}" "$@" 2>&1)"
    else
        out="$(sbatch --parsable "$@" 2>&1)"
    fi
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "SUBMIT REFUSED: ${out}" >&2
        return 1
    fi
    # sbatch --parsable prints "jobid;cluster" on success.  Anything
    # else is an error message that Slurm put on stdout, and the check
    # for a leading number is what tells the two apart.  The check is
    # done on the FIRST FIELD, because the job id is the first field of
    # the success form.
    local first="${out%%;*}"
    if [[ ! "${first}" =~ ^[0-9]+ ]]; then
        echo "SUBMIT REFUSED: ${out}" >&2
        return 1
    fi
    echo "${first}"
}

# wait_for <job id> <label>
# Blocks until the job leaves the queue, then reports its final state.
# The states are read from sacct, because squeue no longer knows a job
# that has finished.
wait_for() {
    local jid="$1" label="$2"
    local waited=0
    while squeue -h -j "${jid}" 2>/dev/null | grep -q .; do
        if (( waited >= SEED_TIMEOUT )); then
            echo "TIMEOUT: ${label} (job ${jid}) did not finish in ${SEED_TIMEOUT}s" >&2
            return 1
        fi
        sleep "${POLL_INTERVAL}"
        waited=$(( waited + POLL_INTERVAL ))
    done
    local state
    # sacct pads its column, so the state arrives as " COMPLETED "
    # with spaces on both sides.  A pattern like COMPLETED* does NOT
    # match that (the pattern has to match the WHOLE word), so the
    # value is trimmed first.  Getting this wrong made the submitter
    # declare a perfectly good training array FAILED.
    state="$(sacct -j "${jid}" --format=State --noheader 2>/dev/null \
             | head -1 | tr -d '[:space:]')"
    echo "--- ${label} (job ${jid}) finished: ${state} ---"
    case "${state}" in
        COMPLETED) return 0 ;;
        "") echo "FATAL: ${label} (job ${jid}) has no state in sacct." >&2
           echo "       It may have been cancelled before it started;" >&2
           echo "       check with: sacct -j ${jid}" >&2
           return 1 ;;
        *) echo "FATAL: ${label} (job ${jid}) ended in state '${state}'." >&2
           echo "       Its results are not usable; fix the cause and" >&2
           echo "       resubmit rather than reading a broken batch." >&2
           return 1 ;;
    esac
}

# --- 1. the three seeds, one at a time ----------------------------------
for SEED in $(seq "${START_SEED}" 2); do
    echo
    echo "--- seed ${SEED}: the training array (7 sub-tasks, one per arm) ---"
    # A seed whose training is already done is not trained again.  The
    # check is the seven checkpoints themselves, not a marker file: a
    # marker can survive a run that wrote nothing, and a missing
    # checkpoint is what the evaluation would fail on.
    TRAIN_READY=1
    for ARM in B0 A1 A2 A3a A3b A4 A4b; do
        if ! ls -1 "${RUN_ROOT}/${ARM}_s${SEED}"/train/*/epoch_*.pt \
             >/dev/null 2>&1; then
            TRAIN_READY=0
            echo "  ${ARM}_s${SEED}: no checkpoint yet"
        fi
    done

    if [[ "${TRAIN_READY}" -eq 1 ]]; then
        echo "  the training of seed ${SEED} is already on disk; not redoing it"
        TRAIN_JOB=""
    else
        TRAIN_JOB="$(submit_job "" --export=ALL,SEED="${SEED}" scripts/e45_train.sbatch)"
        if [[ -z "${TRAIN_JOB}" ]]; then
            echo "FATAL: the training array for seed ${SEED} was not submitted." >&2
            exit 1
        fi
        echo "training job id : ${TRAIN_JOB}"
        if ! wait_for "${TRAIN_JOB}" "the training array of seed ${SEED}"; then
            exit 1
        fi
    fi

    echo
    echo "--- seed ${SEED}: the evaluation ---"
    if [[ -n "${TRAIN_JOB}" ]]; then
        STAGE2_DEP="afterok:${TRAIN_JOB}"
    else
        STAGE2_DEP=""
    fi
    STAGE2_JOB="$(submit_job "${STAGE2_DEP}" --export=ALL,SEED="${SEED}" scripts/e45_stage2.sbatch)"
    if [[ -z "${STAGE2_JOB}" ]]; then
        echo "FATAL: the evaluation of seed ${SEED} was not submitted." >&2
        exit 1
    fi
    echo "evaluation job id : ${STAGE2_JOB}"
    if ! wait_for "${STAGE2_JOB}" "the evaluation of seed ${SEED}"; then
        exit 1
    fi
done

# --- 2. the summary -----------------------------------------------------
# The equal-bias control needs the control arm AND every other arm of
# the same seed, so it runs here, after all three evaluation jobs.
echo
echo "--- the summary (S3 and the three-seed table) ---"
SUMMARY_JOB="$(submit_job "" scripts/e45_summary.sbatch)"
if [[ -z "${SUMMARY_JOB}" ]]; then
    echo "FATAL: the summary job was not submitted." >&2
    exit 1
fi
echo "summary job id : ${SUMMARY_JOB}"
wait_for "${SUMMARY_JOB}" "the summary"

echo
echo "=== E45 submission finished $(date -Is) ==="
squeue -u "$USER" 2>&1 | head -12
