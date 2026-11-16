#!/bin/bash
# E45 -- train one arm of the seven-arm ablation.
#
# Submitted as a Slurm ARRAY; the array index picks the arm, and the
# seed comes from the environment so that the three seeds of one arm
# are three separate sub-tasks of the same array.
#
# Why sbatch and not a background process: the platform rules forbid
# running heavy compute outside Slurm (rule 1.3), and the GPU is not
# visible to a login session at all -- the device cgroup of the login
# slice refuses /dev/nvidiactl, so cuInit returns 100 there no matter
# how healthy the driver is.  Only a job that Slurm has given a GRES
# allocation can open the device.  The device may be renumbered inside
# the job, so the script uses cuda:0 and never touches the host indices.
#
# The array table is NOT written here: it lives in scripts/_e45_arms.sh,
# which the evaluation script reads too.  A table written twice drifts,
# and the drift is silent -- a wrong arm still trains and still scores.
# The index is the Slurm ARRAY TASK ID:
#
#   0  B0   simvp_v2             --loss balanced   (same-batch control)
#   1  A1   amp_monotone_gate    --loss balanced   (the gate is a monotone function of the amplitude)
#   2  A2   amp_hard_support     --loss balanced   (a hard support with a straight-through gradient)
#   3  A3a  topo_simvp           --loss topo       (the topology term, amplitude cap only)
#   4  A3b  topo_simvp           --loss topo       (the topology term, gradient redistribution only)
#   5  A4   scaled_structure     --loss balanced   (the amount and the structure, separated)
#   6  A4b  scaled_structure_own --loss balanced   (the same, the amount carried by the model's own head)
#
# ONE array of SEVEN sub-tasks per seed, not batches of three.  The
# earlier batching existed to keep the account's running-job allowance
# free, but it did the opposite: the arm list was passed through
# --export, where a COMMA SEPARATES VARIABLES, so "ARM_LIST=0,1,2"
# was parsed as three variables and the list arrived truncated to its
# first element.  Every job then trained exactly one arm, and four
# arms per seed never ran at all.  The array index itself carries the
# arm, so nothing has to travel through the environment.
#
# Everything except --model, --loss and the topology flags is identical
# across the arms, so a difference in the score is a difference in the
# mechanism.  The topology flags are TRAINING-SIDE ONLY: the evaluation
# never reads the loss, so evaluate.py needs no flag at all.  That is
# deliberate -- a runtime switch that the evaluation side forgets to
# pass would silently evaluate a different mechanism (the trap that
# produced the "own" variant of A4 being a separate MODEL NAME).
#
# The loss of the load-bearing arm is --loss balanced --alpha-csi 0.5:
# that is what the 18-model matrix and the 60 evaluations of the master
# document used.  E41 used --loss standard, which is a known weak
# objective (it predicts only a third to a half of the rain that fell)
# and left the control arm at CSI@2.0 = 0.0000; repeating that would
# make the pairing meaningless.

#SBATCH --job-name=e45-train
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
# The seven arms are the seven sub-tasks of this array.  The directive
# has to be HERE: without it the script runs once, without an array task
# id, and it refuses to run at all -- which is the point of the guard
# further down.
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# The log file must name the ARRAY TASK, not the job: several array
# tasks share one job id, so "%x-%j" would have all of them write the
# same file and overwrite each other's output.  "%A" is the array's job
# id and "%a" the task index within it, so every task gets its own log.
#SBATCH --output=run_logs/slurm/%x-%A_%a.out
#SBATCH --error=run_logs/slurm/%x-%A_%a.err

set -eo pipefail

# The conda activation scripts reference variables that are not set, so
# ``set -u`` has to be off while conda is being activated.  With it on,
# the job dies in the activation script before python is ever reached:
#
#     libblas_mkl_activate.sh: line 1: MKL_INTERFACE_LAYER: unbound variable
#
# The nounset protection is restored once the environment is ready.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

cd "${SLURM_SUBMIT_DIR}"

# 8 logical CPUs are 4 physical cores with two threads each, so the
# compute threads are set to 4.  The dataloader workers are counted
# separately.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e45}"

EPOCHS="${EPOCHS:-3}"
# A 64-pixel patch with a stride of 64 gives the benchmark's symmetric
# origins on the grid, so the stitched field has no gap.  A stride that
# does not tile the grid makes the verifier score nothing, which is a
# silent failure.
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
TOPO_WEIGHT="${TOPO_WEIGHT:-0.1}"
TOPO_THRESHOLD="${TOPO_THRESHOLD:-2.0}"

# The arm table is read from scripts/_e45_arms.sh, the ONE place the
# index -> arm mapping is written down.  The evaluation script reads the
# same file, so the two can never disagree about which arm a score
# belongs to.
#
# The array task id IS the arm index.  Nothing about the arm travels
# through the environment, so nothing about it can be truncated there:
# the earlier attempt passed the list as ARM_LIST="0,1,2", and Slurm's
# --export treats a COMMA as the separator between variables, so the
# list arrived as the single value "0".
source "${REPO_ROOT}/scripts/_e45_arms.sh"

IDX="${SLURM_ARRAY_TASK_ID:-}"
SEED="${SEED:-0}"
if [[ -z "${IDX}" ]]; then
    echo "FATAL: this script must run as an array task (SLURM_ARRAY_TASK_ID is unset)." >&2
    echo "       The arm is the array index; there is no fallback index." >&2
    exit 2
fi

ARM="$(e45_arm "${IDX}")" || {
    echo "FATAL: array index ${IDX} is outside the arm table (0..$((E45_N_ARMS - 1)))." >&2
    exit 2
}
MODEL="$(e45_model "${IDX}")"
LOSS="$(e45_loss "${IDX}")"

# The topology flags come from the arm's own row, never from the
# environment.  They were environment variables before, and the
# submitter never exported them, so both arms received 0.0 -- which
# means "off" -- and neither arm tested the mechanism it was named
# after.  The flags are now part of the arm's definition.
TOPO_ARGS=()
TRAIN_FLAGS="$(e45_field "${IDX}" 4)"
if [[ -n "${TRAIN_FLAGS}" ]]; then
    read -r -a TOPO_ARGS <<< "${TRAIN_FLAGS}"
fi

OUT_DIR="${RUN_ROOT}/${ARM}_s${SEED}"
mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " E45 training   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}   array: ${SLURM_ARRAY_TASK_ID:-none}"
echo "   arm        : ${ARM}   model: ${MODEL}   loss: ${LOSS}   seed: ${SEED}"
echo "   topo flags : ${TOPO_ARGS[*]:-none}"
echo "   epochs     : ${EPOCHS}   patch: ${PATCH_SIZE}   stride: ${SLIDE_STRIDE}"
echo "   out        : ${OUT_DIR}"
echo "   python     : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
PY

# --- the S0 gate runs inside every array task -------------------------
# The gate is a property of the implementation, not of the data, so it
# is checked once per task: if it fails, the task exits non-zero and the
# evaluation job, which depends on this one, never starts.  Checking it
# here rather than in a separate job means the gate cannot be skipped
# by a re-submission.
echo
echo "--- S0 gate ------------------------------------------------------------"
python -u "${REPO_ROOT}/tools/_e45_s0_equivalence.py" \
    --out "${OUT_DIR}/s0_equivalence.json"
s0_rc=$?
echo "S0 rc=${s0_rc}"
if [[ "${s0_rc}" -ne 0 ]]; then
    echo "FATAL: S0 failed -- the arms would differ by more than the" >&2
    echo "       mechanism, so the comparison is void.  Fix the" >&2
    echo "       implementation; do NOT change the criterion." >&2
    exit 3
fi

echo
echo "--- training -----------------------------------------------------------"
python -u "${REPO_ROOT}/scripts/train_keep_epochs.py" \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split train \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --lr 2e-4 \
    --loss "${LOSS}" \
    --alpha-csi "${ALPHA_CSI}" \
    --topo-weight "${TOPO_WEIGHT}" \
    --topo-threshold "${TOPO_THRESHOLD}" \
    "${TOPO_ARGS[@]}" \
    --qpe-weight 0.0 \
    --log-every 50 \
    --seed "${SEED}" \
    --device cuda \
    --out-dir "${OUT_DIR}/train"
train_rc=$?
echo "train rc=${train_rc}"

# The checkpoint must exist and must not be empty.  A run that "succeeds"
# without writing a checkpoint is a silent failure that the evaluation
# would otherwise turn into a zero score.
CKPT_COUNT=$(ls -1 "${OUT_DIR}"/train/*/epoch_*.pt 2>/dev/null | wc -l)
echo "checkpoints written: ${CKPT_COUNT}"
if [[ "${CKPT_COUNT}" -lt 1 ]]; then
    echo "FATAL: no checkpoint under ${OUT_DIR}/train" >&2
    exit 4
fi

echo
echo "=========================================================="
echo " ${ARM} seed ${SEED} finished $(date -Is)"
echo "   train rc = ${train_rc}   checkpoints = ${CKPT_COUNT}"
echo "=========================================================="

exit "${train_rc}"
