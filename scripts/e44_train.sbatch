#!/bin/bash
# E44 -- train one arm of the four-branch ablation.
#
# Submitted as a Slurm ARRAY; the array index picks the arm, and every
# arm uses the same seed so that the comparison is paired.
#
# Why sbatch and not a background process: the platform rules forbid
# running heavy compute outside Slurm (rule 1.3), and the GPU is not
# visible to a login session at all -- the device cgroup of the login
# slice refuses /dev/nvidiactl, so cuInit returns 100 there no matter
# how healthy the driver is.  Only a job that Slurm has given a GRES
# allocation can open the device.  The device may be renumbered inside
# the job, so the script uses cuda:0 and never touches the host indices.
#
# The array table (the index picks one row):
#
#   0  B0   simvp_v2      --loss balanced   (same-batch control)
#   1  A1   lr_sparse_bg  --loss balanced   (the low-rank constrains the background)
#   2  A2   lr_sparse_add --loss balanced   (an additive decomposition)
#   3  A3   topo_simvp    --loss topo      (a clDice connectivity term)
#   4  A4   fno_simvp     --loss balanced   (an FNO decoder)
#
# Everything except --model and --loss is identical across the arms, so
# a difference in the score is a difference in the mechanism.
#
# One seed per arm (seed 0).  This is tier 0 of the promotion ladder in
# section 11.9 P2 as revised on 2026-11-09: one seed may say "this arm
# passed the floor" and nothing more.  A second seed is added only for
# the arms that pass the floor.
#
# The loss of the load-bearing arm is --loss balanced --alpha-csi 0.5:
# that is what the 18-model matrix and the 60 evaluations of the master
# document used.  E41 used --loss standard, which is a known weak
# objective (it predicts only a third to a half of the rain that fell)
# and left the control arm at CSI@2.0 = 0.0000; repeating that would
# make the pairing meaningless.

#SBATCH --job-name=e44-train
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

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
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e44}"

EPOCHS="${EPOCHS:-3}"
# A 64-pixel patch with a stride of 64 gives the benchmark's symmetric
# origins on the 301x301 grid, so the stitched field has no gap.  A
# stride that does not tile the grid makes the verifier score nothing,
# which is a silent failure.
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
TOPO_WEIGHT="${TOPO_WEIGHT:-0.1}"
TOPO_THRESHOLD="${TOPO_THRESHOLD:-2.0}"

# The array index picks the arm.
IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "${IDX}" in
    0) ARM="B0"; MODEL="simvp_v2";      LOSS="balanced" ;;
    1) ARM="A1"; MODEL="lr_sparse_bg";  LOSS="balanced" ;;
    2) ARM="A2"; MODEL="lr_sparse_add"; LOSS="balanced" ;;
    3) ARM="A3"; MODEL="topo_simvp";    LOSS="topo" ;;
    4) ARM="A4"; MODEL="fno_simvp";     LOSS="balanced" ;;
    *) echo "unknown array index ${IDX}" >&2; exit 2 ;;
esac
SEED=0

OUT_DIR="${RUN_ROOT}/${ARM}_s${SEED}"
mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " E44 training   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}   array: ${SLURM_ARRAY_TASK_ID:-none}"
echo "   arm        : ${ARM}   model: ${MODEL}   loss: ${LOSS}   seed: ${SEED}"
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
python -u "${REPO_ROOT}/tools/_e44_s0_equivalence.py" \
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
