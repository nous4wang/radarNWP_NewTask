#!/bin/bash
# E43 -- the discriminating read-out: the shape of the learned support mask.
#
# The login session cannot see a GPU (the device cgroup of the login
# slice refuses /dev/nvidiactl), so the probe runs as a Slurm job with a
# GRES allocation, exactly like the evaluation.
#
# The probe reads the trained checkpoints and the test data.  It trains
# nothing and it produces no score, so it changes no number that any
# criterion reads; it only measures the mask that the A3 arm actually
# used, which E43 section 3 asks for as the guard on how S1's four
# numbers are to be interpreted.

#SBATCH --job-name=e43-mask-probe
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

set -eo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# The probe imports the benchmark package and the model's own module, and
# the package is not installed into the environment -- it is imported
# from the repository, so the repository has to be on the path.
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e43}"

echo "=========================================================="
echo " E43 mask probe   $(date -Is)"
echo "   job id : ${SLURM_JOB_ID}"
echo "   python : $(command -v python)"
echo "=========================================================="

python -u "${REPO_ROOT}/tools/_e43_mask_probe.py" \
    --runs "${RUN_ROOT}" \
    --arms A3 \
    --seeds 0 1 \
    --epoch 00 \
    --data-root "${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}" \
    --patch-size 64 \
    --slide-stride 64 \
    --batch-size 8 \
    --num-workers 4 \
    --out "${RUN_ROOT}/mask_probe.json"
rc=$?
echo "mask probe rc=${rc}"

echo
echo "=========================================================="
echo " mask probe finished $(date -Is)   rc=${rc}"
echo "=========================================================="

exit "${rc}"
