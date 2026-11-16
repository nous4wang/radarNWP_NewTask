#!/bin/bash
# E45 round 91 -- bootstrap RNG-seed robustness of the S3 guard rail.
#
# The published S3 verdict rests on ONE bootstrap random seed.  This job
# re-runs the resampling with twenty seeds (the registered one first,
# as a reproduction check, then nineteen others) and records for each
# arm whether the "the interval contains zero" answer changes.
#
# It needs no GPU: it reads the dumped fields and the window tables, the
# same inputs the guard rail read.  One CPU slot, the GPUs stay free.

#SBATCH --job-name=e45-s3seedrobust
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e45}"

mkdir -p run_logs/slurm

echo "=========================================================="
echo " E45 S3 seed robustness   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   run root   : ${RUN_ROOT}"
echo "=========================================================="

python -u "${REPO_ROOT}/tools/_e45_s3_seedrobust.py" \
    --repo "${REPO_ROOT}" \
    --runs "${RUN_ROOT}" \
    --out "${RUN_ROOT}/s3_seedrobust.json"
rc=$?
echo "seedrobust rc=${rc}"

echo
echo "=========================================================="
echo " finished $(date -Is)  rc=${rc}"
echo "=========================================================="

exit ${rc}
