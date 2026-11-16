#!/bin/bash
# E45 -- the two READ-ONLY diagnostic probes for the three open items.
#
# These are not part of E45's verdict.  E45 is closed (v2.44/v2.45):
# six arms judged negative, no arm promoted to ten seeds.  This job
# exists only to LOCALISE the three registered implementation defects
# at source level, so the fix can be written against the real cause
# instead of the recorded (partly wrong) explanation.
#
#   _e45_meanout_diag.py   -- why is `mean(out) = T` violated?  It measures
#                             mean(S), m = mean(F_mm) and the per-frame
#                             error directly on the real checkpoints.
#   _e45_maxratio_diag.py  -- why are the gate arms' maxratio / tail
#                             read-outs NaN?  It reproduces E44's
#                             `_max_ratio` per batch and compares
#                             np.median against np.nanmedian.
#
# CPU only -- both scripts are numpy/torch on CPU, they need no GPU.
# Read-only: they load checkpoints and the dataset, they write nothing
# into runs/.
#
#SBATCH --job-name=e45-diag
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

set -eo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"

echo "=========================================================="
echo " E45 diagnostic probes   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   repo root  : ${REPO_ROOT}"
echo "=========================================================="

echo
echo "########## MEANOUT DIAG (D2: mean(out)=T) ##########"
python -u "${REPO_ROOT}/tools/_e45_meanout_diag.py"
echo "meanout rc=${?}"

echo
echo "########## MAXRATIO DIAG (D3: NaN read-outs) ##########"
python -u "${REPO_ROOT}/tools/_e45_maxratio_diag.py"
echo "maxratio rc=${?}"

echo
echo "=== E45 diagnostic probes finished $(date -Is) ==="
