#!/bin/bash
# E43 -- recompute the S3 equal-bias control for all three batches with
# the fixed bootstrap.
#
# The bootstrap derived the window count from ``len(c[0])`` instead of
# ``len(c)``, so it resampled three windows out of 538 and every S3
# interval ever read from that tool described a three-window sampling
# distribution.  E41's and E42's verdicts were taken from such
# intervals, so they are recomputed here rather than left standing on a
# number the tool got wrong.
#
# The recomputation reads only artefacts that already exist: the
# forecast fields and the per-window tables that the evaluations wrote.
# No model is trained and no score is produced, so section 11.6's device
# rule does not apply to the numbers here.
#
# The job is CPU-only: it is numpy over fields that are already on disk.

#SBATCH --job-name=e43-s3-recompute
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"

echo "=========================================================="
echo " S3 recomputation   $(date -Is)"
echo "   job id : ${SLURM_JOB_ID}"
echo "=========================================================="

overall_rc=0

# --- E43: the low-rank + sparse arm ------------------------------------
echo
echo "--- E43 (A3 vs B0, seeds 0 1) -------------------------------------"
python -u "${REPO_ROOT}/tools/_e41_s3_equalbias.py" \
    --runs "${REPO_ROOT}/runs/e43" \
    --control B0 \
    --arms A3 \
    --seeds 0 1 \
    --epoch 00 \
    --out "${REPO_ROOT}/runs/e43/s3_equalbias_fixed.json"
rc=$?
echo "E43 rc=${rc}"
[[ ${rc} -ne 0 ]] && overall_rc=1

# --- E41: the output-parameterisation arm -------------------------------
echo
echo "--- E41 (A1 and A2 vs B0, seeds 0 1 2) ----------------------------"
python -u "${REPO_ROOT}/tools/_e41_s3_equalbias.py" \
    --runs "${REPO_ROOT}/runs/e41" \
    --control B0 \
    --arms A1 A2 \
    --seeds 0 1 2 \
    --epoch 00 \
    --out "${REPO_ROOT}/runs/e41/s3_equalbias_fixed.json"
rc=$?
echo "E41 rc=${rc}"
[[ ${rc} -ne 0 ]] && overall_rc=1

# --- E42: the loss-geometry arm ----------------------------------------
echo
echo "--- E42 (A2p vs B0, seeds 0 1 2) ----------------------------------"
python -u "${REPO_ROOT}/tools/_e41_s3_equalbias.py" \
    --runs "${REPO_ROOT}/runs/e42" \
    --control B0 \
    --arms A2p \
    --seeds 0 1 2 \
    --epoch 00 \
    --out "${REPO_ROOT}/runs/e42/s3_equalbias_fixed.json"
rc=$?
echo "E42 rc=${rc}"
[[ ${rc} -ne 0 ]] && overall_rc=1

echo
echo "=========================================================="
echo " S3 recomputation finished $(date -Is)"
echo "   overall rc = ${overall_rc}"
echo "=========================================================="

exit "${overall_rc}"
