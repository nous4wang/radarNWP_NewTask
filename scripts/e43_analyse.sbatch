#!/bin/bash
# E43 -- structure metrics (S1) and the equal-bias control (S3).
#
# A single CPU job that runs AFTER the evaluation array, because both
# analyses need the fields that the evaluation dumped.
#
# These two analyses are CPU-only by nature: they read the forecast
# fields that scripts/evaluate.py already wrote and measure them with
# numpy.  No model is run and no score is produced, so section 11.6
# rule S1 does not apply -- the numbers here are structure quantities
# (quantile ratios, pixel counts, dry fractions), which the section 2
# probe already measures on the CPU and which are unit-free by
# construction.
#
# The structure metrics use the SAME implementation as the section 2
# probe: tools/_e41_struct_metrics.py imports metrics() from
# tools/_struct_probe_one_case.py rather than copying it.  Writing a
# second, looser statistic would break the criterion.

#SBATCH --job-name=e43-analyse
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e43}"

echo "=========================================================="
echo " E43 analysis   $(date -Is)"
echo "   job id : ${SLURM_JOB_ID}"
echo "   python : $(command -v python)"
echo "=========================================================="

TAGS="B0_s0_ep00 B0_s1_ep00 A3_s0_ep00 A3_s1_ep00"

echo
echo "--- structure metrics (S1) --------------------------------------------"
python -u "${REPO_ROOT}/tools/_e41_struct_metrics.py" \
    --fields "${RUN_ROOT}/fields" \
    --tags ${TAGS} \
    --out "${RUN_ROOT}/struct_metrics.json"
struct_rc=$?
echo "struct rc=${struct_rc}"

echo
echo "--- equal-bias control (S3) ------------------------------------------"
python -u "${REPO_ROOT}/tools/_e41_s3_equalbias.py" \
    --runs "${RUN_ROOT}" \
    --control B0 \
    --arms A3 \
    --seeds 0 1 \
    --epoch 00 \
    --out "${RUN_ROOT}/s3_equalbias.json"
s3_rc=$?
echo "s3 rc=${s3_rc}"

echo
echo "--- score summary ----------------------------------------------------"
python -u "${REPO_ROOT}/tools/_e41_collect_scores.py" \
    --evals "${RUN_ROOT}/evals" \
    --out "${RUN_ROOT}/scores_summary.json"
sum_rc=$?
echo "summary rc=${sum_rc}"

echo
echo "=========================================================="
echo " analysis finished $(date -Is)"
echo "   struct rc = ${struct_rc}   s3 rc = ${s3_rc}   summary rc = ${sum_rc}"
echo "=========================================================="

exit $(( struct_rc | s3_rc | sum_rc ))
