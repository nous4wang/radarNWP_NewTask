#!/bin/bash
# E45 -- the equal-bias control (S3) and the cross-seed summary.
#
# The S3 control needs the control arm AND every other arm of the same
# seed, so it cannot live in a per-arm evaluation batch: it runs here,
# after all the evaluation batches of the seed have finished.
#
# The summary gathers the three seeds into the report format the
# project uses: the MEDIAN and the PASS RATE, never a single point
# score on its own.  The reason is measured, not stylistic: the ten
# seeds of base_concat at epoch 0 have a peak-end spread of 9.59x
# (0.0194-0.1865), so a single score cannot be read as a level.
#
# This job needs no GPU: the resampling is numpy over the dumped
# fields and the window tables, and the summary is over the JSON
# reports.  It therefore costs one CPU slot and leaves the GPUs free.

#SBATCH --job-name=e45-summary
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
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e45}"
EPOCH="${EPOCH:-00}"

echo "=========================================================="
echo " E45 summary   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   run root   : ${RUN_ROOT}"
echo "=========================================================="

# --- the equal-bias control, over every arm and every seed -------------
echo
echo "--- equal-bias control (S3) ------------------------------------------"
python -u "${REPO_ROOT}/tools/_e41_s3_equalbias.py" \
    --runs "${RUN_ROOT}" \
    --control B0 \
    --arms A1 A2 A3a A3b A4 A4b \
    --seeds 0 1 2 \
    --epoch "${EPOCH}" \
    --out "${RUN_ROOT}/s3_equalbias.json"
s3_rc=$?
echo "s3 rc=${s3_rc}"

# --- the structure quantities (S1), one file per seed --------------------
# The three seeds' structure metrics are merged into one file, because
# the S1 verdict compares an arm against the control over ALL the
# seeds -- a per-seed file would compare against a single seed's
# control and the median over the seeds would be missing.
echo
echo "--- merging the structure metrics of the three seeds ------------------"
python -u "${REPO_ROOT}/tools/_e45_merge_struct.py" \
    --inputs "${RUN_ROOT}/struct_metrics_s0.json" \
              "${RUN_ROOT}/struct_metrics_s1.json" \
              "${RUN_ROOT}/struct_metrics_s2.json" \
    --out "${RUN_ROOT}/struct_metrics_all.json"
merge_rc=$?
echo "merge rc=${merge_rc}"

echo
echo "--- S1 verdict: the four structure quantities, per arm ---------------"
if [[ -f "${RUN_ROOT}/struct_metrics_all.json" ]]; then
    python -u "${REPO_ROOT}/tools/_e41_s1_verdict.py" \
        --struct "${RUN_ROOT}/struct_metrics_all.json" \
        --scores "${RUN_ROOT}/scores_all.json" \
        --control B0
    s1_rc=$?
else
    echo "no merged structure metrics; S1 cannot be judged" >&2
    s1_rc=1
fi
echo "s1 rc=${s1_rc}"

# --- the summary: medians and pass rates, never a bare point score ----
echo
echo "--- the three-seed summary --------------------------------------------"
python -u "${REPO_ROOT}/tools/_e45_collect_scores.py" \
    --evals "${RUN_ROOT}/evals" \
    --out "${RUN_ROOT}/scores_summary.json"
sum_rc=$?
echo "summary rc=${sum_rc}"

# The per-tag scores, in the flat form the S1 verdict reads.
python -u "${REPO_ROOT}/tools/_e41_collect_scores.py" \
    --evals "${RUN_ROOT}/evals" \
    --out "${RUN_ROOT}/scores_all.json"
flat_rc=$?
echo "flat scores rc=${flat_rc}"

echo
echo "=========================================================="
echo " summary finished $(date -Is)"
echo "   s3 rc = ${s3_rc}   s1 rc = ${s1_rc}   merge rc = ${merge_rc}"
echo "   summary rc = ${sum_rc}   flat rc = ${flat_rc}"
echo "=========================================================="

exit $(( s3_rc | s1_rc | merge_rc | sum_rc | flat_rc ))
