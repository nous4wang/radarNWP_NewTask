#!/bin/bash
# E43 -- score one arm of the low-rank + sparse ablation.
#
# Submitted as a Slurm ARRAY; the array index picks the arm and the seed,
# exactly as in scripts/e43_train.sbatch.  The evaluation depends on the
# training job (--dependency=afterok), so it can only start once every
# training task finished successfully -- including the S0 gate, which
# runs inside each training task.
#
# The job asks for a GPU even though the verifier's cost is the Hann
# stitching and the FSS integrals: the master document's section 11.6
# rule S1 says the main ruler for every training and evaluation number
# is the remote CUDA device, and a CPU number may only serve as a
# diagnostic.  The threshold counting in the verifier is a discrete
# operation, so section 11.6 rule S2 applies as well: a CPU-scored
# number would have to be checked for cross-device consistency before it
# could carry any verdict.  Scoring on CUDA avoids the question entirely
# and keeps these numbers comparable with every other table in the
# master document.
#
# The fields are dumped for the structure metrics (S1) and for the
# equal-bias control (S3), which need the raw forecast, not just the
# score.

#SBATCH --job-name=e43-eval
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e43}"

PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "${IDX}" in
    0) ARM="B0"; MODEL="simvp_v2";  SEED=0 ;;
    1) ARM="B0"; MODEL="simvp_v2";  SEED=1 ;;
    2) ARM="A3"; MODEL="lr_sparse"; SEED=0 ;;
    3) ARM="A3"; MODEL="lr_sparse"; SEED=1 ;;
    *) echo "unknown array index ${IDX}" >&2; exit 2 ;;
esac

OUT_DIR="${RUN_ROOT}/${ARM}_s${SEED}"
EVAL_DIR="${RUN_ROOT}/evals"
FIELDS_DIR="${RUN_ROOT}/fields"
mkdir -p "${EVAL_DIR}" "${FIELDS_DIR}"

echo "=========================================================="
echo " E43 evaluation   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}   array: ${SLURM_ARRAY_TASK_ID:-none}"
echo "   arm        : ${ARM}   model: ${MODEL}   seed: ${SEED}"
echo "   python     : $(command -v python)"
echo "=========================================================="

# The benchmark writes the checkpoint into a timestamped subdirectory, so
# the epoch_00 file is found by glob rather than assumed.
CKPT="$(ls -1 "${OUT_DIR}"/train/*/epoch_00.pt 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
    echo "FATAL: no epoch_00 checkpoint under ${OUT_DIR}/train" >&2
    exit 3
fi
echo "checkpoint: ${CKPT}"

TAG="${ARM}_s${SEED}_ep00"

echo
echo "--- scoring -----------------------------------------------------------"
python -u "${REPO_ROOT}/scripts/evaluate.py" \
    --model "${MODEL}" \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split test \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --thresholds 0.1 0.5 2.0 4.0 \
    --device cuda \
    --out-json "${EVAL_DIR}/${TAG}.eval.json" \
    --per-window-out "${EVAL_DIR}/${TAG}.windows.json" \
    --dump-fields "${FIELDS_DIR}/${TAG}"
eval_rc=$?
echo "eval rc=${eval_rc}"

REPORT="${EVAL_DIR}/${TAG}.eval.json"
if [[ -f "${REPORT}" ]]; then
    if grep -qi "nan" "${REPORT}"; then
        echo "FATAL: the report contains NaN -- nothing was scored" >&2
        exit 4
    fi
    python - "${REPORT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
t = report["thresholds"]
print("CSI@2.0 :", round(t["2.0"]["csi"], 6))
print("BIAS@2.0:", round(t["2.0"]["bias"], 6))
print("CSI@4.0 :", round(t["4.0"]["csi"], 6))
print("windows :", report.get("n_samples"))
PY
    report_rc=$?
else
    echo "FATAL: no report at ${REPORT}" >&2
    report_rc=5
fi
echo "report rc=${report_rc}"

echo
echo "=========================================================="
echo " ${ARM} seed ${SEED} scored $(date -Is)"
echo "   eval rc = ${eval_rc}   report rc = ${report_rc}"
echo "   report   = ${REPORT}"
echo "=========================================================="

exit $(( eval_rc | report_rc ))
