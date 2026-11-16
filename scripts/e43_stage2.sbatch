#!/bin/bash
# E43 -- score the low-rank + sparse ablation and run the two analyses.
#
# This is ONE job, not an array, and that is a deliberate change to the
# execution method (not to the criteria): the account's QoS allows 12
# submitted jobs and the training array already took the last of the
# allowance, so a four-task evaluation array plus a separate analysis
# job could not be submitted at all.  A single job that walks the four
# arms in turn needs one submitted slot and one GPU, and the queue
# accepts it where the array was refused.
#
# The order inside the job is fixed by the data flow: the structure
# metrics (S1) and the equal-bias control (S3) read the forecast fields
# that the scoring dumps, so they come after the scoring.  Section 11.6
# rule S1 says the main ruler for every evaluation number is the remote
# CUDA device; the threshold counting in the verifier is a discrete
# operation, so rule S2 applies as well and a CPU-scored number would
# have to be checked for cross-device consistency before it could carry
# any verdict.  Scoring on CUDA avoids the question and keeps these
# numbers comparable with every other table in the master document.
#
# The analysis at the end is pure numpy over the dumped fields: no model
# is run and no score is produced, so the S1 device rule does not apply
# to it -- the structure quantities are unit-free by construction, and
# the section 2 probe already measures them on the CPU.
#
# The structure metrics reuse the section 2 implementation: the tool
# imports metrics() from tools/_struct_probe_one_case.py rather than
# copying it.  Writing a second, looser statistic would break the
# criterion.

#SBATCH --job-name=e43-stage2
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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

EVAL_DIR="${RUN_ROOT}/evals"
FIELDS_DIR="${RUN_ROOT}/fields"
mkdir -p "${EVAL_DIR}" "${FIELDS_DIR}"

echo "=========================================================="
echo " E43 stage 2   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   python     : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "   run root   : ${RUN_ROOT}"
echo "=========================================================="

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
PY

# ---------------------------------------------------------------- scoring
# The arm and the seed table, identical to scripts/e43_train.sbatch.
score_one() {
    local ARM="$1" MODEL="$2" SEED="$3"
    local OUT_DIR="${RUN_ROOT}/${ARM}_s${SEED}"
    local TAG="${ARM}_s${SEED}_ep00"

    echo
    echo "=========================================================="
    echo " scoring ${ARM} seed ${SEED}  ($(date -Is))"
    echo "=========================================================="

    # The benchmark writes the checkpoint into a timestamped subdirectory,
    # so the epoch_00 file is found by glob rather than assumed.
    local CKPT
    CKPT="$(ls -1 "${OUT_DIR}"/train/*/epoch_00.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${CKPT}" ]]; then
        echo "FATAL: no epoch_00 checkpoint under ${OUT_DIR}/train" >&2
        return 3
    fi
    echo "checkpoint: ${CKPT}"

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
    local eval_rc=$?
    echo "eval rc=${eval_rc}"

    local REPORT="${EVAL_DIR}/${TAG}.eval.json"
    if [[ -f "${REPORT}" ]]; then
        if grep -qi "nan" "${REPORT}"; then
            echo "FATAL: the report contains NaN -- nothing was scored" >&2
            return 4
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
    else
        echo "FATAL: no report at ${REPORT}" >&2
        return 5
    fi
    return "${eval_rc}"
}

overall_rc=0
score_one "B0" "simvp_v2"  0 || overall_rc=1
score_one "B0" "simvp_v2"  1 || overall_rc=1
score_one "A3" "lr_sparse" 0 || overall_rc=1
score_one "A3" "lr_sparse" 1 || overall_rc=1

echo
echo "--- structure metrics (S1) --------------------------------------------"
python -u "${REPO_ROOT}/tools/_e41_struct_metrics.py" \
    --fields "${FIELDS_DIR}" \
    --tags B0_s0_ep00 B0_s1_ep00 A3_s0_ep00 A3_s1_ep00 \
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
    --evals "${EVAL_DIR}" \
    --out "${RUN_ROOT}/scores_summary.json"
sum_rc=$?
echo "summary rc=${sum_rc}"

echo
echo "=========================================================="
echo " stage 2 finished $(date -Is)"
echo "   overall rc = ${overall_rc}"
echo "   struct rc  = ${struct_rc}   s3 rc = ${s3_rc}   summary rc = ${sum_rc}"
echo "=========================================================="

exit $(( overall_rc | struct_rc | s3_rc | sum_rc ))
