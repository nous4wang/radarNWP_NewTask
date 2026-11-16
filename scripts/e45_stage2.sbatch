#!/bin/bash
# E45 -- score one seed's seven arms and run the analyses.
#
# ONE job per seed.  The seven arms of that seed are walked in turn, so
# no list of arms has to travel through the environment: the arm table
# (scripts/_e45_arms.sh) is read inside the job, and the seed -- a
# single integer -- is the only thing that was exported.
#
# That is not a stylistic choice.  Slurm separates the variables of an
# --export list with COMMAS, so an arm list passed that way arrives
# truncated to its first element: the first attempt of this batch
# scored one arm per job, left six unevaluated, and said nothing about
# it.  The list now never leaves the job.
#
# The order inside the job is fixed by the data flow: the structure
# metrics (S1) and the mechanism probe read the forecast fields that
# the scoring dumps, so they come after the scoring.  Section 11.6 rule
# S1 says the main ruler for every evaluation number is the remote CUDA
# device; the threshold counting in the verifier is a discrete
# operation, so rule S2 applies as well and a CPU-scored number would
# have to be checked for cross-device consistency before it could carry
# any verdict.  Scoring on CUDA avoids the question entirely and keeps
# these numbers comparable with every other table in the master
# document.
#
# The analysis at the end is pure numpy over the dumped fields: no model
# is run and no score is produced, so the S1 device rule does not apply
# to it -- the structure quantities are unit-free by construction, and
# the section 2 probe already measures them on the CPU.

#SBATCH --job-name=e45-stage2
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/e45}"

SEED="${SEED:-0}"
EPOCH="${EPOCH:-00}"

PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"

EVAL_DIR="${RUN_ROOT}/evals"
FIELDS_DIR="${RUN_ROOT}/fields"
mkdir -p "${EVAL_DIR}" "${FIELDS_DIR}"

# The arm table: the ONE place the index -> arm mapping is written down.
# The training script reads the same file, so the arm that was trained
# and the arm that is scored can never be different arms.
source "${REPO_ROOT}/scripts/_e45_arms.sh"

echo "=========================================================="
echo " E45 stage 2   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}   seed: ${SEED}"
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
# score_one <index>
score_one() {
    local IDX="$1"
    local ARM MODEL
    ARM="$(e45_arm "${IDX}")"
    MODEL="$(e45_model "${IDX}")"
    local OUT_DIR="${RUN_ROOT}/${ARM}_s${SEED}"
    local TAG="${ARM}_s${SEED}_ep${EPOCH}"

    echo
    echo "=========================================================="
    echo " scoring ${ARM} seed ${SEED}  ($(date -Is))"
    echo "=========================================================="

    # The benchmark writes the checkpoint into a timestamped subdirectory,
    # so the epoch_00 file is found by glob rather than assumed.
    local CKPT
    CKPT="$(ls -1 "${OUT_DIR}"/train/*/epoch_${EPOCH}.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${CKPT}" ]]; then
        echo "FATAL: no epoch_${EPOCH} checkpoint under ${OUT_DIR}/train" >&2
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
ARMS_SCORED=()
TAGS_TO_MEASURE=()
for IDX in $(seq 0 $((E45_N_ARMS - 1))); do
    ARM="$(e45_arm "${IDX}")"
    if score_one "${IDX}"; then
        ARMS_SCORED+=("${ARM}")
        TAGS_TO_MEASURE+=("${ARM}_s${SEED}_ep${EPOCH}")
    else
        overall_rc=1
    fi
done

# --- the mechanism probe --------------------------------------------------
# The discriminating read-outs.  A3a and A3b share the model topo_simvp
# and differ ONLY in the loss flags, and the probe rebuilds the term it
# measures -- so each arm's flags are passed inline, from the arm
# table's fifth column.
#
# The probe reads the arm's MODEL from its own table, so --model is not
# passed: one --model for several arms whose default models differ is
# exactly the mistake the probe refuses to make.  The checkpoint is
# named explicitly, because the probe reads the trained weights and a
# checkpoint that is missing is a mistake in the command line, not a
# result.
#
# B0 carries no mechanism read-out, so it is not probed.
echo
echo "--- mechanism probe (the discriminating read-outs) -------------------"
PROBE_ARGS=()
for IDX in $(seq 0 $((E45_N_ARMS - 1))); do
    ARM="$(e45_arm "${IDX}")"
    [[ "${ARM}" == "B0" ]] && continue
    CKPT="$(ls -1 "${RUN_ROOT}/${ARM}_s${SEED}"/train/*/epoch_${EPOCH}.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${CKPT}" ]]; then
        echo "FATAL: no epoch_${EPOCH} checkpoint for ${ARM} seed ${SEED}" >&2
        overall_rc=1
        continue
    fi
    PROBE_SPEC="${ARM}"
    PROBE_FLAGS="$(e45_field "${IDX}" 5)"
    if [[ -n "${PROBE_FLAGS}" ]]; then
        PROBE_SPEC="${ARM}|${PROBE_FLAGS}"
    fi
    PROBE_ARGS+=(--checkpoint "${CKPT}" --arm "${PROBE_SPEC}")
done
if [[ ${#PROBE_ARGS[@]} -gt 0 ]]; then
    python -u "${REPO_ROOT}/tools/_e45_mechanism_probe.py" \
        --data-root "${DATA_ROOT}" \
        --out "${RUN_ROOT}/diag_probe_s${SEED}.json" \
        "${PROBE_ARGS[@]}"
    probe_rc=$?
else
    echo "no mechanism arm to probe"
    probe_rc=0
fi
echo "probe rc=${probe_rc}"

# --- the structure quantities (S1) ---------------------------------------
echo
echo "--- structure metrics (S1) --------------------------------------------"
if [[ ${#TAGS_TO_MEASURE[@]} -gt 0 ]]; then
    python -u "${REPO_ROOT}/tools/_e41_struct_metrics.py" \
        --fields "${FIELDS_DIR}" \
        --tags "${TAGS_TO_MEASURE[@]}" \
        --out "${RUN_ROOT}/struct_metrics_s${SEED}.json"
    struct_rc=$?
else
    echo "no arm to measure"
    struct_rc=0
fi
echo "struct rc=${struct_rc}"

# The equal-bias control (S3) needs the control arm AND every other arm
# of the same seed, so it cannot run in a per-arm batch: it runs in the
# separate e45_summary job, which starts after all three evaluation
# jobs have finished.

echo
echo "--- score summary ----------------------------------------------------"
python -u "${REPO_ROOT}/tools/_e41_collect_scores.py" \
    --evals "${EVAL_DIR}" \
    --out "${RUN_ROOT}/scores_summary_s${SEED}.json"
sum_rc=$?
echo "summary rc=${sum_rc}"

echo
echo "=========================================================="
echo " stage 2 seed ${SEED} finished $(date -Is)"
echo "   overall rc = ${overall_rc}   probe rc = ${probe_rc}"
echo "   struct rc  = ${struct_rc}   summary rc = ${sum_rc}"
echo "=========================================================="

exit $(( overall_rc | probe_rc | struct_rc | sum_rc ))
