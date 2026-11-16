#!/bin/bash
# E45 -- the EPOCH AXIS: evaluate the SAME 21 cells at a second and a third
# training epoch, under the frozen stage-2 protocol.
#
# WHY THIS JOB EXISTS
# -------------------
# E45 evaluated every cell at epoch 00 ONLY.  The rule "epoch 00 is the
# peak, last.pt is the collapse end" was never checked inside E45: the
# training loss falls monotonically and its argmin sits at epoch 02, which
# is the opposite end of the skill curve.  So the epoch choice is an
# ASSUMPTION the whole E45 table rests on, and it was never tested.
# Round 58 registered this as an open item and said it needs a separate
# Slurm job; the Slurm controller was down at the time.
#
# WHAT IT DOES NOT DO
# -------------------
# It does not re-run the E45 experiment and it does not touch E45's
# published artifacts.  It reads the checkpoints that training already
# produced, scores them with scripts/evaluate.py using exactly the
# arguments scripts/e45_stage2.sbatch uses, and writes EVERY output into
# a SEPARATE directory (runs/e45_epochaxis/).  runs/e45/ is read only.
#
# The epoch 00 re-run is the in-protocol control: it re-scores a cell
# E45 already published, under the same code, so that a difference
# between epochs cannot be confused with pipeline drift.
#
# The output is a diagnostic.  It does not create a criterion, it does
# not change E45's verdict, and no arm is promoted by it.

#SBATCH --job-name=e45-epochaxis
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
# The checkpoints E45's training already wrote.  READ ONLY.
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/runs/e45}"
# Every output goes here.  runs/e45/ is never written to.
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/e45_epochaxis}"

EPOCH="${EPOCH:-00}"
SEEDS="${SEEDS:-0 1 2}"

PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"

mkdir -p "${OUT_ROOT}/evals" "${OUT_ROOT}/windows" "${OUT_ROOT}/probe"

# The arm table: the ONE place the index -> arm mapping is written down.
source "${REPO_ROOT}/scripts/_e45_arms.sh"

echo "=========================================================="
echo " E45 epoch axis   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   epoch      : ${EPOCH}    seeds: ${SEEDS}"
echo "   checkpoint : ${CKPT_ROOT}   (read only)"
echo "   output     : ${OUT_ROOT}"
echo "=========================================================="

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
PY

overall_rc=0

for SEED in ${SEEDS}; do
    echo
    echo "############################################################"
    echo "# seed ${SEED}   epoch ${EPOCH}"
    echo "############################################################"

    # --- scoring ---------------------------------------------------------
    for IDX in $(seq 0 $((E45_N_ARMS - 1))); do
        ARM="$(e45_arm "${IDX}")"
        MODEL="$(e45_model "${IDX}")"
        CKPT="$(ls -1 "${CKPT_ROOT}/${ARM}_s${SEED}"/train/*/epoch_${EPOCH}.pt 2>/dev/null | head -1 || true)"
        if [[ -z "${CKPT}" ]]; then
            echo "FATAL: no epoch_${EPOCH} checkpoint for ${ARM} seed ${SEED}" >&2
            overall_rc=1
            continue
        fi
        TAG="${ARM}_s${SEED}_ep${EPOCH}"
        echo
        echo "=========================================================="
        echo " scoring ${TAG}   ($(date -Is))"
        echo "   checkpoint: ${CKPT}"
        echo "=========================================================="
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
            --out-json "${OUT_ROOT}/evals/${TAG}.eval.json" \
            --per-window-out "${OUT_ROOT}/windows/${TAG}.windows.json"
        eval_rc=$?
        echo "eval rc=${eval_rc} for ${TAG}"
        [[ "${eval_rc}" -ne 0 ]] && overall_rc=1
    done

    # --- mechanism probe (the discriminating read-outs) -------------------
    # Exactly the invocation e45_stage2.sbatch uses: one --checkpoint and
    # one --arm per mechanism arm, the arm's own loss flags inline.
    PROBE_ARGS=()
    for IDX in $(seq 0 $((E45_N_ARMS - 1))); do
        ARM="$(e45_arm "${IDX}")"
        [[ "${ARM}" == "B0" ]] && continue
        CKPT="$(ls -1 "${CKPT_ROOT}/${ARM}_s${SEED}"/train/*/epoch_${EPOCH}.pt 2>/dev/null | head -1 || true)"
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
        echo
        echo "--- mechanism probe, seed ${SEED} epoch ${EPOCH} -----------------"
        python -u "${REPO_ROOT}/tools/_e45_mechanism_probe.py" \
            --data-root "${DATA_ROOT}" \
            --out "${OUT_ROOT}/probe/diag_probe_s${SEED}_ep${EPOCH}.json" \
            "${PROBE_ARGS[@]}"
        probe_rc=$?
        echo "probe rc=${probe_rc}"
        [[ "${probe_rc}" -ne 0 ]] && overall_rc=1
    fi
done

echo
echo "--- summary of this epoch -----------------------------------------"
python -u - "${OUT_ROOT}" "${EPOCH}" <<'PY'
import json
import os
import sys

out_root, epoch = sys.argv[1], sys.argv[2]
rows = []
for name in sorted(os.listdir(os.path.join(out_root, "evals"))):
    if not name.endswith(".eval.json"):
        continue
    tag = name[: -len(".eval.json")]
    if not tag.endswith("_ep" + epoch):
        continue
    d = json.load(open(os.path.join(out_root, "evals", name), encoding="utf-8"))
    t = d["thresholds"]["2.0"]
    rows.append((tag, t["csi"], t["bias"], t["pod"], t["far"], d.get("n_samples")))
print(f"{'tag':22s} {'csi@2.0':>10s} {'bias@2.0':>10s} {'pod':>10s} {'far':>10s} {'n':>6s}")
for tag, csi, bias, pod, far, n in rows:
    print(f"{tag:22s} {csi:10.6f} {bias:10.4f} {pod:10.6f} {far:10.6f} {n:6d}")
PY

# One tarball per epoch, so the download is one file per epoch.
tar -czf "${OUT_ROOT}/epoch_axis_ep${EPOCH}.tar.gz" -C "${OUT_ROOT}" evals windows probe
md5sum "${OUT_ROOT}/epoch_axis_ep${EPOCH}.tar.gz"

echo
echo "=== E45 epoch axis (epoch ${EPOCH}) finished rc=${overall_rc} $(date -Is) ==="
exit "${overall_rc}"
