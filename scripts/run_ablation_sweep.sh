#!/usr/bin/env bash
#
# The NWP ablation sweep: how much of the score is actually NWP?
#
# Three cells per model, and the difference between them is the answer:
#
#   with_nwp   trained with the NWP fields, evaluated with them
#               -> the reference. The full multimodal model.
#
#   zero_nwp   trained with the NWP fields, evaluated with them blanked
#               -> does the trained model *depend* on the dynamical and
#                  thermal fields, or did it learn to treat NWP as noise?
#
#   no_nwp     trained with the NWP fields withheld, evaluated the same way
#               -> the pure-radar ceiling. The difference from with_nwp is
#                  what the NWP is worth when it is available at training
#                  time.
#
# The two differences answer different questions and they are not
# interchangeable. A model can lose little under zero_nwp and still be
# beaten by the no_nwp model: that means it ignored the NWP during training
# but the NWP still carries information it could have used.
#
# The architectures are identical in all three cells -- the same number of
# weights, the same checkpoint layout, the same windows. Only the
# information in the NWP input differs. That is what makes the comparison
# an ablation rather than a contest between two different networks.
#
# Usage:
#   bash scripts/run_ablation_sweep.sh
#   MODELS="mocast metnet3" EPOCHS=5 bash scripts/run_ablation_sweep.sh
#
# Environment overrides:
#   DATA_ROOT     root of the packed (mmap) dataset
#   EPOCHS        epochs per run
#   PATCH_SIZE    patch side in radar pixels
#   SLIDE_STRIDE  sliding stride over the 301x301 grid (empty = centre crop)
#   BATCH_SIZE    samples per batch
#   WORKERS       dataloader workers per model
#   MODELS        space-separated model names (default: the paper baselines)
#   OUT_DIR       where the runs are written
#   N_GPUS        number of GPUs to spread the models over
#   PYTHON        the interpreter to use
#   REUSE         1 to reuse an existing training run instead of retraining
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
EPOCHS="${EPOCHS:-20}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
LR="${LR:-3e-4}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/ablation}"
N_GPUS="${N_GPUS:-8}"
REUSE="${REUSE:-0}"

# The eight paper baselines. The registry is discovered further down and
# printed, so a baseline that is added later shows up as a warning rather
# than as a silently missing row.
MODELS="${MODELS:-mocast diffcast cascast_nwp pw_foucast metnet3 fusioncast furecast mmf_rnn}"

SLIDE_ARG=()
if [[ -n "${SLIDE_STRIDE}" && "${SLIDE_STRIDE}" != "none" ]]; then
    SLIDE_ARG=(--slide-stride "${SLIDE_STRIDE}")
fi

mkdir -p "${OUT_DIR}"

echo "=============================================================="
echo " radar-nwp-bench :: NWP ablation sweep"
echo "=============================================================="
echo "  models      : ${MODELS}"
echo "  data root   : ${DATA_ROOT}"
echo "  epochs      : ${EPOCHS}"
echo "  patch/stride: ${PATCH_SIZE} / ${SLIDE_STRIDE:-none}"
echo "  batch/worker: ${BATCH_SIZE} / ${WORKERS}"
echo "  output      : ${OUT_DIR}"
echo "  gpus        : ${N_GPUS}"
echo

# The registry is the source of truth for which baselines exist. A name that
# is not registered is a typo, and a typo that silently produces an empty
# row in the results table is worse than an error.
REGISTERED="$(
    "${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from radar_nwp_bench.models import list_models
print(" ".join(list_models()))
PY
)"
echo "  registered  : ${REGISTERED}"
for model in ${MODELS}; do
    if ! printf '%s\n' ${REGISTERED} | grep -qx -- "${model}"; then
        echo "error: '${model}' is not a registered baseline." >&2
        echo "       registered: ${REGISTERED}" >&2
        exit 1
    fi
done
echo

# ------------------------------------------------------------------ helpers --

# The newest checkpoint of a run directory, or the empty string.
latest_checkpoint() {
    ls -dt "$1"/*/last.pt 2>/dev/null | head -1 || true
}

# Train one model in one mode. The mode is either "with_nwp" or "no_nwp";
# the difference is one flag, and it is the only difference.
train_one() {
    local model="$1" mode="$2" gpu="$3"
    local tag="${model}_${mode}"
    local rundir="${OUT_DIR}/${tag}"
    local log="${OUT_DIR}/${tag}.log"
    local flag=""
    if [[ "${mode}" == "no_nwp" ]]; then
        flag="--no-nwp"
    fi

    mkdir -p "${rundir}"
    echo "  -> ${model} [${mode}] on GPU ${gpu} (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" "${REPO_ROOT}/scripts/train.py" \
        --model "${model}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split train \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        "${SLIDE_ARG[@]}" \
        --lr "${LR}" \
        --log-every 200 \
        --device cuda \
        --out-dir "${rundir}" \
        ${flag} \
        > "${log}" 2>&1
}

# Evaluate one model on the held-out test cases and write the report.
# The tiles are stitched back into the 301x301 grid before scoring:
# scoring the tiles independently would double-count the overlap and split
# every storm that straddles a tile boundary.
evaluate_one() {
    local model="$1" mode="$2" checkpoint="$3" tag="$4"
    local flag=""
    if [[ "${mode}" == "no_nwp" ]]; then
        flag="--no-nwp"
    elif [[ "${mode}" == "zero_nwp" ]]; then
        flag="--zero-nwp"
    fi

    "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
        --model "${model}" \
        --checkpoint "${checkpoint}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split test \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        "${SLIDE_ARG[@]}" \
        --device cuda \
        --out-json "${OUT_DIR}/${tag}.eval.json" \
        ${flag} \
        > "${OUT_DIR}/${tag}.eval.log" 2>&1
}

# ---------------------------------------------------------------------------
# phase 1: train every model twice -- once with the NWP, once without
# ---------------------------------------------------------------------------
i=0
for model in ${MODELS}; do
    for mode in with_nwp no_nwp; do
        gpu=$(( i % N_GPUS ))
        rundir="${OUT_DIR}/${model}_${mode}"
        ckpt="$(latest_checkpoint "${rundir}")"
        if [[ "${REUSE}" == "1" && -n "${ckpt}" ]]; then
            echo "  -> ${model} [${mode}]: reusing ${ckpt}"
        else
            CUDA_VISIBLE_DEVICES="${gpu}" train_one "${model}" "${mode}" "${gpu}" &
            i=$(( i + 1 ))
        fi
    done
done
wait
echo
echo "all training runs finished"

# ---------------------------------------------------------------------------
# phase 2: evaluate the three cells
# ---------------------------------------------------------------------------
echo
echo "evaluating on the test split (tiles stitched back into the full grid)"
for model in ${MODELS}; do
    with_ckpt="$(latest_checkpoint "${OUT_DIR}/${model}_with_nwp")"
    no_ckpt="$(latest_checkpoint "${OUT_DIR}/${model}_no_nwp")"

    if [[ -n "${with_ckpt}" ]]; then
        # cell 1: the multimodal model, NWP available
        evaluate_one "${model}" with_nwp "${with_ckpt}" "${model}_with_nwp"
        echo "  [${model}] with_nwp  evaluated"
        # cell 2: the same weights, NWP blanked at inference
        evaluate_one "${model}" zero_nwp "${with_ckpt}" "${model}_zero_nwp"
        echo "  [${model}] zero_nwp  evaluated"
    else
        echo "  [${model}] no with_nwp checkpoint -- skipped"
    fi

    if [[ -n "${no_ckpt}" ]]; then
        # cell 3: the model that never saw the NWP
        evaluate_one "${model}" no_nwp "${no_ckpt}" "${model}_no_nwp"
        echo "  [${model}] no_nwp    evaluated"
    else
        echo "  [${model}] no no_nwp checkpoint -- skipped"
    fi
done

# ---------------------------------------------------------------------------
# phase 3: the comparison table
# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo " NWP ablation summary (test split)"
echo "=============================================================="
"${PYTHON}" - "${OUT_DIR}" "${MODELS}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
models = sys.argv[2].split()

NA = float("nan")


def read(tag):
    """Read one cell's report, returning the scores at 0.5 and 2.0 mm."""
    path = out / f"{tag}.eval.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    thresholds = report.get("thresholds", {})
    cell = {"mode": report.get("ablation", {}).get("mode", "?")}
    for thr in ("0.5", "2.0"):
        entry = thresholds.get(thr, {})
        cell[f"csi{thr}"] = entry.get("csi", NA)
        cell[f"fss{thr}"] = entry.get("fss", NA)
    return cell


def fmt(value):
    return "n/a" if value is None or value != value else f"{value:.3f}"


print(f"  {'model':14s} {'CSI@0.5':>8s} {'zero':>8s} {'no-nwp':>8s} "
      f"{'d(zero)':>8s} {'d(no)':>8s} | {'CSI@2.0':>8s} {'d(zero)':>8s} {'d(no)':>8s}")
print("  " + "-" * 88)

for model in models:
    full = read(f"{model}_with_nwp")
    zero = read(f"{model}_zero_nwp")
    none = read(f"{model}_no_nwp")
    if full is None:
        print(f"  {model:14s} {'no results':>8s}")
        continue

    def delta(other, key):
        if other is None:
            return NA
        a, b = full[key], other[key]
        return a - b if a == a and b == b else NA

    print(f"  {model:14s} {fmt(full['csi0.5']):>8s} {fmt(zero and zero['csi0.5']):>8s} "
          f"{fmt(none and none['csi0.5']):>8s} {fmt(delta(zero, 'csi0.5')):>8s} "
          f"{fmt(delta(none, 'csi0.5')):>8s} | {fmt(full['csi2.0']):>8s} "
          f"{fmt(delta(zero, 'csi2.0')):>8s} {fmt(delta(none, 'csi2.0')):>8s}")

print()
print("  d(zero) = with_nwp - zero_nwp : how much of the score the model")
print("                                 actually needed the NWP for")
print("  d(no)   = with_nwp - no_nwp   : how much the NWP was worth when it")
print("                                 was available during training")
print()
print(f"  reports: {out}/<model>_<cell>.eval.json")
PY
