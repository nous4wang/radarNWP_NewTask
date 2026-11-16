#!/usr/bin/env bash
#
# Train and evaluate the baselines that the main sweep does not cover.
#
# The registry has grown past the eight paper baselines: there are fifteen
# registered models now, and the sweep script covers eight of them. This
# script runs the remainder on the GPUs that are free, so the two can sit
# side by side on the eight-card server without either waiting for the other.
#
# The models are discovered from the registry rather than listed here, and
# the ones the main sweep already covers are skipped, so this script stays
# correct as the registry changes.
#
# Usage:
#   bash scripts/run_extra_baselines.sh
#   MODELS="convlstm predrnn" bash scripts/run_extra_baselines.sh
#
# Environment overrides:
#   DATA_ROOT     root of the packed (mmap) dataset
#   EPOCHS        epochs per model
#   PATCH_SIZE    patch side in radar pixels
#   SLIDE_STRIDE  sliding stride over the 301x301 grid (empty = centre crop)
#   BATCH_SIZE    samples per batch
#   WORKERS       dataloader workers per model
#   MODELS        space-separated model names (default: every registered
#                 baseline that the main sweep does not cover)
#   OUT_DIR       where the runs are written
#   GPUS          the GPU indices this script may use (space separated)
#   PYTHON        the interpreter to use
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
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/sweep}"
N_GPUS="${N_GPUS:-8}"
# The GPU indices this script may use. It defaults to the first N_GPUS of
# them, but the list has to be explicit when another sweep is already
# running on the same server: putting two jobs on one card halves both.
GPUS="${GPUS:-$(seq 0 $(( N_GPUS - 1 )) | tr '\n' ' ')}"

# The eight baselines the main sweep covers. Everything else in the registry
# is run here.
SWEEP_MODELS="${SWEEP_MODELS:-mocast diffcast cascast_nwp pw_foucast metnet3 fusioncast furecast mmf_rnn}"

SLIDE_ARG=()
if [[ -n "${SLIDE_STRIDE}" && "${SLIDE_STRIDE}" != "none" ]]; then
    SLIDE_ARG=(--slide-stride "${SLIDE_STRIDE}")
fi

mkdir -p "${OUT_DIR}"

echo "=============================================================="
echo " radar-nwp-bench :: the baselines the main sweep does not cover"
echo "=============================================================="
echo "  data root   : ${DATA_ROOT}"
echo "  epochs      : ${EPOCHS}"
echo "  patch/stride: ${PATCH_SIZE} / ${SLIDE_STRIDE:-none}"
echo "  batch/worker: ${BATCH_SIZE} / ${WORKERS}"
echo "  output      : ${OUT_DIR}"
echo

# The registry is the source of truth. A model that is already covered by
# the main sweep, or that has already been trained here, is skipped.
REGISTERED="$(
    "${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from radar_nwp_bench.models import list_models
print(" ".join(list_models()))
PY
)"
echo "  registered  : ${REGISTERED}"
echo "  already in the main sweep: ${SWEEP_MODELS}"
echo

if [[ -n "${MODELS:-}" ]]; then
    TARGETS="${MODELS}"
else
    TARGETS=""
    for model in ${REGISTERED}; do
        skip=0
        for done in ${SWEEP_MODELS}; do
            if [[ "${model}" == "${done}" ]]; then
                skip=1
                break
            fi
        done
        if [[ "${skip}" == "0" ]]; then
            TARGETS="${TARGETS} ${model}"
        fi
    done
    TARGETS="${TARGETS# }"
fi

if [[ -z "${TARGETS}" ]]; then
    echo "nothing to do: every registered baseline is already covered"
    exit 0
fi
echo "  to run      : ${TARGETS}"
echo

latest_checkpoint() {
    ls -dt "$1"/*/last.pt 2>/dev/null | head -1 || true
}

train_one() {
    local model="$1" gpu="$2"
    local rundir="${OUT_DIR}/${model}"
    local log="${OUT_DIR}/${model}.log"

    mkdir -p "${rundir}"
    echo "  -> ${model} on GPU ${gpu} (log: ${log})"
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
        > "${log}" 2>&1
    echo "  [${model}] training finished, log tail:"
    tail -3 "${log}" | sed 's/^/      /'
}

# A GPU is available when nothing is already running on it. Another sweep on
# the same server owns its cards, and putting a second job on one of them
# halves the speed of both.
gpu_is_free() {
    local gpu="$1"
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "${gpu}" 2>/dev/null || echo 0)"
    # A few MiB is the driver's own allocation; a running job holds hundreds.
    [ "${used:-0}" -lt 100 ]
}

# ---------------------------------------------------------------------------
# phase 1: train them, spread over the free GPUs
# ---------------------------------------------------------------------------
GPU_LIST=(${GPUS})
echo "  allowed gpus: ${GPU_LIST[*]}"
echo

i=0
for model in ${TARGETS}; do
    ckpt="$(latest_checkpoint "${OUT_DIR}/${model}")"
    if [[ -n "${ckpt}" ]]; then
        echo "  -> ${model}: already trained (${ckpt}), skipped"
        continue
    fi

    # Find a free card, starting from where the last model landed. If every
    # allowed card is busy -- another sweep is holding them -- the model waits
    # for one to free up rather than being dropped: a model that is silently
    # not run is a hole in the final table.
    gpu=""
    for attempt in $(seq 1 240); do
        for _ in "${!GPU_LIST[@]}"; do
            candidate="${GPU_LIST[$(( (i + ${_}) % ${#GPU_LIST[@]} ))]}"
            if gpu_is_free "${candidate}"; then
                gpu="${candidate}"
                break
            fi
        done
        if [[ -n "${gpu}" ]]; then
            break
        fi
        if [ "${attempt}" = "1" ] || [ $(( attempt % 30 )) -eq 0 ]; then
            echo "  -> ${model}: all allowed GPUs busy, waiting ..."
            sleep 60
        else
            sleep 15
        fi
    done
    if [[ -z "${gpu}" ]]; then
        echo "  -> ${model}: no free GPU after waiting -- skipped"
        continue
    fi

    train_one "${model}" "${gpu}" &
    i=$(( i + 1 ))
    # Give the job a moment to claim its card before the next one looks.
    sleep 3
done
wait
echo
echo "all training runs finished"

# ---------------------------------------------------------------------------
# phase 2: evaluate every model that now has a checkpoint
# ---------------------------------------------------------------------------
echo
echo "evaluating on the test split (tiles stitched back into the full grid)"
eval_gpu=""
for candidate in ${GPU_LIST}; do
    if gpu_is_free "${candidate}"; then
        eval_gpu="${candidate}"
        break
    fi
done
if [[ -z "${eval_gpu}" ]]; then
    eval_gpu="${GPU_LIST[0]}"
fi
echo "  (evaluating on GPU ${eval_gpu})"
for model in ${TARGETS}; do
    ckpt="$(latest_checkpoint "${OUT_DIR}/${model}")"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        continue
    fi
    CUDA_VISIBLE_DEVICES="${eval_gpu}" \
    "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split test \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        "${SLIDE_ARG[@]}" \
        --device cuda \
        --out-json "${OUT_DIR}/${model}.eval.json" \
        > "${OUT_DIR}/${model}.eval.log" 2>&1
    echo "  [${model}] $(grep -E '^\s+0\.5 mm' "${OUT_DIR}/${model}.eval.log" | head -1 || echo 'evaluated')"
done

# ---------------------------------------------------------------------------
# phase 3: the ranking table over everything that has been evaluated
# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo " ranking (test split, every evaluated baseline)"
echo "=============================================================="
"${PYTHON}" - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*.eval.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    thr = report.get("thresholds", {}).get("0.5", {})
    thr2 = report.get("thresholds", {}).get("2.0", {})
    rows.append((
        path.stem.replace(".eval", ""),
        thr.get("csi", float("nan")),
        thr2.get("csi", float("nan")),
        thr.get("ets", float("nan")),
        thr.get("hss", float("nan")),
        thr.get("fss", float("nan")),
        report.get("n_samples", 0),
    ))

rows.sort(key=lambda r: (-r[1] if r[1] == r[1] else 0))
if not rows:
    print("  no evaluation results")
else:
    print(f"  {'model':20s} {'CSI@0.5':>8s} {'CSI@2.0':>8s} {'ETS':>7s} "
          f"{'HSS':>7s} {'FSS':>7s} {'n':>6s}")
    print("  " + "-" * 68)
    for name, csi, csi2, ets, hss, fss, n in rows:
        print(f"  {name:20s} {csi:8.3f} {csi2:8.3f} {ets:7.3f} {hss:7.3f} "
              f"{fss:7.3f} {n:6d}")
    print()
    print(f"  {len(rows)} baselines evaluated")
PY
