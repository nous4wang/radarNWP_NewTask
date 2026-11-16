#!/usr/bin/env bash
#
# Evaluate the trained baselines on the held-out test cases, several at once.
#
# The training phase spreads one model per GPU, but the evaluation in
# run_sweep.sh is sequential -- one model at a time on one card, while the
# other seven sit idle. Evaluating a model is the same kind of work as
# training it (the tiles have to be stitched back into the whole 301x301
# field before anything is scored), so it parallelises the same way.
#
# Models that already have a report are skipped, so this can be run again
# after an interruption without redoing the work.
#
# Usage:
#   bash scripts/run_parallel_eval.sh
#   MODELS="mocast metnet3" bash scripts/run_parallel_eval.sh
#
# Environment overrides:
#   DATA_ROOT     root of the packed (mmap) dataset
#   BATCH_SIZE    samples per batch
#   WORKERS       dataloader workers per model
#   OUT_DIR       where the runs and the reports live
#   GPUS          the GPU indices this script may use
#   MODELS        space-separated model names (default: every registered one)
#   PYTHON        the interpreter to use
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/sweep}"
N_GPUS="${N_GPUS:-8}"
GPUS="${GPUS:-$(seq 0 $(( N_GPUS - 1 )) | tr '\n' ' ')}"

SLIDE_ARG=()
if [[ -n "${SLIDE_STRIDE}" && "${SLIDE_STRIDE}" != "none" ]]; then
    SLIDE_ARG=(--slide-stride "${SLIDE_STRIDE}")
fi

mkdir -p "${OUT_DIR}"

# Every registered baseline, unless the caller names some.
if [[ -n "${MODELS:-}" ]]; then
    TARGETS="${MODELS}"
else
    TARGETS="$(
        "${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from radar_nwp_bench.models import list_models
print(" ".join(list_models()))
PY
    )"
fi

echo "=============================================================="
echo " radar-nwp-bench :: parallel evaluation"
echo "=============================================================="
echo "  data root : ${DATA_ROOT}"
echo "  output    : ${OUT_DIR}"
echo "  gpus      : ${GPUS}"
echo "  models    : ${TARGETS}"
echo

latest_checkpoint() {
    ls -dt "$1"/*/last.pt 2>/dev/null | head -1 || true
}

gpu_is_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "${1}" 2>/dev/null || echo 0)"
    [ "${used:-0}" -lt 100 ]
}

evaluate_one() {
    local model="$1" gpu="$2" checkpoint="$3"
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
        --out-json "${OUT_DIR}/${model}.eval.json" \
        > "${OUT_DIR}/${model}.eval.log" 2>&1
    echo "  [${model}] $(grep -E '^\s+0\.5 mm' "${OUT_DIR}/${model}.eval.log" | head -1 || echo 'evaluated')"
}

# ---------------------------------------------------------------------------
# phase 1: evaluate every model that needs it, spread over the free GPUs
# ---------------------------------------------------------------------------
GPU_LIST=(${GPUS})
i=0
for model in ${TARGETS}; do
    report="${OUT_DIR}/${model}.eval.json"
    if [[ -f "${report}" ]]; then
        echo "  -> ${model}: already evaluated, skipped"
        continue
    fi
    checkpoint="$(latest_checkpoint "${OUT_DIR}/${model}")"
    if [[ -z "${checkpoint}" ]]; then
        echo "  -> ${model}: no checkpoint -- skipped"
        continue
    fi

    gpu=""
    for _ in "${!GPU_LIST[@]}"; do
        candidate="${GPU_LIST[$(( (i + ${_}) % ${#GPU_LIST[@]} ))]}"
        if gpu_is_free "${candidate}"; then
            gpu="${candidate}"
            break
        fi
    done
    if [[ -z "${gpu}" ]]; then
        echo "  -> ${model}: no free GPU right now -- skipped"
        continue
    fi

    echo "  -> ${model} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" evaluate_one "${model}" "${gpu}" "${checkpoint}" &
    i=$(( i + 1 ))
    sleep 3
done
wait
echo
echo "all evaluations finished"

# ---------------------------------------------------------------------------
# phase 2: the ranking table
# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo " ranking (test split)"
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
    thresholds = report.get("thresholds", {})
    a = thresholds.get("0.5", {})
    b = thresholds.get("2.0", {})
    rows.append((
        path.stem.replace(".eval", ""),
        a.get("csi", float("nan")),
        b.get("csi", float("nan")),
        a.get("ets", float("nan")),
        a.get("hss", float("nan")),
        a.get("fss", float("nan")),
        report.get("n_samples", 0),
    ))

rows.sort(key=lambda r: (-r[1] if r[1] == r[1] else 0))
if not rows:
    print("  no evaluation results")
else:
    print(f"  {'model':22s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}{'ETS':>8s}"
          f"{'HSS':>8s}{'FSS':>8s}{'n':>8s}")
    print("  " + "-" * 72)
    for name, c1, c2, ets, hss, fss, n in rows:
        print(f"  {name:22s}{c1:9.3f}{c2:9.3f}{ets:8.3f}{hss:8.3f}"
              f"{fss:8.3f}{n:8d}")
    print(f"\n  {len(rows)} baselines evaluated")
PY
