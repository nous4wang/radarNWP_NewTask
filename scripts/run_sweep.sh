#!/usr/bin/env bash
# Run the baseline sweep on the benchmark server, one model per GPU.
#
# The eight 4090s are independent, so the sweep parallelises across models
# rather than across data: each model owns a GPU and a slice of the CPU
# workers. The models differ in cost by more than 2x, so the sweep is
# ordered longest-first and the short models fill the gaps.
#
# Environment overrides:
#   DATA_ROOT     root of the packed (mmap) dataset
#   EPOCHS        epochs per model
#   PATCH_SIZE    patch side in radar pixels
#   SLIDE_STRIDE  sliding stride over the 301x301 grid (empty = centre crop)
#   BATCH_SIZE    samples per batch
#   WORKERS       dataloader workers per model
#   MODELS        space-separated model names (default: the eight paper baselines)
#   OUT_DIR       where the runs are written
#   PYTHON        the interpreter to use
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

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

# The eight paper baselines, longest-running first (mocast's autoregressive
# loop and the two diffusion models are the slow ones).
MODELS="${MODELS:-mocast diffcast cascast_nwp pw_foucast metnet3 fusioncast furecast mmf_rnn}"

SLIDE_ARG=()
if [[ -n "${SLIDE_STRIDE}" && "${SLIDE_STRIDE}" != "none" ]]; then
    SLIDE_ARG=(--slide-stride "${SLIDE_STRIDE}")
fi

mkdir -p "${OUT_DIR}"
echo "=============================================================="
echo " radar-nwp-bench sweep"
echo "=============================================================="
echo "  models      : ${MODELS}"
echo "  data root   : ${DATA_ROOT}"
echo "  epochs      : ${EPOCHS}"
echo "  patch/stride: ${PATCH_SIZE} / ${SLIDE_STRIDE:-none}"
echo "  batch/worker: ${BATCH_SIZE} / ${WORKERS}"
echo "  output      : ${OUT_DIR}"
echo "  gpus        : ${N_GPUS}"
echo

# discover the registered baselines so a new one is picked up without edits
DISCOVERED="$(
    cd "${REPO_ROOT}"
    "${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from radar_nwp_bench.models import list_models
print(" ".join(list_models()))
PY
)"
echo "  registered  : ${DISCOVERED}"
echo

launch() {
    local model="$1" gpu="$2" slot="$3"
    local log="${OUT_DIR}/${model}.log"
    local rundir="${OUT_DIR}/${model}"

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
    echo "  [${model}] training finished (exit $?), log tail:"
    tail -3 "${log}" | sed 's/^/      /'
}

# ---------------------------------------------------------------------------
# phase 1: the eight paper baselines, one per GPU
# ---------------------------------------------------------------------------
i=0
for model in ${MODELS}; do
    gpu=$(( i % N_GPUS ))
    launch "${model}" "${gpu}" "${i}" &
    i=$(( i + 1 ))
done
wait
echo
echo "all training runs finished"

# ---------------------------------------------------------------------------
# phase 2: evaluate every trained model on the held-out test cases
# ---------------------------------------------------------------------------
echo
echo "evaluating on the test split (tiles stitched back into the full grid)"
for model in ${MODELS}; do
    ckpt="$(ls -dt "${OUT_DIR}/${model}"/*/last.pt 2>/dev/null | head -1)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] no checkpoint -- skipped"
        continue
    fi
    # The tiles are stitched back into the 301x301 grid before scoring:
    # scoring the tiles independently would double-count the overlap and
    # split every storm that straddles a tile boundary.
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
# phase 3: the summary table
# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo " summary (test split)"
echo "=============================================================="
"${PYTHON}" - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*.eval.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    thresholds = report.get("thresholds", {})
    thr = thresholds.get("0.5") or thresholds.get(0.5) or {}
    rows.append((
        path.stem.replace(".eval", ""),
        thr.get("csi", float("nan")),
        thr.get("ets", float("nan")),
        thr.get("hss", float("nan")),
        thr.get("pod", float("nan")),
        thr.get("far", float("nan")),
        report.get("n_samples", 0),
    ))

rows.sort(key=lambda r: (-r[1] if r[1] == r[1] else 0))
if not rows:
    print("  no evaluation results")
else:
    print(f"  {'model':14s} {'CSI':>7s} {'ETS':>7s} {'HSS':>7s} {'POD':>7s} {'FAR':>7s} {'n':>6s}")
    print("  " + "-" * 60)
    for name, csi, ets, hss, pod, far, n in rows:
        print(f"  {name:14s} {csi:7.3f} {ets:7.3f} {hss:7.3f} {pod:7.3f} {far:7.3f} {n:6d}")
PY
echo
echo "artifacts in ${OUT_DIR}"
