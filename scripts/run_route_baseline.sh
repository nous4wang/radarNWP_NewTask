#!/usr/bin/env bash
# The baseline arms, retrained under the current code.
#
# The routes are measured against these, so they have to come from the
# same code as the routes -- not from the matrix that was produced
# before the history rainfall was added to the dataset.
#
# Two baselines, on the two cards that are free:
#
#   simvp_v2            the backbone without the NWP channel
#   simvp_v2_nwp_concat  the best NWP arm under the score-aware objective
#
# Both use the score-aware objective, because that is the objective the
# routes are compared under.
#
# The acceptance criterion is a band, not a point. The run-to-run spread
# of the score-aware objective has been measured at 1.4 times on the
# strong-rain score: the same configuration and the same seed gave 0.0384
# in one run and 0.027 in another. A retrained baseline that lands
# outside [0.02, 0.06] on CSI@2.0 would mean something moved; inside
# that band it means the data layer did not.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_baseline}"
EPOCHS="${EPOCHS:-20}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
GPUS="${GPUS:-0 1}"

mkdir -p "${OUT_DIR}"

MODELS=(
    "simvp_v2 ${GPUS_0:-0}"
    "simvp_v2_nwp_concat ${GPUS_1:-1}"
)

echo "=========================================================="
echo " baseline arms, retrained under the current code"
echo " tiers=${TIERS} alpha_csi=${ALPHA_CSI} epochs=${EPOCHS}"
echo "=========================================================="

for entry in "${MODELS[@]}"; do
    read -r model gpu <<< "${entry}"
    log="${OUT_DIR}/${model}.log"
    echo "  -> ${model} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=4 \
    "${PYTHON}" scripts/train.py \
        --model "${model}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split train \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        --slide-stride "${SLIDE_STRIDE}" \
        --device cuda \
        --out-dir "${OUT_DIR}/${model}" \
        --loss balanced \
        --alpha-csi "${ALPHA_CSI}" \
        --tier-thresholds ${TIERS} \
        > "${log}" 2>&1 &
done

wait

echo "training done; scoring on the clean test split"

for entry in "${MODELS[@]}"; do
    read -r model gpu <<< "${entry}"
    ckpt="$(ls -dt "${OUT_DIR}/${model}"/*/last.pt 2>/dev/null | head -1)"
    if [[ -z "${ckpt}" ]]; then
        echo "  ${model}: no checkpoint, skipped"
        continue
    fi
    echo "  -> scoring ${model}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" scripts/evaluate.py \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split test \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        --slide-stride "${SLIDE_STRIDE}" \
        --device cuda \
        --thresholds ${TIERS} \
        --out-json "${OUT_DIR}/${model}.eval.json" \
        > "${OUT_DIR}/${model}.eval.log" 2>&1
done

echo "=========================================================="
echo " the summary"
echo "=========================================================="
"${PYTHON}" - "${OUT_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
print(f"{'model':24s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
      f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}{'n':>7s}")
print("-" * 84)
for name in ("simvp_v2", "simvp_v2_nwp_concat"):
    path = out_dir / f"{name}.eval.json"
    if not path.exists():
        print(f"{name:24s}(not finished)")
        continue
    report = json.loads(path.read_text())
    t = report["thresholds"]
    g = lambda th, key: t.get(th, {}).get(key, float("nan"))
    print(f"{name:24s}"
          + "".join(f"{g(th, 'csi'):9.3f}"
                    for th in ("0.1", "0.5", "2.0", "4.0"))
          + f"{g('0.5', 'far'):9.3f}{g('0.5', 'bias'):9.3f}"
          + f"{report.get('n_samples', 0):7d}")
PYEOF
