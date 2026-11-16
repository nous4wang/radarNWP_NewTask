#!/usr/bin/env bash
# Routes R1 and R2: the experiment matrix.
#
# The arms, and what each one is for:
#
#   base_concat   the best NWP arm under the score-aware objective.
#                 The comparison for R1, which does not change the task.
#   qpe_0          the R1 model with the history supervision off.
#                 It has the same architecture as the arm below, so the
#                 difference between them is the supervision and not the
#                 capacity. Without it the claim cannot be made.
#   qpe_05         R1 proper: the history rainfall supervises the
#                 inversion, and the inference-time input is unchanged.
#   assim_lag11    the R2 model with no observation used at inference.
#                 It is the member of the assimilation family that the
#                 pure radar baselines can be compared against.
#   assim_lag4     twelve of the last frames are missing.
#   assim_lag2     twenty-four of the last frames are missing.
#   assim_lag0     the whole history is observed: the upper bound.
#
# The assimilation arms are a family of their own. They consume ground
# observations at inference, so their scores are not comparable with a
# pure radar extrapolation and they are reported in their own table.
#
# Every arm uses the score-aware objective, because that is the
# objective the routes are meant to improve, and the plain regression
# puts every model at CSI@2.0 = 0.004 where nothing can be measured.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_r1r2}"
EPOCHS="${EPOCHS:-20}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
QPE_WEIGHT="${QPE_WEIGHT:-0.5}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

mkdir -p "${OUT_DIR}"

# name  model  gpu  lag  qpe_weight
# The lag is -1 for the arms that do not assimilate anything.
ARMS=(
    "base_concat simvp_v2_nwp_concat 0 -1 0.0"
    "qpe_0 simvp_v2_qpe_guided 1 -1 0.0"
    "qpe_05 simvp_v2_qpe_guided 2 -1 ${QPE_WEIGHT}"
    "assim_lag11 simvp_v2_assimilated 3 11 ${QPE_WEIGHT}"
    "assim_lag4 simvp_v2_assimilated 4 4 ${QPE_WEIGHT}"
    "assim_lag2 simvp_v2_assimilated 5 2 ${QPE_WEIGHT}"
    "assim_lag0 simvp_v2_assimilated 6 0 ${QPE_WEIGHT}"
)

echo "=========================================================="
echo " routes R1 and R2"
echo " tiers=${TIERS} alpha_csi=${ALPHA_CSI} qpe_weight=${QPE_WEIGHT}"
echo "=========================================================="

for arm in "${ARMS[@]}"; do
    read -r name model gpu lag qpe <<< "${arm}"
    log="${OUT_DIR}/${name}.log"
    echo "  -> ${name} (${model}) on GPU ${gpu}"
    # The lag goes into the constructor through the JSON override. The
    # JSON is built here rather than quoted in the table, because the
    # table is split on whitespace and the quotes would not survive it.
    if [[ "${lag}" -ge 0 ]]; then
        overrides="{\"obs_lag_steps\":${lag}}"
    else
        overrides=""
    fi
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
        --out-dir "${OUT_DIR}/${name}" \
        --tier-thresholds ${TIERS} \
        --loss balanced \
        --alpha-csi "${ALPHA_CSI}" \
        --qpe-weight "${qpe}" \
        ${overrides:+--model-kwargs "${overrides}"} \
        > "${log}" 2>&1 &
done

wait

echo "training done; scoring on the clean test split"

for arm in "${ARMS[@]}"; do
    read -r name model gpu lag qpe <<< "${arm}"
    ckpt="$(ls -dt "${OUT_DIR}/${name}"/*/last.pt 2>/dev/null | head -1)"
    if [[ -z "${ckpt}" ]]; then
        echo "  ${name}: no checkpoint, skipped"
        continue
    fi
    if [[ "${lag}" -ge 0 ]]; then
        overrides="{\"obs_lag_steps\":${lag}}"
    else
        overrides=""
    fi
    echo "  -> scoring ${name}"
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
        ${overrides:+--model-kwargs "${overrides}"} \
        --out-json "${OUT_DIR}/${name}.eval.json" \
        > "${OUT_DIR}/${name}.eval.log" 2>&1
done

echo "=========================================================="
echo " the summary"
echo "=========================================================="
"${PYTHON}" - "${OUT_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])

# The arms, in the order they are read. The first group shares the
# task with the radar baselines; the second does not.
RADAR_TASKS = [("base_concat", "M1 concat (same task)"),
                ("qpe_0", "R1 ablation: no history supervision"),
                ("qpe_05", "R1: history supervises the inversion")]
ASSIM_TASKS = [("assim_lag11", "R2: no observation (24 min late)"),
                ("assim_lag4", "R2: 24 min late"),
                ("assim_lag2", "R2: 12 min late"),
                ("assim_lag0", "R2: no delay")]

HEADER = (f"{'arm':16s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
          f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}{'n':>7s}")


def show(rows, title):
    print(f"\n{title}")
    print(HEADER)
    print("-" * len(HEADER))
    for name, label in rows:
        path = out_dir / f"{name}.eval.json"
        if not path.exists():
            print(f"{label:16s}(not finished)")
            continue
        report = json.loads(path.read_text())
        t = report["thresholds"]
        g = lambda th, key: t.get(th, {}).get(key, float("nan"))
        print(f"{label:16s}"
              + "".join(f"{g(th, 'csi'):9.3f}"
                        for th in ("0.1", "0.5", "2.0", "4.0"))
              + f"{g('0.5', 'far'):9.3f}{g('0.5', 'bias'):9.3f}"
              + f"{report.get('n_samples', 0):7d}")


show(RADAR_TASKS,
     "SAME TASK as the radar baselines (comparable with them)")
show(ASSIM_TASKS,
     "DIFFERENT TASK: these consume ground observations at inference")
print("\nThe persistence reference on the same split: CSI@2.0 = 0.0924.")
print("Anything below that has not learned to place the heavy rain.")
PYEOF
