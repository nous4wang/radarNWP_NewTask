#!/usr/bin/env bash
# The comparison that decides whether the new architecture is worth
# anything.
#
# The question is narrow and has to be answered against the right
# controls. The matrix already showed two things: the naive concatenation
# (M1) was the only NWP variant that was not behind its backbone, and
# every model on the benchmark predicts between two and a third of the
# rain that actually fell. So the new architecture has to beat M1 on the
# frequency bias without losing the rain it gets right.
#
# Four runs, all on the same cards, at the same time, from the same code:
#
#   hqma_standard    the new head, the benchmark's own loss
#   hqma_balanced    the new head, the score-aware loss
#   simvp_v2         the backbone, the benchmark's own loss   (the control)
#   simvp_v2_nwp_concat  M1, the benchmark's own loss        (the M1 control)
#
# The first two separate the architecture from the loss; the last two are
# already in the matrix and are repeated here so that every number in the
# comparison comes from the same run rather than from two different
# evenings.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/hqma}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-3e-4}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
GPUS="${GPUS:-0 1 2 3}"

read -r -a CARDS <<< "${GPUS}"

# name:model:loss
RUNS=(
    "hqma_standard:simvp_v2_hqma:standard"
    "hqma_balanced:simvp_v2_hqma:balanced"
    "simvp_v2:simvp_v2:standard"
    "simvp_v2_nwp_concat:simvp_v2_nwp_concat:standard"
)

mkdir -p "${OUT_DIR}"

train_one() {
    local name="$1" model="$2" loss="$3" gpu="$4"
    local rundir="${OUT_DIR}/${name}"
    local log="${OUT_DIR}/${name}.log"

    mkdir -p "${rundir}"
    echo "  -> ${name} (${model}, loss=${loss}) on GPU ${gpu}"
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
        --slide-stride "${SLIDE_STRIDE}" \
        --lr "${LR}" \
        --loss "${loss}" \
        --alpha-csi "${ALPHA_CSI}" \
        --log-every 200 \
        --device cuda \
        --out-dir "${rundir}" \
        > "${log}" 2>&1
    echo "  [${name}] $(grep -E '^epoch [0-9]+:' "${log}" | tail -1)"
}

echo "=== phase 1: training ==="
i=0
for entry in "${RUNS[@]}"; do
    IFS=: read -r name model loss <<< "${entry}"
    train_one "${name}" "${model}" "${loss}" "${CARDS[$(( i % ${#CARDS[@]} ))]}" &
    i=$(( i + 1 ))
    sleep 4
done
wait
echo "all training finished"

echo
echo "=== phase 2: evaluation (clean test split) ==="
i=0
for entry in "${RUNS[@]}"; do
    IFS=: read -r name model loss <<< "${entry}"
    ckpt="$(ls -dt "${OUT_DIR}/${name}"/*/last.pt 2>/dev/null | head -1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${name}] no checkpoint -- skipped"
        continue
    fi
    (
        CUDA_VISIBLE_DEVICES="${CARDS[$(( i % ${#CARDS[@]} ))]}" \
        "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
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
            --out-json "${OUT_DIR}/${name}.eval.json" \
            > "${OUT_DIR}/${name}.eval.log" 2>&1
        echo "  [${name}] evaluated"
    ) &
    i=$(( i + 1 ))
    sleep 2
done
wait

echo
echo "=== phase 3: the summary ==="
"${PYTHON}" - <<PYEOF
import json
from pathlib import Path

out = Path("${OUT_DIR}")
rows = []
for name in ["simvp_v2", "simvp_v2_nwp_concat", "hqma_standard", "hqma_balanced"]:
    path = out / f"{name}.eval.json"
    if not path.exists():
        continue
    r = json.loads(path.read_text())
    t = r["thresholds"]
    rows.append((
        name,
        t["0.1"]["csi"], t["0.5"]["csi"], t["2.0"]["csi"], t["4.0"]["csi"],
        t["0.5"]["far"], t["0.5"]["pod"], t["0.5"]["bias"],
    ))

print(f"{'run':24s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
      f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'POD@0.5':>9s}{'BIAS@0.5':>9s}")
print("-" * 96)
for row in rows:
    print(f"{row[0]:24s}" + "".join(f"{v:9.3f}" for v in row[1:]))
print()
print("the number to watch is BIAS: the models under-predict, and the")
print("architecture is meant to lift it towards the physical baseline's 0.90")
PYEOF

echo
echo "=== done ==="
