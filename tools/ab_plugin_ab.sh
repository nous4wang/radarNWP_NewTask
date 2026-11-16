#!/usr/bin/env bash
# A/B the quantum-inspired NWP plugin against the plain baseline.
#
# The plugin is a plug-in, so the question is whether it helps: the same
# backbone, the same head, the same seed, the same budget -- and the only
# difference is the coupling.
#
# The two runs go side by side on separate cards, so they see the same
# machine and the same data at the same time.
#
# Usage:
#   bash tools/ab_plugin_ab.sh
set -uo pipefail

cd /home/wf/radarNWP
PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
OUT=${OUT:-/home/wf/radarNWP/runs/plugin_ab}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-16}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}

mkdir -p "${OUT}"

run_one() {
    local model="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON}" scripts/train.py \
        --model "${model}" \
        --data-root "${DATA}" \
        --packed \
        --split train \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH}" \
        --num-workers 6 \
        --patch-size 64 \
        --slide-stride 64 \
        --lr 3e-4 \
        --log-every 200 \
        --device cuda \
        --out-dir "${OUT}/${model}" \
        > "${OUT}/${model}.log" 2>&1
}

echo "=============================================================="
echo " A/B: simvp_v2  vs  simvp_v2_quantum"
echo " (GPU ${GPU_A} and GPU ${GPU_B}, ${EPOCHS} epochs each)"
echo "=============================================================="

run_one simvp_v2 "${GPU_A}" &
pid_a=$!
run_one simvp_v2_quantum "${GPU_B}" &
pid_b=$!
wait "${pid_a}" "${pid_b}"

echo
echo "=== loss curves ==="
for model in simvp_v2 simvp_v2_quantum; do
    echo "--- ${model} ---"
    grep -E '^epoch [0-9]+:' "${OUT}/${model}.log" | head -25
done

echo
echo "=== 评测 ==="
for model in simvp_v2 simvp_v2_quantum; do
    ckpt="$(ls -dt "${OUT}/${model}"/*/last.pt 2>/dev/null | head -1)"
    if [ -z "${ckpt}" ]; then
        echo "  [${model}] no checkpoint"
        continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU_A}" \
    "${PYTHON}" scripts/evaluate.py \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --data-root "${DATA}" \
        --packed \
        --split test \
        --batch-size "${BATCH}" \
        --num-workers 6 \
        --patch-size 64 \
        --slide-stride 64 \
        --device cuda \
        --out-json "${OUT}/${model}.eval.json" \
        > "${OUT}/${model}.eval.log" 2>&1
    echo "  [${model}] $(grep -E '^\s+0\.5 mm' "${OUT}/${model}.eval.log" | head -1 || echo evaluated)"
done

echo
"${PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
print("=" * 60)
print("  A/B 结果")
print("=" * 60)
rows = []
for tag in ("simvp_v2", "simvp_v2_quantum"):
    path = out / f"{tag}.eval.json"
    if not path.exists():
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    thresholds = report.get("thresholds", {})
    a = thresholds.get("0.5", {})
    b = thresholds.get("2.0", {})
    rows.append((tag, a.get("csi", float("nan")), b.get("csi", float("nan")),
                 a.get("fss", float("nan")), report.get("n_samples", 0)))
if rows:
    print(f"  {'model':22s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}{'FSS@0.5':>9s}{'n':>8s}")
    print("  " + "-" * 56)
    for tag, c1, c2, fss, n in rows:
        print(f"  {tag:22s}{c1:9.3f}{c2:9.3f}{fss:9.3f}{n:8d}")
    if len(rows) == 2:
        print()
        print(f"  Δ CSI@0.5 (plugin - plain) = {rows[1][1] - rows[0][1]:+.4f}")
        print(f"  Δ CSI@2.0 (plugin - plain) = {rows[1][2] - rows[0][2]:+.4f}")
PY
