#!/usr/bin/env bash
# Two things, both cheap, both needed before anything is written down.
#
# 1. A control re-evaluation of a checkpoint that has not changed since the
#    old reports were produced.  If the number comes back identical, then
#    scripts/evaluate.py has not changed and the difference between the old
#    headline and a fresh evaluation of the same path is a change in the
#    weights, not in the scoring.
#
# 2. A training run that keeps every epoch's weights, so the heavy-rain
#    score can be read off a known epoch instead of off a moment.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
TIERS="0.1 0.5 2.0 4.0"
CKEEP="${REPO_ROOT}/runs/epoch_curve"
CTRL="${REPO_ROOT}/runs/eval_control"

mkdir -p "${CKEEP}" "${CTRL}"

echo "=== mtimes of the scoring code and of the checkpoints involved ==="
stat -c '%y  %n' "${REPO_ROOT}/scripts/evaluate.py" "${REPO_ROOT}/scripts/train.py" 2>/dev/null
stat -c '%y  %n' \
  "${REPO_ROOT}/runs/tgrs/simvp_v2/simvp_v2_1793530231/last.pt" \
  "${REPO_ROOT}/runs/hqma_scan4/scan_D1/simvp_v2_1793548445/last.pt" 2>/dev/null
ls -1dt "${REPO_ROOT}"/runs/tgrs/simvp_v2/*/ 2>/dev/null
echo
echo "=== git state of the remote tree ==="
cd "${REPO_ROOT}" && git log --oneline -3 2>/dev/null || echo "(no git)"
git status --porcelain scripts/evaluate.py scripts/train.py 2>/dev/null || true
echo

echo "=== control: re-evaluate the standard-loss backbone and compare with the stored report ==="
CTRL_CKPT="$(ls -dt "${REPO_ROOT}"/runs/tgrs/simvp_v2/*/last.pt 2>/dev/null | head -1)"
echo "  checkpoint: ${CTRL_CKPT}"
CUDA_VISIBLE_DEVICES=0 \
"${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
    --model simvp_v2 \
    --checkpoint "${CTRL_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --packed --split test \
    --batch-size 16 --num-workers 4 \
    --patch-size 64 --slide-stride 64 \
    --thresholds ${TIERS} \
    --device cuda \
    --out-json "${CTRL}/tgrs_simvp_v2_recheck.eval.json" \
    > "${CTRL}/recheck.log" 2>&1
echo "  rc=$?"

"${PYTHON}" - "${CTRL}" "${REPO_ROOT}" <<'PYEOF'
import json
import sys
from pathlib import Path

ctrl = Path(sys.argv[1])
repo = Path(sys.argv[2])
new = json.loads((ctrl / "tgrs_simvp_v2_recheck.eval.json").read_text())
old = json.loads((repo / "runs/tgrs/simvp_v2.eval.json").read_text())
print(f"{'threshold':>10s}{'stored CSI':>12s}{'recheck CSI':>13s}{'stored hits':>13s}{'recheck hits':>14s}")
for t in ("0.1", "0.5", "2.0", "4.0"):
    a = old["thresholds"][t]
    b = new["thresholds"][t]
    print(f"{t:>10s}{a['csi']:12.6f}{b['csi']:13.6f}{a['hits']:13.0f}{b['hits']:14.0f}")
same = all(
    abs(old["thresholds"][t]["csi"] - new["thresholds"][t]["csi"]) < 1e-12
    for t in ("0.1", "0.5", "2.0", "4.0")
)
print(f"\nscoring pipeline unchanged: {'YES' if same else 'NO -- the code or the data moved'}")
PYEOF

echo
echo "=== the per-epoch run: simvp_v2 under the score-aware objective, seed 0 ==="
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
"${PYTHON}" "${REPO_ROOT}/scripts/train_keep_epochs.py" \
    --model simvp_v2 \
    --data-root "${DATA_ROOT}" \
    --packed --split train \
    --epochs 20 \
    --batch-size 16 --num-workers 6 \
    --patch-size 64 --slide-stride 64 \
    --lr 3e-4 \
    --loss balanced --alpha-csi 0.5 \
    --tier-thresholds ${TIERS} \
    --log-every 400 \
    --seed 0 \
    --device cuda \
    --out-dir "${CKEEP}/seed0" \
    > "${CKEEP}/seed0.log" 2>&1
echo "  training rc=$?"

echo
echo "=== and the same run under the standard objective, seed 0 ==="
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
"${PYTHON}" "${REPO_ROOT}/scripts/train.py" \
    --model simvp_v2 \
    --data-root "${DATA_ROOT}" \
    --packed --split train \
    --epochs 20 \
    --batch-size 16 --num-workers 6 \
    --patch-size 64 --slide-stride 64 \
    --lr 3e-4 \
    --log-every 400 \
    --seed 0 \
    --device cuda \
    --out-dir "${CKEEP}/std_seed0" \
    > "${CKEEP}/std_seed0.log" 2>&1
echo "  training rc=$?"

echo
echo "=== kept checkpoints ==="
ls -1 "${CKEEP}"/seed0/*/epoch_*.pt 2>/dev/null | wc -l
echo "=== done ==="
date '+%Y-%m-%d %H:%M:%S'
