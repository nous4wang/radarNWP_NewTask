#!/usr/bin/env bash
# Evaluate every kept epoch of the score-aware run, and the standard-loss
# run for reference.
#
# The old headline (CSI@2.0 = 0.118) came from a checkpoint that the
# training run was still overwriting.  Whether the score-aware objective
# peaks early and then decays, or whether 0.118 was a fluke, is a question
# about the epoch, so the epoch is the variable that has to be swept.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
CKEEP="${REPO_ROOT}/runs/epoch_curve"
TIERS="0.1 0.5 2.0 4.0"

run_dir="$(ls -dt "${CKEEP}"/seed0/*/ 2>/dev/null | head -1)"
if [[ -z "${run_dir}" ]]; then
  echo "no score-aware run directory under ${CKEEP}/seed0"
  exit 1
fi
echo "run: ${run_dir}"

gpu=0
for ck in "${run_dir}"epoch_*.pt; do
  [[ -f "${ck}" ]] || continue
  name="sa_$(basename "${ck}" .pt)"
  echo "  [${name}] on GPU ${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
        --model simvp_v2 \
        --checkpoint "${ck}" \
        --data-root "${DATA_ROOT}" \
        --packed --split test \
        --batch-size 16 --num-workers 4 \
        --patch-size 64 --slide-stride 64 \
        --thresholds ${TIERS} \
        --device cuda \
        --out-json "${CKEEP}/eval/${name}.eval.json" \
        > "${CKEEP}/eval/${name}.eval.log" 2>&1
  ) &
  gpu=$(( (gpu + 1) % 8 ))
  sleep 2
done
wait

std_dir="$(ls -dt "${CKEEP}"/std_seed0/*/ 2>/dev/null | head -1)"
if [[ -n "${std_dir}" && -f "${std_dir}last.pt" ]]; then
  echo "  [std_final] standard-loss final checkpoint on GPU 0"
  CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
      --model simvp_v2 \
      --checkpoint "${std_dir}last.pt" \
      --data-root "${DATA_ROOT}" \
      --packed --split test \
      --batch-size 16 --num-workers 4 \
      --patch-size 64 --slide-stride 64 \
      --thresholds ${TIERS} \
      --device cuda \
      --out-json "${CKEEP}/eval/std_final.eval.json" \
      > "${CKEEP}/eval/std_final.eval.log" 2>&1
fi

echo
echo "=== the score-aware objective, read off a known epoch ==="
"${PYTHON}" - "${CKEEP}/eval" <<'PYEOF'
import json
import sys
from pathlib import Path

rows = []
for p in sorted(Path(sys.argv[1]).glob("*.eval.json")):
    try:
        r = json.loads(p.read_text())
    except ValueError:
        continue
    t = r["thresholds"]
    rows.append((p.stem.replace(".eval", ""), t["0.1"]["csi"], t["0.5"]["csi"],
                 t["2.0"]["csi"], t["4.0"]["csi"], t["2.0"]["pod"], t["2.0"]["far"],
                 t["2.0"]["bias"], t["0.5"]["bias"], t["2.0"]["hits"], r.get("n_samples")))

def epoch_of(name):
    if name.startswith("sa_epoch_"):
        return int(name.split("_")[-1])
    return None

rows.sort(key=lambda r: (epoch_of(r[0]) is None, epoch_of(r[0]) or 0, r[0]))

hdr = (f"{'arm':14s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}{'CSI@4.0':>9s}"
       f"{'POD@2':>8s}{'FAR@2':>8s}{'bias@2':>8s}{'hits@2':>11s}{'n':>6s}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:14s}{r[1]:9.4f}{r[2]:9.4f}{r[3]:9.4f}{r[4]:9.4f}"
          f"{r[5]:8.4f}{r[6]:8.4f}{r[7]:8.4f}{r[8]:11.0f}{r[10]:6d}")

sa = [r for r in rows if epoch_of(r[0]) is not None]
if sa:
    best = max(sa, key=lambda r: r[3])
    print(f"\nbest score-aware epoch by CSI@2.0: {best[0]}  CSI@2.0={best[3]:.4f}")
    print(f"  spread over epochs: min={min(r[3] for r in sa):.4f}  "
          f"max={max(r[3] for r in sa):.4f}  "
          f"ratio={max(r[3] for r in sa)/max(min(r[3] for r in sa),1e-9):.1f}x")
PYEOF
echo
echo "=== done ==="
date '+%Y-%m-%d %H:%M:%S'
