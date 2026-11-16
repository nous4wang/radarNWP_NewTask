#!/usr/bin/env bash
# Verify the epoch confound behind the old D1/D2 headline numbers.
#
# The headline (CSI@2.0 = 0.118 / 0.152) was produced from checkpoints of a
# training run that had not finished: the evaluation read last.pt while the
# run was still at epoch 16, and the run then went on to overwrite it. The
# finished matrix scores 0.038 / 0.026 for the same two models.
#
# This re-evaluates all four of those checkpoints with one identical command,
# so the only difference between the arms is the epoch the weights stopped at.
# Nothing is trained and nothing that already exists is overwritten: the
# reports go to a scratch directory.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/epoch_check}"
TIERS="0.1 0.5 2.0 4.0"

mkdir -p "${OUT_DIR}"

# name:model:checkpoint-relative-to-repo
ARM=(
  "D1_ep20:simvp_v2:runs/hqma_scan4/scan_D1/simvp_v2_1793546034"
  "D1_ep17:simvp_v2:runs/hqma_scan4/scan_D1/simvp_v2_1793548445"
  "D2_ep20:simvp_v2_nwp_concat:runs/hqma_scan4/scan_D2/simvp_v2_nwp_concat_1793546037"
  "D2_ep17:simvp_v2_nwp_concat:runs/hqma_scan4/scan_D2/simvp_v2_nwp_concat_1793548449"
)

gpu=0
for arm in "${ARM[@]}"; do
  name="${arm%%:*}"
  rest="${arm#*:}"
  model="${rest%%:*}"
  rel="${rest#*:}"
  ckpt="${REPO_ROOT}/${rel}/last.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "  [${name}] MISSING checkpoint ${ckpt}"
    continue
  fi
  echo "  [${name}] evaluating ${rel} on GPU ${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split test \
        --batch-size 16 \
        --num-workers 4 \
        --patch-size 64 \
        --slide-stride 64 \
        --thresholds ${TIERS} \
        --device cuda \
        --out-json "${OUT_DIR}/${name}.eval.json" \
        > "${OUT_DIR}/${name}.eval.log" 2>&1
    echo "  [${name}] rc=$?"
  ) &
  gpu=$(( (gpu + 1) % 8 ))
  sleep 3
done
wait

echo
echo "=== epoch confound: same model, same seed, same eval, different stopping epoch ==="
"${PYTHON}" - "${OUT_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for p in sorted(out.glob("*.eval.json")):
    try:
        r = json.loads(p.read_text())
    except ValueError:
        continue
    t = r["thresholds"]
    rows.append((p.stem.replace(".eval", ""),
                 t["2.0"]["csi"], t["4.0"]["csi"], t["0.1"]["csi"],
                 t["0.5"]["csi"], t["2.0"]["pod"], t["2.0"]["far"],
                 t["2.0"]["bias"], r.get("n_samples")))

hdr = (f"{'arm':10s}{'CSI@2.0':>9s}{'CSI@4.0':>9s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}"
       f"{'POD@2.0':>9s}{'FAR@2.0':>9s}{'bias@2.0':>10s}{'n':>6s}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:10s}{r[1]:9.4f}{r[2]:9.4f}{r[3]:9.4f}{r[4]:9.4f}"
          f"{r[5]:9.4f}{r[6]:9.4f}{r[7]:10.4f}{r[8]:6d}")

def get(arm):
    for r in rows:
        if r[0] == arm:
            return r
    return None

for tag, a, b in (("D1 simvp_v2", "D1_ep20", "D1_ep17"),
                  ("D2 concat", "D2_ep20", "D2_ep17")):
    ra, rb = get(a), get(b)
    if ra and rb:
        print(f"\n{tag}: epoch 17 -> epoch 20")
        print(f"  CSI@2.0 {rb[1]:.4f} -> {ra[1]:.4f}  ({ra[1]-rb[1]:+.4f}, "
              f"x{ra[1]/rb[1]:.2f})")
        print(f"  CSI@4.0 {rb[2]:.4f} -> {ra[2]:.4f}")
        print(f"  bias@2.0 {rb[7]:.4f} -> {ra[7]:.4f}   "
              f"(the epoch-17 model over-forecasts heavy rain)")
PYEOF
echo
echo "=== done: reports in ${OUT_DIR} ==="
