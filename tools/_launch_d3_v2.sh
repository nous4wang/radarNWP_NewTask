#!/usr/bin/env bash
# D3 (direction 3) RE-RUN with epoch-keeping wrapper.
#
# Why this exists: the first D3 launch used scripts/train.py, which writes only
# last.pt (= epoch 2 final).  Section 23 says the peak is at epoch 0, and the
# pre-registered D3-A criterion (MASTER 4.20.2) targets ep00, so ep00/ep01
# weights must be recovered.  scripts/train_keep_epochs.py wraps the SAME
# train.py CLI and patches torch.save to also emit epoch_00/01/02.pt.
#
# One-variable contract: gate / nogate differ ONLY by --model-kwargs use_gate.
# v1 reference (section 4.19) stays untouched at runs/physical_qpe/.
#
# Self-contained: no separate watcher (the old d3_watch waited 23 min on a
# file that could never appear).  Every stage has a marker in the log.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
CFG=configs/models/physical_qpe_v2.yaml

echo "D3_V2_START $(date)"

DATA_ARGS="--data-root /home/wf/data/radarNWP/BJ_mmap --packed"
TRAIN_ARGS="--model physical_qpe_v2 --config ${CFG} ${DATA_ARGS} --epochs 3 --loss balanced --qpe-single-frame --seed 0 --patch-size 64 --slide-stride 64 --batch-size 16 --num-workers 4 --alignment forecast"

# ---------------------------------------------------------------- stage 1: re-train
echo "STAGE1_TRAIN_START $(date)"
"$PY" scripts/train_keep_epochs.py ${TRAIN_ARGS} \
  --model-kwargs '{"use_gate": true}' \
  --out-dir runs/physical_qpe_v2/gate/seed0_keep \
  > runs/physical_qpe_v2_train_gate_keep.log 2>&1
echo "GATE_TRAIN_EXIT=$? $(date)"

"$PY" scripts/train_keep_epochs.py ${TRAIN_ARGS} \
  --model-kwargs '{"use_gate": false}' \
  --out-dir runs/physical_qpe_v2/nogate/seed0_keep \
  > runs/physical_qpe_v2_train_nogate_keep.log 2>&1
echo "NOGATE_TRAIN_EXIT=$? $(date)"

GATE_CKPT=$(find runs/physical_qpe_v2/gate/seed0_keep -name epoch_00.pt -type f | head -1)
NOGATE_CKPT=$(find runs/physical_qpe_v2/nogate/seed0_keep -name epoch_00.pt -type f | head -1)
if [ -z "${GATE_CKPT}" ] || [ -z "${NOGATE_CKPT}" ]; then
  echo "D3_V2_ABORT: missing epoch_00.pt (gate='${GATE_CKPT}' nogate='${NOGATE_CKPT}') $(date)"
  exit 1
fi
GATE_RUN=$(dirname "${GATE_CKPT}")
NOGATE_RUN=$(dirname "${NOGATE_CKPT}")
echo "GATE_RUN=${GATE_RUN} NOGATE_RUN=${NOGATE_RUN} $(date)"

# ---------------------------------------------------------------- stage 2: determinism
# MASTER 35/38: same seed + same code => bit-identical weights, independent of
# training budget.  Compare the new epoch_02.pt with the OLD train.py last.pt.
#
# WARNING (learned the hard way): torch.save is NOT byte-reproducible (zip
# metadata), so FILE md5 is the WRONG tool -- it reports MATCH=NO even when the
# tensors are identical.  Always torch.load and compare tensors key by key.
echo "STAGE2_DETERMINISM $(date)"
"$PY" - <<PYEOF
import glob, torch
pairs = [
    ("gate",   "runs/physical_qpe_v2/gate/seed0/*/last.pt",
               "${GATE_RUN}/epoch_02.pt"),
    ("nogate", "runs/physical_qpe_v2/nogate/seed0/*/last.pt",
               "${NOGATE_RUN}/epoch_02.pt"),
]
for arm, old_pat, new in pairs:
    old = glob.glob(old_pat)[0]
    a, b = torch.load(old, map_location="cpu", weights_only=True), torch.load(new, map_location="cpu", weights_only=True)
    diffs = [float((a[k].float() - b[k].float()).abs().max()) for k in a]
    print(f"{arm}: same_keys={set(a)==set(b)} n={len(a)} max|diff|={max(diffs):.3e} old={old}")
PYEOF

# ---------------------------------------------------------------- stage 3: eval (6 arms)
echo "STAGE3_EVAL_START $(date)"
mkdir -p runs/physical_qpe_v2/gate/eval runs/physical_qpe_v2/nogate/eval
PIDS=""
for ARM in gate nogate; do
  if [ "${ARM}" = "gate" ]; then
    GK='{"use_gate": true}'
    RUN="${GATE_RUN}"
  else
    GK='{"use_gate": false}'
    RUN="${NOGATE_RUN}"
  fi
  for EP in 00 01 02; do
    "$PY" scripts/evaluate.py --model physical_qpe_v2 --config ${CFG} ${DATA_ARGS} \
      --model-kwargs "${GK}" \
      --checkpoint "${RUN}/epoch_${EP}.pt" \
      --patch-size 64 --slide-stride 64 --alignment forecast \
      --thresholds 0.1 0.5 2.0 4.0 \
      --out-json runs/physical_qpe_v2/${ARM}/eval/epoch_${EP}.eval.json \
      --per-window-out runs/physical_qpe_v2/${ARM}/eval/epoch_${EP}.windows.json \
      > runs/physical_qpe_v2_${ARM}_ep${EP}.eval.log 2>&1 &
    PIDS="${PIDS} $!"
  done
done
for P in ${PIDS}; do
  wait ${P}
  echo "EVAL_PID ${P} EXIT=$? $(date)"
done

# ---------------------------------------------------------------- stage 4: paired bootstrap
# Per-window tables are ~318 kB > SSH 256 kB truncation => must run REMOTE.
echo "STAGE4_BOOTSTRAP_START $(date)"
BOOT=tools/_bootstrap_qpe_vs_zr.py
V1=runs/physical_qpe/eval/epoch_00.windows.json
G=runs/physical_qpe_v2/gate/eval/epoch_00.windows.json
N=runs/physical_qpe_v2/nogate/eval/epoch_00.windows.json
ALL_OK=1
for F in "${V1}" "${G}" "${N}"; do
  if [ ! -f "${F}" ]; then echo "MISSING ${F}"; ALL_OK=0; fi
done
if [ "${ALL_OK}" = "1" ]; then
  "$PY" ${BOOT} --a "${G}" --b "${V1}" --name-a gate_ep00 --name-b v1_ep00 --n-boot 10000 \
    > runs/bootstrap_d3a_gate_vs_v1.log 2>&1
  echo "D3A_EXIT=$? $(date)"
  "$PY" ${BOOT} --a "${G}" --b "${N}" --name-a gate_ep00 --name-b nogate_ep00 --n-boot 10000 \
    > runs/bootstrap_d3b_gate_vs_nogate.log 2>&1
  echo "D3B_EXIT=$? $(date)"
  "$PY" ${BOOT} --a "${N}" --b "${V1}" --name-a nogate_ep00 --name-b v1_ep00 --n-boot 10000 \
    > runs/bootstrap_d3_check_nogate_vs_v1.log 2>&1
  echo "D3CHECK_EXIT=$? $(date)"
else
  echo "STAGE4_SKIPPED: per-window tables incomplete $(date)"
fi

# ---------------------------------------------------------------- stage 5: summary
echo "=== CSI@2.0 SUMMARY ==="
"$PY" -c "
import json
for arm in ('gate','nogate'):
  for ep in ('00','01','02'):
    p='runs/physical_qpe_v2/%s/eval/epoch_%s.eval.json'%(arm,ep)
    try:
      d=json.load(open(p))
      print('%s ep%s CSI@2.0=%.4f n=%d'%(arm,ep,d['thresholds']['2.0']['csi'],d['n_samples']))
    except Exception as e:
      print('%s ep%s MISSING (%s)'%(arm,ep,e))
"
echo "=== BOOTSTRAP LOGS ==="
for F in runs/bootstrap_d3a_gate_vs_v1.log runs/bootstrap_d3b_gate_vs_nogate.log runs/bootstrap_d3_check_nogate_vs_v1.log; do
  echo "--- ${F} ---"; tail -20 "${F}" 2>/dev/null
done
echo "D3_V2_DONE $(date)"
