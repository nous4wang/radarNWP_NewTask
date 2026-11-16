#!/usr/bin/env bash
# D3 (direction 3): physical_qpe_v2 gate on/off paired ablation.
# One-variable contract: the two arms differ ONLY by --model-kwargs use_gate.
# v1 reference (section 4.19) stays untouched at runs/physical_qpe/.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
CFG=configs/models/physical_qpe_v2.yaml

echo "D3_START $(date)"

mkdir -p runs/physical_qpe_v2/gate/eval runs/physical_qpe_v2/nogate/eval

DATA_ARGS="--data-root /home/wf/data/radarNWP/BJ_mmap --packed"

TRAIN_ARGS="--model physical_qpe_v2 --config ${CFG} ${DATA_ARGS} --epochs 3 --loss balanced --qpe-single-frame --seed 0 --patch-size 64 --slide-stride 64 --batch-size 16 --num-workers 4 --alignment forecast"

"$PY" scripts/train.py ${TRAIN_ARGS} \
  --model-kwargs '{"use_gate": true}' \
  --out-dir runs/physical_qpe_v2/gate/seed0 \
  > runs/physical_qpe_v2_train_gate.log 2>&1
echo "GATE_TRAIN_EXIT=$? $(date)"

"$PY" scripts/train.py ${TRAIN_ARGS} \
  --model-kwargs '{"use_gate": false}' \
  --out-dir runs/physical_qpe_v2/nogate/seed0 \
  > runs/physical_qpe_v2_train_nogate.log 2>&1
echo "NOGATE_TRAIN_EXIT=$? $(date)"

GATE_CKPT=$(find runs/physical_qpe_v2/gate/seed0 -name epoch_00.pt -type f | head -1)
NOGATE_CKPT=$(find runs/physical_qpe_v2/nogate/seed0 -name epoch_00.pt -type f | head -1)
if [ -z "${GATE_CKPT}" ] || [ -z "${NOGATE_CKPT}" ]; then
  echo "D3_ABORT: missing epoch_00.pt (gate='${GATE_CKPT}' nogate='${NOGATE_CKPT}') $(date)"
  exit 1
fi
GATE_RUN=$(dirname "${GATE_CKPT}")
NOGATE_RUN=$(dirname "${NOGATE_CKPT}")
echo "GATE_RUN=${GATE_RUN} NOGATE_RUN=${NOGATE_RUN} $(date)"

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

echo "D3_DONE $(date)"
