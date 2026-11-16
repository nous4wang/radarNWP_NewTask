#!/usr/bin/env bash
# Wait for the D3 per-epoch per-window tables, then run the D3 paired tests:
#   D3-A  gate ep00  vs  v1 ep00 (section 4.19 reference)
#   D3-B  gate ep00  vs  nogate ep00 (one-variable gate ablation)
#   CHECK nogate ep00 vs  v1 ep00 (must be ~0; the zero experiment extension)
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
BOOT=tools/_bootstrap_qpe_vs_zr.py
G=runs/physical_qpe_v2/gate/eval/epoch_00.windows.json
N=runs/physical_qpe_v2/nogate/eval/epoch_00.windows.json
V=runs/physical_qpe/eval/epoch_00.windows.json

echo "D3_WATCH_START $(date)"
for I in $(seq 1 160); do
  if [ -f "${G}" ] && [ -f "${N}" ]; then break; fi
  sleep 30
done
if [ ! -f "${G}" ] || [ ! -f "${N}" ]; then
  echo "D3_WATCH_TIMEOUT G=$([ -f "${G}" ] && echo yes || echo no) N=$([ -f "${N}" ] && echo yes || echo no) $(date)"
  exit 1
fi
echo "D3_TABLES_FOUND $(date)"

run_boot() {
  local NAME="$1" A="$2" B="$3" NA="$4" NB="$5"
  local LOG="runs/bootstrap_d3_${NAME}.log"
  "$PY" "${BOOT}" --a "${A}" --b "${B}" \
    --name-a "${NA}" --name-b "${NB}" --n-boot 10000 \
    > "${LOG}" 2>&1
  echo "BOOT_EXIT ${NAME} = $? $(date)"
  tail -22 "${LOG}"
}

run_boot d3a_gate_vs_v1      "${G}" "${V}" gate_ep00 v1_ep00
run_boot d3b_gate_vs_nogate  "${G}" "${N}" gate_ep00 nogate_ep00
run_boot check_nogate_vs_v1  "${N}" "${V}" nogate_ep00 v1_ep00

echo "=== D3 PER-EPOCH CSI@2.0 SUMMARY ==="
for ARM in gate nogate; do
  for EP in 00 01 02; do
    F="runs/physical_qpe_v2/${ARM}/eval/epoch_${EP}.eval.json"
    if [ -f "${F}" ]; then
      "$PY" -c "
import json,sys
d=json.load(open('${F}'))
print('${ARM} ep${EP}  CSI@2.0 = %.4f  BIAS@2.0 = %.4f' % (
    d['thresholds']['2.0']['csi'], d['thresholds']['2.0']['bias']))
" 2>/dev/null || echo "${ARM} ep${EP} parse failed"
    else
      echo "${ARM} ep${EP} MISSING"
    fi
  done
done
echo "D3_WATCH_DONE $(date)"
