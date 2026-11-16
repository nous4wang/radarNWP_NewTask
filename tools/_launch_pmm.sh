#!/usr/bin/env bash
# One-shot detached launcher for the D2 PMM rescoring (local temp, not committed).
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
echo "PMM_START $(date)"
"$PY" tools/_calibrate_pmm.py \
  --data-root /home/wf/data/radarNWP/BJ_mmap \
  --out-json runs/pmm_persistence.json \
  --per-window-prefix runs/pmm
rc=$?
echo "PMM_EXIT=${rc} $(date)"
exit ${rc}
