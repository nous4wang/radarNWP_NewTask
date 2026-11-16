#!/usr/bin/env bash
# E24 / MASTER 4.26 -- detached launcher for the per-case Z-R bias probe.
# Zero compute: analytical field only, no model, no training.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
echo "E24_START $(date)"
"$PY" tools/_case_zr_bias.py \
  --data-root /home/wf/data/radarNWP/BJ_mmap \
  --patch-size 301 --slide-stride 301 \
  --batch-size 4 --num-workers 4 \
  --out-json runs/case_zr_bias.json
rc=$?
echo "E24_EXIT=${rc} $(date)"
echo "E24_DONE"
exit "${rc}"
