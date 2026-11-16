#!/usr/bin/env bash
# Launch the per-epoch evaluation exactly once.
set -uo pipefail

LOCK=/home/wf/radarNWP/runs/.epoch_curve_eval.lock
LOG=/home/wf/radarNWP/runs/epoch_curve_eval.log

if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "another instance holds ${LOCK}; not starting"
  exit 0
fi

echo "started at $(date '+%Y-%m-%d %H:%M:%S') pid=$$"
bash /home/wf/radarNWP/scripts/epoch_curve_eval.sh
echo "finished at $(date '+%Y-%m-%d %H:%M:%S')"
