#!/usr/bin/env bash
# Launch the epoch-curve experiment exactly once.
set -uo pipefail

LOCK=/home/wf/radarNWP/runs/.epoch_curve.lock
LOG=/home/wf/radarNWP/runs/epoch_curve.log

if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "another instance holds ${LOCK}; not starting"
  exit 0
fi

echo "started at $(date '+%Y-%m-%d %H:%M:%S') pid=$$"
bash /home/wf/radarNWP/scripts/epoch_curve.sh
echo "finished at $(date '+%Y-%m-%d %H:%M:%S')"
