#!/usr/bin/env bash
# Launch the epoch check exactly once.
#
# The paramiko + setsid launch path can execute the same script twice, and two
# copies writing the same report file would corrupt it. A directory lock makes
# the second copy exit before it does any work.
set -uo pipefail

LOCK=/home/wf/radarNWP/runs/.epoch_check.lock
LOG=/home/wf/radarNWP/runs/epoch_check.log

if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "another instance holds ${LOCK}; not starting"
  exit 0
fi

echo "started at $(date '+%Y-%m-%d %H:%M:%S') pid=$$"
bash /home/wf/radarNWP/scripts/epoch_check.sh
echo "finished at $(date '+%Y-%m-%d %H:%M:%S')"
