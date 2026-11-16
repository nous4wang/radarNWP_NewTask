#!/usr/bin/env bash
# Background waiter: poll the remote D3_V2_DONE marker, then dump the whole log.
# Judged by FILE CONTENT / log marker, never by the launcher exit code (MASTER
# remote-ops notes: SSH banner errors make remote.py exit codes unreliable).
cd /r/1radarNWPprediction/radar-nwp-bench || exit 1
PY=C:/Users/nous/anaconda3/python.exe
for i in $(seq 1 100); do
  R=$(MSYS_NO_PATHCONV=1 "$PY" tools/remote.py 'cd /home/wf/radarNWP && if grep -q D3_V2_DONE runs/d3_v2_launch.log; then echo READY; else echo WAIT; fi' 2>/dev/null)
  case "$R" in
    *READY*)
      echo "=== D3_V2 READY (poll $i) ==="
      MSYS_NO_PATHCONV=1 "$PY" tools/remote.py 'cd /home/wf/radarNWP; tail -140 runs/d3_v2_launch.log'
      exit 0;;
  esac
  sleep 30
done
echo "=== D3_V2 WAITER TIMEOUT (100 polls x 30s) ==="
exit 1
