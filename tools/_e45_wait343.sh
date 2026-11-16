#!/bin/bash
# Wait for the E45 round-91 S3 seed-robustness job (343) on the remote,
# then pull its result back and re-run the local audit with the
# equal-bias half merged in.
#
# Read-only with respect to the experiment: the job was already
# submitted; this only waits for it and downloads its output.
cd /r/1radarNWPprediction/radar-nwp-bench || exit 1

PY=/c/Users/nous/anaconda3/python
LOG=tools/_evals/e45/_r91_wait343.log
: > "$LOG"

for i in $(seq 1 90); do
  states=$(MSYS_NO_PATHCONV=1 python tools/remote.py -- \
    'cd /home/wf/radarNWP && sacct -j 343 --format=State --noheader -n 2>/dev/null | tr "\n" ","' \
    2>/dev/null | tr -d ' ')
  echo "[$(date +%H:%M:%S)] poll ${i}: ${states}" >> "$LOG"
  case "${states}" in
    *COMPLETED*)
      echo "=== COMPLETED at $(date -Is) ===" >> "$LOG"
      MSYS_NO_PATHCONV=1 python tools/remote.py --download \
        /home/wf/radarNWP/runs/e45/s3_seedrobust.json \
        --out tools/_evals/e45/s3_seedrobust.json >> "$LOG" 2>&1
      echo "=== download rc=$? ===" >> "$LOG"
      ls -l tools/_evals/e45/s3_seedrobust.json >> "$LOG" 2>&1
      "${PY}" tools/_e45_round91.py >> "$LOG" 2>&1
      echo "=== local audit rc=$? ===" >> "$LOG"
      "${PY}" tools/_e45_log_round91.py >> "$LOG" 2>&1
      echo "=== writeback rc=$? ===" >> "$LOG"
      exit 0
      ;;
    *FAILED*|*CANCELLED*|*TIMEOUT*)
      echo "=== TERMINAL BAD STATE: ${states} ===" >> "$LOG"
      MSYS_NO_PATHCONV=1 python tools/remote.py -- \
        'cd /home/wf/radarNWP && tail -30 run_logs/slurm/e45-s3seedrobust-343.out; echo ---ERR---; tail -20 run_logs/slurm/e45-s3seedrobust-343.err' \
        >> "$LOG" 2>&1
      exit 1
      ;;
  esac
  sleep 60
done
echo "=== TIMEOUT: still not finished after 90 polls ===" >> "$LOG"
exit 2
