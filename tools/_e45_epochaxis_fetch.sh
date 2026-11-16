#!/bin/bash
# Wait for the three E45 epoch-axis jobs, then pull the products back.
#
# Why one final tarball instead of the per-job ones: all three jobs write
# into the SAME output root, so a per-job tarball can capture another
# job's half-written file.  After every job is done, one tar of the whole
# root is atomic with respect to the writers.
#
# It is also ONE download, which is what the remote tool wants: several
# --download groups in one invocation, only the first one takes effect.

set -eo pipefail
cd /r/1radarNWPprediction/radar-nwp-bench
export MSYS_NO_PATHCONV=1

OUT_DIR="tools/_evals/e45/epoch_axis"
mkdir -p "${OUT_DIR}"

echo "[$(date +%H:%M:%S)] waiting for jobs 315 316 317 ..."
DONE=0
for i in $(seq 1 600); do
    STATE=$(python tools/remote.py -- 'sacct -j 315,316,317 --format=JobID,State --noheader 2>/dev/null | tr -d " " | grep -E "^(315|316|317)" ' 2>&1 || true)
    NCOMP=$(printf '%s\n' "${STATE}" | grep -c "COMPLETED" || true)
    NFAIL=$(printf '%s\n' "${STATE}" | grep -c "FAILED" || true)
    echo "[$(date +%H:%M:%S)] completed=${NCOMP} failed=${NFAIL}"
    if [ "${NFAIL}" -gt 0 ]; then
        echo "=== a job FAILED -- dumping the tail of each log ==="
        python tools/remote.py -- 'cd /home/wf/radarNWP && for J in 315 316 317; do echo "--- $J ---"; tail -25 run_logs/slurm/e45-epochaxis-$J.out 2>&1 | cut -c1-200; tail -10 run_logs/slurm/e45-epochaxis-$J.err 2>&1 | cut -c1-200; done'
        exit 2
    fi
    if [ "${NCOMP}" -ge 3 ]; then DONE=1; break; fi
    sleep 60
done

if [ "${DONE}" -ne 1 ]; then
    echo "=== timed out waiting for the epoch-axis jobs ==="
    exit 3
fi

echo "[$(date +%H:%M:%S)] all three jobs COMPLETED -- packing the products ==="
python tools/remote.py -- 'cd /home/wf/radarNWP && rm -f runs/e45_epochaxis/epoch_axis_all.tar.gz && tar -czf runs/e45_epochaxis/epoch_axis_all.tar.gz -C runs/e45_epochaxis evals windows probe 2>/dev/null; md5sum runs/e45_epochaxis/epoch_axis_all.tar.gz; ls runs/e45_epochaxis/evals | wc -l; ls runs/e45_epochaxis/probe | wc -l'

echo "[$(date +%H:%M:%S)] downloading ..."
python tools/remote.py --download /home/wf/radarNWP/runs/e45_epochaxis/epoch_axis_all.tar.gz --out "${OUT_DIR}/epoch_axis_all.tar.gz"

echo "[$(date +%H:%M:%S)] unpacking ..."
cd "${OUT_DIR}"
tar -xzf epoch_axis_all.tar.gz
echo "[$(date +%H:%M:%S)] done: $(ls evals | wc -l) eval files, $(ls probe 2>/dev/null | wc -l) probe files"
