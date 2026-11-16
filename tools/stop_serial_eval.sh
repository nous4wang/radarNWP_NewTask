#!/usr/bin/env bash
# Stop the serial evaluation that run_extra_baselines.sh runs, so the
# parallel evaluator can take over. The checkpoints and the reports that
# already exist are left alone.
set -u

echo "=== before ==="
ps -eo pid,args | grep '[e]valuate.py' | grep -oE '\-\-model [a-z_0-9]+' | sort -u

# The evaluation process currently running.
pids=$(ps -eo pid,args | grep '[e]valuate.py' | awk '{print $1}')
if [ -n "${pids}" ]; then
    echo "stopping the running evaluation: ${pids}"
    kill -TERM ${pids} 2>/dev/null || true
fi

# The driver script itself -- its training phase is long finished, and its
# serial evaluation is what we are replacing.
pids=$(ps -eo pid,args | grep '[r]un_extra_baselines.sh' | awk '{print $1}')
if [ -n "${pids}" ]; then
    echo "stopping the extra-baselines driver: ${pids}"
    kill -TERM ${pids} 2>/dev/null || true
fi

sleep 5
pids=$(ps -eo pid,args | grep '[e]valuate.py' | awk '{print $1}')
if [ -n "${pids}" ]; then
    echo "force stopping: ${pids}"
    kill -KILL ${pids} 2>/dev/null || true
fi
sleep 2

echo "=== after ==="
ps -eo pid,args | grep -E '[e]valuate.py|[r]un_extra' | head -5
echo "(empty above = stopped)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
