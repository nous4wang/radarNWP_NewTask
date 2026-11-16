#!/usr/bin/env bash
# Stop only the training runs started by run_extra_baselines.sh, leaving the
# main sweep's own runs (mocast / cascast_nwp / furecast) untouched.
set -u

EXTRA="convlstm earthformer predrnn simvp_early_concat simvp_radar_only simvp_v2"

echo "=== before ==="
ps -eo pid,args | grep '[t]rain.py' | grep -oE '\-\-model [a-z_0-9]+' | sort | uniq -c

for model in ${EXTRA}; do
    pids=$(ps -eo pid,args | grep '[t]rain.py' | grep -- "--model ${model} " | awk '{print $1}')
    if [ -n "${pids}" ]; then
        echo "killing ${model}: ${pids}"
        kill -TERM ${pids} 2>/dev/null || true
    fi
done

# the driver script itself
pids=$(ps -eo pid,args | grep '[r]un_extra_baselines.sh' | awk '{print $1}')
if [ -n "${pids}" ]; then
    echo "killing driver: ${pids}"
    kill -TERM ${pids} 2>/dev/null || true
fi

sleep 8

# anything that survived the polite request
for model in ${EXTRA}; do
    pids=$(ps -eo pid,args | grep '[t]rain.py' | grep -- "--model ${model} " | awk '{print $1}')
    if [ -n "${pids}" ]; then
        echo "force killing ${model}: ${pids}"
        kill -KILL ${pids} 2>/dev/null || true
    fi
done
sleep 3

echo "=== after ==="
ps -eo pid,args | grep '[t]rain.py' | grep -oE '\-\-model [a-z_0-9]+' | sort | uniq -c
echo "=== gpus ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
