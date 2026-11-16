#!/usr/bin/env bash
# Wait for the D2 PMM products, then run all six paired bootstrap tests
# (3 PMM variants x 2 references k1 / k2) on the remote, where the ~318 kB
# per-window tables do not have to cross the SSH channel.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
BOOT=tools/_bootstrap_qpe_vs_zr.py

echo "PMM_WATCH_START $(date)"
for I in $(seq 1 100); do
  if [ -f runs/pmm_persistence.json ] && [ -f runs/pmm_pmm.windows.json ]; then
    break
  fi
  sleep 30
done
if [ ! -f runs/pmm_persistence.json ]; then
  echo "PMM_WATCH_TIMEOUT $(date)"
  exit 1
fi
echo "PMM_RESULT_FOUND $(date)"
cat runs/pmm_persistence.json
echo

for ARM in pmm pmm_cond pmm_rain; do
  for REF in k1 k2; do
    LOG="runs/bootstrap_pmm_${ARM}_vs_${REF}.log"
    "$PY" "${BOOT}" \
      --a "runs/pmm_${ARM}.windows.json" \
      --b "runs/pmm_${REF}.windows.json" \
      --name-a "${ARM}" --name-b "${REF}" --n-boot 10000 \
      > "${LOG}" 2>&1
    echo "BOOT_EXIT ${ARM} vs ${REF} = $? $(date)"
    tail -22 "${LOG}"
  done
done
echo "PMM_WATCH_DONE $(date)"
