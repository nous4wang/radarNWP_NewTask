#!/usr/bin/env bash
# Follow-up for the field batch: the first offline probe crashed on a JSON
# key typo (f"{2.0:g}" -> "2" instead of "2.0") after the A2 loop, so A4 and
# all A.2/A.4 paired bootstraps never produced anything. The ten float32
# field files are already on disk, so this needs NO re-inference.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
export CUDA_VISIBLE_DEVICES=0
PY=/home/wf/.conda/envs/radar_nwp/bin/python
echo "FIELD_FINISH_START $(date)"

FIELDS=()
for i in 0 1 2 3 4 5 6 7 8 9; do
  F="runs/fields/base_concat_s${i}.pred.npy"
  [ -f "${F}" ] && FIELDS+=("${F}")
done
echo "fields=${#FIELDS[@]}"
"$PY" tools/_field_probe.py \
  --fields "${FIELDS[@]}" \
  --truth runs/fields/base_concat_s0.truth.npy \
  --eval-json runs/fields/base_concat_s0.eval.json \
  --out-json runs/field_probe.json \
  --windows-prefix runs/field_probe \
  > runs/field_probe.log 2>&1
PROBE_EXIT=$?
echo "PROBE_EXIT=${PROBE_EXIT} $(date)"
tail -12 runs/field_probe.log
if [ "${PROBE_EXIT}" -ne 0 ] || [ ! -f runs/field_probe.json ]; then
  echo "FIELD_FINISH_ABORT: offline probe failed $(date)"
  exit 1
fi

# ---- paired bootstrap against the pre-registered k=1 persistence floor
for arm in ensemble_mean ensemble_vote maxpool ampctrl; do
  A="runs/field_probe_${arm}.windows.json"
  [ -f "${A}" ] || continue
  "$PY" tools/_bootstrap_qpe_vs_zr.py \
    --a "${A}" --b runs/pmm_k1.windows.json \
    --name-a "${arm}" --name-b floor_k1 --n-boot 10000 \
    > "runs/bootstrap_${arm}_vs_k1.log" 2>&1
  echo "bootstrap ${arm} vs floor_k1 exit=$?"
done

# ---- A4-A: ensemble mean / vote vs the LOWER-MEDIAN single seed as well
MED=$("$PY" -c '
import json
d = json.load(open("runs/field_probe.json"))
v = d["A4"]["per_seed_csi2"]
print("%02d" % sorted(range(len(v)), key=lambda i: v[i])[(len(v)-1)//2])
')
echo "lower_median_seed_index=${MED}"
for arm in ensemble_mean ensemble_vote; do
  A="runs/field_probe_${arm}.windows.json"
  B="runs/field_probe_seed${MED}.windows.json"
  [ -f "${A}" ] && [ -f "${B}" ] || continue
  "$PY" tools/_bootstrap_qpe_vs_zr.py \
    --a "${A}" --b "${B}" \
    --name-a "${arm}" --name-b "median_seed_s${MED}" --n-boot 10000 \
    > "runs/bootstrap_${arm}_vs_median_seed.log" 2>&1
  echo "bootstrap ${arm} vs median_seed exit=$?"
done
echo "FIELD_FINISH_DONE $(date)"
