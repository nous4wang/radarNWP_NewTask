#!/usr/bin/env bash
# A.2 / A.4 / queue-7 batch.  Zero TRAINING compute: existing ep00 weights
# are re-inferred once, the stitched whole fields are dumped, and all
# downstream analysis is offline.  Detached on the remote.
#
# The remote has 8x RTX 4090 / 144 cores but this evaluation is CPU-bound
# (Hann stitching + FSS integral images); a serial launch leaves the machine
# at load ~1.4.  Seeds are therefore evaluated in waves of four.  All waves
# are pinned to CUDA device 0 so the numbers stay comparable with the
# earlier single-GPU results (§11.6 S1, and §24's "evaluation is fully
# deterministic"): no cross-GPU arithmetic is introduced.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
export CUDA_VISIBLE_DEVICES=0
PY=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
CFG=configs/models/simvp_v2_nwp_concat.yaml
MODEL=simvp_v2_nwp_concat
PAR=4
mkdir -p runs/fields
# runs/fields is dedicated to this batch.  An earlier launch dumped float16,
# which rounds 0.1 mm to 0.0999756 and breaks the per-window invariant; wipe
# only this batch's own artifacts so a re-launch cannot mix dtypes.
rm -f runs/fields/*.pred.npy runs/fields/*.truth.npy \
      runs/fields/*.eval.json runs/fields/*.windows.json \
      runs/fields/*.log runs/fields/parallel_status.log
rm -f runs/t7_on.eval.json runs/t7_off.eval.json \
      runs/t7_on.windows.json runs/t7_off.windows.json \
      runs/t7_on_paired.windows.json runs/t7_alignment.log \
      runs/t7_on.log runs/t7_off.log runs/bootstrap_t7_off_vs_on.log
rm -f runs/field_dtype_check.log runs/field_probe.json runs/field_probe.log \
      runs/field_probe_*.windows.json runs/bootstrap_ensemble* \
      runs/bootstrap_maxpool* runs/bootstrap_ampctrl*
echo "FIELD_BATCH_START $(date) parallel=${PAR} cuda=${CUDA_VISIBLE_DEVICES}"

CKPTS=()
for s in 0 1 2; do
  CK=$(find "runs/route_p2/seeds/base_concat_s${s}" -name epoch_00.pt | head -1)
  CKPTS+=("${CK}")
done
for s in 3 4 5 6 7 8 9; do
  CK=$(find "runs/route_seeds/train/base_concat_s${s}" -name epoch_00.pt | head -1)
  CKPTS+=("${CK}")
done
echo "checkpoints: ${#CKPTS[@]}"

# Multiscale FSS is computed offline from the dumped fields by
# tools/_field_probe.py, so --fss-radii is deliberately NOT requested here:
# three extra radii x four thresholds on every 301x301 window roughly
# triples evaluation time for no additional information.
run_seed() {
  local i="$1" CK="$2"
  if [ -z "${CK}" ] || [ ! -f "${CK}" ]; then
    echo "MISSING_CKPT seed=${i} path=${CK}" >> runs/fields/parallel_status.log
    return 1
  fi
  echo "--- seed ${i}: ${CK} $(date) ---" >> runs/fields/parallel_status.log
  "$PY" scripts/evaluate.py \
    --model "${MODEL}" --config "${CFG}" --checkpoint "${CK}" \
    --data-root "${DATA}" --packed --split test \
    --patch-size 64 --slide-stride 64 --in-len 11 --out-len 10 \
    --thresholds 0.1 0.5 2.0 4.0 \
    --out-json "runs/fields/base_concat_s${i}.eval.json" \
    --per-window-out "runs/fields/base_concat_s${i}.windows.json" \
    --dump-fields "runs/fields/base_concat_s${i}" \
    > "runs/fields/base_concat_s${i}.log" 2>&1
  local rc=$?
  echo "seed ${i} exit=${rc} $(date)" >> runs/fields/parallel_status.log
  return "${rc}"
}

for start in 0 4 8; do
  PIDS=()
  for i in $(seq "${start}" $((start + PAR - 1))); do
    [ "${i}" -lt "${#CKPTS[@]}" ] || continue
    run_seed "${i}" "${CKPTS[$i]}" &
    PIDS+=("$!")
  done
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for p in "${PIDS[@]}"; do wait "${p}"; done
  fi
  echo "wave ${start} done $(date)"
done
cat runs/fields/parallel_status.log
echo "FIELDS_DONE $(date)"

# ---- queue 7 (E28): same ep00 ckpt, only the interpolation switch changes
CK0="${CKPTS[0]}"
run_t7() {
  local mode="$1" FLAG=""
  if [ "${mode}" = "off" ]; then FLAG="--no-interpolate"; fi
  "$PY" scripts/evaluate.py \
    --model "${MODEL}" --config "${CFG}" --checkpoint "${CK0}" \
    --data-root "${DATA}" --packed --split robustness_missing \
    --patch-size 64 --slide-stride 64 --in-len 11 --out-len 10 \
    --thresholds 0.1 0.5 2.0 4.0 ${FLAG} \
    --out-json "runs/t7_${mode}.eval.json" \
    --per-window-out "runs/t7_${mode}.windows.json" \
    > "runs/t7_${mode}.log" 2>&1
  echo "T7 ${mode} exit=$? $(date)"
}
T7_PIDS=()
for mode in on off; do
  run_t7 "${mode}" &
  T7_PIDS+=("$!")
done
if [ "${#T7_PIDS[@]}" -gt 0 ]; then
  for p in "${T7_PIDS[@]}"; do wait "${p}"; done
fi
echo "T7_DONE $(date)"

# ---- offline analysis (A.2 + A.4)
FIELDS=()
for i in 0 1 2 3 4 5 6 7 8 9; do
  F="runs/fields/base_concat_s${i}.pred.npy"
  [ -f "${F}" ] && FIELDS+=("${F}")
done
echo "offline probe with ${#FIELDS[@]} fields $(date)"
# Gate 2 pre-flight: the dumped fields must reproduce the float32 window
# tables before any downstream score is trusted.
"$PY" tools/_check_field_dtype.py > runs/field_dtype_check.log 2>&1
DTYPE_EXIT=$?
cat runs/field_dtype_check.log
if [ "${DTYPE_EXIT}" -ne 0 ]; then
  echo "FIELD_BATCH_ABORT: dumped fields are not threshold-safe $(date)"
  exit 1
fi
echo "FIELD_BATCH_DTYPE_OK $(date)"
"$PY" tools/_field_probe.py \
  --fields "${FIELDS[@]}" \
  --truth runs/fields/base_concat_s0.truth.npy \
  --eval-json runs/fields/base_concat_s0.eval.json \
  --out-json runs/field_probe.json \
  --windows-prefix runs/field_probe \
  > runs/field_probe.log 2>&1
echo "PROBE_EXIT=$? $(date)"

# ---- paired bootstrap against the pre-registered k=1 persistence floor
for arm in ensemble_mean ensemble_vote maxpool ampctrl; do
  A="runs/field_probe_${arm}.windows.json"
  [ -f "${A}" ] || continue
  "$PY" tools/_bootstrap_qpe_vs_zr.py \
    --a "${A}" --b runs/pmm_k1.windows.json \
    --name-a "${arm}" --name-b floor_k1 --n-boot 10000 \
    > "runs/bootstrap_${arm}_vs_k1.log" 2>&1
  echo "bootstrap ${arm} exit=$?"
done
# ---- T7 (E28): --no-interpolate DROPS windows (962 -> 813 in §4.16.4), so the
# raw tables are not a paired subset.  Restrict the interpolation-on table to
# the windows the off arm retained, using the case:issue_time keys emitted by
# evaluate.py, then bootstrap on that common set.
if [ -f runs/t7_on.windows.json ] && [ -f runs/t7_off.windows.json ]; then
  "$PY" tools/_align_paired_windows.py \
    --on runs/t7_on.windows.json --off runs/t7_off.windows.json \
    --out runs/t7_on_paired.windows.json \
    > runs/t7_alignment.log 2>&1
  echo "T7 alignment exit=$? $(date)"
  cat runs/t7_alignment.log
  "$PY" tools/_bootstrap_qpe_vs_zr.py \
    --a runs/t7_off.windows.json --b runs/t7_on_paired.windows.json \
    --name-a no_interpolate --name-b interpolate_paired --n-boot 10000 \
    > runs/bootstrap_t7_off_vs_on.log 2>&1
  echo "bootstrap T7 exit=$?"
fi
echo "FIELD_BATCH_DONE $(date)"
