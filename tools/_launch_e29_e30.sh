#!/usr/bin/env bash
# E29 (score-aware spectral matrix rescan) + E30 (gapped-window counterfactual
# fill).  Zero TRAINING compute: existing checkpoints are re-inferred.
#
# Everything is pinned to CUDA device 0 (§11.6 S1).  The evaluation /
# spectrum work is CPU-bound (Hann stitching + 301x301 FFT), so the remote's
# eight cards are not used; waves of four keep the 144 cores busy without
# introducing cross-device arithmetic.
#
# Detached on the remote.  Every stage writes a line to
# runs/e29_e30_stage.log; a failed anchor aborts the whole batch.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
export CUDA_VISIBLE_DEVICES=0
PY=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
CFG=configs/models/simvp_v2_nwp_concat.yaml
MODEL=simvp_v2_nwp_concat
CK0=runs/route_p2/seeds/base_concat_s0/simvp_v2_nwp_concat_1793645968/epoch_00.pt
STAGE=runs/e29_e30_stage.log
PAR=4
mkdir -p runs/e29_rapsd runs/e29_rapsd_epoch runs/e29_anchor runs/e29_logs
: > "$STAGE"
log() { echo "$1 $(date '+%F %T')" | tee -a "$STAGE"; }

# ---------------------------------------------------------------- E30 -----
log "STAGE E30-EVAL start (three arms, no window dropped)"
PIDS=()
for arm in linear prev zero; do
  "$PY" scripts/evaluate.py \
    --model "$MODEL" --config "$CFG" --checkpoint "$CK0" \
    --data-root "$DATA" --packed --split robustness_missing \
    --patch-size 64 --slide-stride 64 --in-len 11 --out-len 10 \
    --thresholds 0.1 0.5 2.0 4.0 --missing-frame-fill "$arm" \
    --out-json "runs/e30_${arm}.eval.json" \
    --per-window-out "runs/e30_${arm}.windows.json" \
    > "runs/e29_logs/e30_${arm}.log" 2>&1 &
  PIDS+=("$!")
done
for p in "${PIDS[@]}"; do wait "$p"; echo "E30 eval pid $p exit=$?" >> "$STAGE"; done
log "STAGE E30-EVAL done"

log "STAGE E30-ANCHOR start"
"$PY" tools/_e30_gap_subset.py \
  --dir runs --prefix e30 \
  --t7-on runs/t7_on.windows.json --t7-off runs/t7_off.windows.json \
  --t7-on-eval runs/t7_on.eval.json \
  > runs/e29_logs/e30_anchor.log 2>&1
ANCHOR_EXIT=$?
echo "E30_ANCHOR_EXIT=$ANCHOR_EXIT" >> "$STAGE"
if [ "$ANCHOR_EXIT" -ne 0 ]; then
  log "ABORT: E30 anchors failed; no bootstrap or E29 compute will be spent"
  exit 1
fi
log "STAGE E30-ANCHOR PASS"

log "STAGE E30-BOOTSTRAP start"
boot() {
  local a="$1" b="$2" na="$3" nb="$4" out="$5"
  "$PY" tools/_bootstrap_qpe_vs_zr.py \
    --a "$a" --b "$b" --name-a "$na" --name-b "$nb" --n-boot 10000 \
    > "runs/$out" 2>&1
  echo "bootstrap $out exit=$?" >> "$STAGE"
}
boot runs/e30_linear_gap149.windows.json runs/e30_zero_gap149.windows.json \
     e30_linear e30_zero bootstrap_e30_linear_vs_zero_gap149.log &
boot runs/e30_linear_gap149.windows.json runs/e30_prev_gap149.windows.json \
     e30_linear e30_prev bootstrap_e30_linear_vs_prev_gap149.log &
boot runs/e30_linear_clean813.windows.json runs/e30_zero_clean813.windows.json \
     e30_linear e30_zero bootstrap_e30_linear_vs_zero_clean813.log &
wait
log "STAGE E30-BOOTSTRAP done"

# ---------------------------------------------------------------- E29 -----
log "STAGE E29-SELFCHECK start"
"$PY" scripts/rapsd.py --self-check > runs/e29_logs/selfcheck.log 2>&1
SELF_EXIT=$?
echo "RAPSD_SELF_CHECK_EXIT=$SELF_EXIT" >> "$STAGE"
if [ "$SELF_EXIT" -ne 0 ]; then
  log "ABORT: rapsd self-check failed"
  exit 1
fi
log "STAGE E29-SELFCHECK PASS"

run_rapsd() {
  local name="$1" ck="$2" out="$3" tag="$4"
  mkdir -p "$out"
  "$PY" scripts/rapsd.py \
    --data-root "$DATA" --packed --split test \
    --patch-size 64 --slide-stride 64 --taper none \
    --model "$name" --checkpoint "$ck" --out-dir "$out" \
    > "runs/e29_logs/${tag}.log" 2>&1
  echo "RAPSD ${tag} exit=$? $(date '+%F %T')" >> "$STAGE"
}

log "STAGE E29-ANCHOR start (two standard-loss re-runs, same parameters)"
run_rapsd simvp_v2 \
  runs/tgrs/simvp_v2/simvp_v2_1793473161/last.pt \
  runs/e29_anchor/simvp_v2 anchor_std_simvp_v2 &
run_rapsd pysteps_adv \
  runs/tgrs/pysteps_adv/pysteps_adv_analytical/last.pt \
  runs/e29_anchor/pysteps_adv anchor_std_pysteps_adv &
wait
log "STAGE E29-ANCHOR done"

MODELS=(
  convlstm predrnn earthformer simvp_v2 metnet3 diffcast cascast_nwp mocast
  pw_foucast fusioncast mmf_rnn furecast simvp_radar_only simvp_early_concat
  simvp_v2_nwp_concat simvp_v2_nwp_xattn simvp_v2_nwp_delta simvp_v2_quantum
)
CKPTS=()
for m in "${MODELS[@]}"; do
  p=$(find "runs/score_matrix/$m" -name last.pt | head -1)
  if [ -z "$p" ]; then
    log "WARNING: no last.pt for $m"
    p=""
  fi
  CKPTS+=("$p")
done

log "STAGE E29-MATRIX start (18 score-aware models, waves of ${PAR})"
for start in 0 4 8 12 16; do
  PIDS=()
  for i in $(seq "$start" $((start + PAR - 1))); do
    [ "$i" -lt "${#MODELS[@]}" ] || continue
    [ -n "${CKPTS[$i]}" ] || continue
    run_rapsd "${MODELS[$i]}" "${CKPTS[$i]}" \
      "runs/e29_rapsd/${MODELS[$i]}" "score_${MODELS[$i]}" &
    PIDS+=("$!")
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  log "STAGE E29-MATRIX wave ${start} done"
done

log "STAGE E29-EPOCH start (ep00 vs ep19, two arms)"
EPOCH_SPECS=(
  "base_concat|simvp_v2_nwp_concat|runs/route_p2/curve/base_concat/simvp_v2_nwp_concat_1793632932/epoch_00.pt|epoch_00"
  "base_concat|simvp_v2_nwp_concat|runs/route_p2/curve/base_concat/simvp_v2_nwp_concat_1793632932/epoch_19.pt|epoch_19"
  "qpe_0|simvp_v2_qpe_guided|runs/route_p2/curve/qpe_0/simvp_v2_qpe_guided_1793632932/epoch_00.pt|epoch_00"
  "qpe_0|simvp_v2_qpe_guided|runs/route_p2/curve/qpe_0/simvp_v2_qpe_guided_1793632932/epoch_19.pt|epoch_19"
)
PIDS=()
for spec in "${EPOCH_SPECS[@]}"; do
  IFS='|' read -r arm model ck ep <<< "$spec"
  run_rapsd "$model" "$ck" \
    "runs/e29_rapsd_epoch/${arm}_${ep}" "epoch_${arm}_${ep}" &
  PIDS+=("$!")
done
for p in "${PIDS[@]}"; do wait "$p"; done
log "STAGE E29-EPOCH done"

log "STAGE E29-SUMMARY start"
"$PY" tools/_e29_spectral_summary.py \
  --score-dir runs/e29_rapsd --epoch-dir runs/e29_rapsd_epoch \
  --anchor-dir runs/e29_anchor --std-json runs/e29_std_high_k.json \
  --eval-dir runs/score_matrix --out runs/e29_spectral_summary.json \
  > runs/e29_logs/spectral_summary.log 2>&1
echo "E29_SUMMARY_EXIT=$?" >> "$STAGE"
log "E29_E30_DONE"
