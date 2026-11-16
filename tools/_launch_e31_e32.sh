#!/usr/bin/env bash
# E31 (standard-loss spectra on the same window set) + E32 (single-model
# frequency-domain intervention).  Zero TRAINING compute: E31 re-infers
# seventeen existing checkpoints, E32 is pure offline analysis of the
# already-dumped whole fields from the E25 batch.
#
# Detached on the remote.  Every stage writes a line to
# runs/e31_e32_stage.log; a failed anchor aborts before the rest runs.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
export CUDA_VISIBLE_DEVICES=0
PY=/home/wf/.conda/envs/radar_nwp/bin/python
STAGE=runs/e31_e32_stage.log
mkdir -p runs/e31_std_rapsd runs/e32_freq_intervention
: > "$STAGE"
log() { echo "$1 $(date '+%F %T')" | tee -a "$STAGE"; }

# ---- E32 anchor first: the control arm must reproduce E25 bit for bit ----
log "STAGE E32-ANCHOR start (one seed, control arm only)"
"$PY" tools/_e32_freq_intervention.py \
  --fields-dir runs/fields --prefix base_concat_s \
  --seeds 1 --arms orig --out-dir runs/e32_anchor \
  > runs/e32_anchor.log 2>&1
echo "E32_ANCHOR_EXIT=$?" >> "$STAGE"
if [ ! -f runs/e32_anchor/summary.json ]; then
  log "ABORT: E32 anchor arm produced no summary"
  exit 1
fi
ANCHOR_CSI=$("$PY" -c '
import json
d = json.load(open("runs/e32_anchor/summary.json"))
s = d["per_seed"]["0"]["orig"]
print("%.9f" % s["high_k_energy_ratio"])
')
log "E32-ANCHOR orig high-k energy ratio = ${ANCHOR_CSI}"

log "STAGE E31 start (seventeen standard-loss spectra, 538 windows)"
"$PY" tools/_e31_std_spectral.py \
  --root /home/wf/radarNWP --out-dir runs/e31_std_rapsd \
  --score-dir runs/e29_rapsd --anchor-dir runs/e29_anchor \
  --eval-dir runs/score_matrix --out runs/e31_spectral_paired.json \
  > runs/e31_std_spectral.log 2>&1
echo "E31_EXIT=$?" >> "$STAGE"
log "STAGE E31 done"

log "STAGE E32 start (ten seeds, four arms)"
"$PY" tools/_e32_freq_intervention.py \
  --fields-dir runs/fields --prefix base_concat_s \
  --seeds 10 --arms orig lowpass truthswap phasescramble \
  --out-dir runs/e32_freq_intervention \
  > runs/e32_freq_intervention.log 2>&1
echo "E32_EXIT=$?" >> "$STAGE"
log "STAGE E32 done"

log "E31_E32_DONE"
