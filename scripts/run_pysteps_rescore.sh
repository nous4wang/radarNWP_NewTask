#!/usr/bin/env bash
# Re-score the analytical baseline after the unit fix.
#
# Why this exists: the pysteps_adv row in the first matrix was produced by
# a model that applied the Z-R inversion to the *compressed* radar scale.
# A 45 dBZ storm came back as 0.004 mm per 6 minutes, two orders of magnitude
# below the weakest scoring threshold, so every pixel was a miss and the
# baseline scored exactly 0.000 with a NaN false-alarm rate. That is a bug
# artefact, not a result.
#
# The scores are written to staging files and moved into place only after
# they pass a sanity check, and only if they do not collide with a report
# that is already there. The gap script promotes its three models into
# runs/tgrs/ and evaluates them separately; this only touches pysteps_adv.
#
# It runs on a card the gap script is not using, and it waits for a free
# card rather than doubling up on one.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/tgrs}"
MODEL="pysteps_adv"

BATCH_SIZE=16
WORKERS=6
PATCH_SIZE=64
SLIDE_STRIDE=64
GPU="${GPU:-5}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

gpu_is_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "$1" 2>/dev/null || echo 0)"
    [ "${used:-0}" -lt 100 ]
}

log "=== waiting for GPU ${GPU} ==="
waited=0
while ! gpu_is_free "${GPU}"; do
    [ "${waited}" -eq 0 ] && log "  GPU ${GPU} is busy, waiting ..."
    sleep 60
    waited=$(( waited + 1 ))
    if [ "${waited}" -ge 120 ]; then
        log "  GPU ${GPU} still busy after 2 h -- skipped"
        exit 1
    fi
done
log "  GPU ${GPU} is free"

ckpt="$(ls -dt "${OUT_DIR}/${MODEL}"/*/last.pt 2>/dev/null | head -1)"
if [ -z "${ckpt}" ]; then
    log "no checkpoint for ${MODEL} -- skipped"
    exit 1
fi
log "  checkpoint: ${ckpt}"

stage_one() {
    local split="$1" suffix="$2"
    local tmp="${OUT_DIR}/.${MODEL}${suffix}.json.tmp"
    local final="${OUT_DIR}/${MODEL}${suffix}.json"
    local archived="${final}.buggy"

    # The report that is already there was produced by the broken model. It
    # is archived rather than kept, because it is not a score: every CSI is
    # zero and every false-alarm rate is NaN. If the new report fails its
    # sanity check the archive is put back, so nothing is ever lost.
    if [ -e "${final}" ]; then
        if [ -e "${archived}" ]; then
            log "  ${final} already exists and is archived -- skipped"
            return 0
        fi
        mv -f "${final}" "${archived}"
        log "  archived the pre-fix report to ${archived}"
    fi

    log "  ${split}: evaluating"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
        --model "${MODEL}" \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split "${split}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --patch-size "${PATCH_SIZE}" \
        --slide-stride "${SLIDE_STRIDE}" \
        --device cuda \
        --out-json "${tmp}" \
        > "${OUT_DIR}/${MODEL}${suffix}.log" 2>&1
    local status=$?
    if [ "${status}" -ne 0 ]; then
        log "  ${split}: failed with status ${status}"
        tail -5 "${OUT_DIR}/${MODEL}${suffix}.log" | sed 's/^/      /'
        rm -f "${tmp}"
        [ -e "${archived}" ] && mv -f "${archived}" "${final}"
        return 1
    fi

    # The sanity check. A report that is all zeros, or that carries a NaN
    # false-alarm rate, is not a score -- it is the old bug again.
    "${PYTHON}" - "${tmp}" << 'PYEOF'
import json
import math
import sys

report = json.load(open(sys.argv[1]))
csi = [report["thresholds"][t]["csi"] for t in ("0.1", "0.5", "2.0")]
far = [report["thresholds"][t]["far"] for t in ("0.1", "0.5", "2.0")]
bad = []
if max(csi) <= 0.0:
    bad.append("every CSI is zero")
if any(math.isnan(v) for v in far):
    bad.append("a false-alarm rate is NaN")
if any(math.isnan(v) for v in csi):
    bad.append("a CSI is NaN")
if bad:
    print("REJECTED: " + "; ".join(bad))
    sys.exit(1)
print("OK csi=" + ", ".join(f"{v:.4f}" for v in csi)
      + " far=" + ", ".join(f"{v:.4f}" for v in far)
      + f" n={report['n_samples']}")
PYEOF
    if [ $? -ne 0 ]; then
        log "  ${split}: the report did not pass the sanity check -- discarded"
        rm -f "${tmp}"
        [ -e "${archived}" ] && mv -f "${archived}" "${final}"
        return 1
    fi

    # The move is atomic, so a table builder reading the directory never
    # sees a half-written report.
    mv -f "${tmp}" "${final}"
    log "  ${split}: promoted to ${final}"
}

log "=== re-scoring ${MODEL} after the unit fix ==="
stage_one test ".eval"
stage_one robustness_missing ".robustness"

log "=== done ==="
log "  eval reports:       $(ls "${OUT_DIR}"/*.eval.json 2>/dev/null | wc -l)"
log "  robustness reports: $(ls "${OUT_DIR}"/*.robustness.json 2>/dev/null | wc -l)"
