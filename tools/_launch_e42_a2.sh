#!/usr/bin/env bash
# E42 -- train the FIXED A2 arm (A2').  The criteria are frozen (E42
# section 2, 2026-11-09 14:22) and are NOT touched here.
#
# What changed relative to E41's A2 arm: ONE thing -- how the UOT term
# normalises its masses (shape normalisation + log mass ratio instead of
# dividing by each sample's own truth total).  The geometry of the term
# (grid, eps, tau, n_iters, cost, rho, mass_weight) is unchanged, the
# model is unchanged, the data is unchanged, the evaluator is unchanged.
#
# Everything else is byte-identical to E41's batch 1 (see
# tools/_launch_e41_batch1.sh), so A2' is comparable with E41's B0 and
# A0 -- and those two weights are REUSED, not retrained.
#
# A one-step smoke of the fixed loss branch runs before the batch: if it
# fails, nothing is launched.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
OUT=/home/wf/radarNWP/runs/e42
TRAIN="${OUT}/train"
LOGS="${OUT}/logs"
STAGE="${OUT}/stage.log"
EPOCHS=3
BATCH=16
WORKERS=6
PATCH=64
STRIDE=64
PAR=3
SEEDS="0 1 2"

mkdir -p "${TRAIN}" "${LOGS}"
: > "${STAGE}"
log() { echo "$1 $(date '+%F %T')" | tee -a "${STAGE}"; }
log "E42 A2' START (scale fixed; criteria frozen 2026-11-09 14:22)"

# F3: one training step of the fixed loss branch.  It writes only under
# /tmp and produces no score, so it is not an arm.
log "SMOKE start: one training step of the fixed uot branch"
CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/train.py \
    --model struct_hurdle \
    --data-root "${DATA}" --packed --split train \
    --epochs 1 --batch-size 2 --num-workers 0 \
    --patch-size "${PATCH}" --slide-stride "${STRIDE}" \
    --device cuda --seed 0 \
    --max-steps 1 --log-every 1 \
    --out-dir "/tmp/e42_smoke" \
    --loss uot --qpe-weight 0.0 \
    --model-kwargs '{}' \
    > "${LOGS}/smoke.log" 2>&1
rc=$?
if [ "${rc}" -ne 0 ]; then
    log "ABORT: smoke rc=${rc}; no training will be launched"
    tail -n 25 "${LOGS}/smoke.log" >> "${STAGE}"
    exit 1
fi
first_loss=$(grep -o "loss=[0-9.]*" "${LOGS}/smoke.log" | head -1 | cut -d= -f2)
log "SMOKE OK first_loss=${first_loss}"
# F3: the fixed term must not produce a loss above 100 on its first step.
big=$(awk -F'loss=' '/step .*loss=/{split($2,a," "); if (a[1]+0 > 100) c++} END{print c+0}' "${LOGS}/smoke.log")
if [ "${big}" -ne 0 ]; then
    log "ABORT: F3 failed -- ${big} step(s) above loss 100 on the smoke run"
    tail -n 25 "${LOGS}/smoke.log" >> "${STAGE}"
    exit 1
fi
log "F3 PASS: no step above loss 100"

run_one() {
    local seed="$1" gpu="$2"
    local d="${TRAIN}/A2p_s${seed}"
    local lf="${LOGS}/A2p_s${seed}.log"
    mkdir -p "${d}"
    if [ -n "$(ls "${d}"/*/epoch_02.pt 2>/dev/null)" ]; then
        log "SKIP A2' s${seed} (epoch_02 already present)"
        return 0
    fi
    (
        cd /home/wf/radarNWP || exit 1
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 \
        "${PY}" scripts/train_keep_epochs.py \
            --model struct_hurdle \
            --data-root "${DATA}" --packed --split train \
            --epochs "${EPOCHS}" --batch-size "${BATCH}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH}" --slide-stride "${STRIDE}" \
            --device cuda --seed "${seed}" \
            --out-dir "${d}" \
            --loss uot \
            --qpe-weight 0.0 \
            --model-kwargs '{}' \
            > "${lf}" 2>&1
    )
    local rc=$?
    echo "EXIT arm=A2p seed=${seed} gpu=${gpu} rc=${rc} $(date '+%F %T')" >> "${STAGE}"
}

slot=0
for seed in ${SEEDS}; do
    gpu=$(( slot % 8 ))
    slot=$(( slot + 1 ))
    log "LAUNCH A2' s${seed} gpu=${gpu}"
    run_one "${seed}" "${gpu}" &
    while [ "$(jobs -rp | wc -l)" -ge "${PAR}" ]; do sleep 5; done
done
wait

missing=0
for seed in ${SEEDS}; do
    n=$(ls "${TRAIN}/A2p_s${seed}"/*/epoch_0{0,1,2}.pt 2>/dev/null | wc -l)
    echo "CHECK A2' s${seed}: ${n}/3 epochs" >> "${STAGE}"
    [ "${n}" -ge 3 ] || missing=$(( missing + 1 ))
done
log "E42 A2' DONE missing_epoch_files=${missing}"
echo "training outputs: ${TRAIN}; logs: ${LOGS}; stage: ${STAGE}"
