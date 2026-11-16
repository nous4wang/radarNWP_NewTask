#!/usr/bin/env bash
# E41 batch 1 -- training only. PRE-REGISTERED criteria: E41 section 5
# (frozen 2026-11-09 09:29). Remote S0 gate PASSED (max_abs_diff 0.0).
#
# Single-variable matrix (section 4):
#   B0  simvp_v2     + standard   (same-batch control)
#   A0  struct_hurdle head=rain  + standard   (implementation equivalence)
#   A1  struct_hurdle head=hurdle + hurdle    (output parameterisation)
#   A2  struct_hurdle head=rain  + uot        (loss geometry)
# A3 is NOT in this batch (only after A1/A2 show a signal).
#
# Everything except model / model-kwargs / loss is identical across arms.
# No evaluation is launched here: S1/S3 need the same test-538 windows and
# the section 2 structure metrics, which is a separate step.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
DATA=/home/wf/data/radarNWP/BJ_mmap
OUT=/home/wf/radarNWP/runs/e41
TRAIN="${OUT}/train"
LOGS="${OUT}/logs"
STAGE="${OUT}/stage.log"
EPOCHS=3
BATCH=16
WORKERS=6
PATCH=64
STRIDE=64
PAR=4
SEEDS="0 1 2"

mkdir -p "${TRAIN}" "${LOGS}"
: > "${STAGE}"
log() { echo "$1 $(date '+%F %T')" | tee -a "${STAGE}"; }
log "E41 BATCH1 START (S0 PASS; criteria frozen 2026-11-09 09:29)"

# A one-step smoke of each NEW loss branch before spending 12 full runs.
# It writes only under /tmp and produces no score, so it is not an arm.
log "SMOKE start: one training step per new loss branch (writes only /tmp)"
for spec in \
    "A1 struct_hurdle {\"head_mode\":\"hurdle\"} hurdle" \
    "A2 struct_hurdle {} uot" ; do
    read -r arm model kwargs loss <<< "${spec}"
    CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/train.py \
        --model "${model}" \
        --data-root "${DATA}" --packed --split train \
        --epochs 1 --batch-size 2 --num-workers 0 \
        --patch-size "${PATCH}" --slide-stride "${STRIDE}" \
        --device cuda --seed 0 \
        --max-steps 1 --log-every 1 \
        --out-dir "/tmp/e41_smoke_${arm}" \
        --loss "${loss}" --qpe-weight 0.0 \
        --model-kwargs "${kwargs}" \
        > "${LOGS}/smoke_${arm}.log" 2>&1
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        log "ABORT: smoke ${arm} rc=${rc}; no batch training will be launched"
        tail -n 25 "${LOGS}/smoke_${arm}.log" >> "${STAGE}"
        exit 1
    fi
    log "SMOKE ${arm} OK"
done
log "SMOKE PASS; launching 12 training runs"

run_one() {
    local arm="$1" model="$2" kwargs="$3" loss="$4" seed="$5" gpu="$6"
    local d="${TRAIN}/${arm}_s${seed}"
    local lf="${LOGS}/${arm}_s${seed}.log"
    mkdir -p "${d}"
    if [ -n "$(ls "${d}"/*/epoch_02.pt 2>/dev/null)" ]; then
        log "SKIP ${arm} s${seed} (epoch_02 already present)"
        return 0
    fi
    (
        cd /home/wf/radarNWP || exit 1
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 \
        "${PY}" scripts/train_keep_epochs.py \
            --model "${model}" \
            --data-root "${DATA}" --packed --split train \
            --epochs "${EPOCHS}" --batch-size "${BATCH}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH}" --slide-stride "${STRIDE}" \
            --device cuda --seed "${seed}" \
            --out-dir "${d}" \
            --loss "${loss}" \
            --qpe-weight 0.0 \
            --model-kwargs "${kwargs}" \
            > "${lf}" 2>&1
    )
    local rc=$?
    echo "EXIT arm=${arm} seed=${seed} gpu=${gpu} rc=${rc} $(date '+%F %T')" >> "${STAGE}"
}

slot=0
for seed in ${SEEDS}; do
    for spec in \
        "B0 simvp_v2 {} standard" \
        "A0 struct_hurdle {} standard" \
        "A1 struct_hurdle {\"head_mode\":\"hurdle\"} hurdle" \
        "A2 struct_hurdle {} uot" ; do
        read -r arm model kwargs loss <<< "${spec}"
        gpu=$(( slot % 8 ))
        slot=$(( slot + 1 ))
        log "LAUNCH ${arm} s${seed} gpu=${gpu} model=${model} loss=${loss} kwargs=${kwargs}"
        run_one "${arm}" "${model}" "${kwargs}" "${loss}" "${seed}" "${gpu}" &
        while [ "$(jobs -rp | wc -l)" -ge "${PAR}" ]; do sleep 5; done
    done
done
wait

missing=0
for seed in ${SEEDS}; do
    for spec in \
        "B0 simvp_v2 {} standard" \
        "A0 struct_hurdle {} standard" \
        "A1 struct_hurdle {\"head_mode\":\"hurdle\"} hurdle" \
        "A2 struct_hurdle {} uot" ; do
        read -r arm model kwargs loss <<< "${spec}"
        n=$(ls "${TRAIN}/${arm}_s${seed}"/*/epoch_0{0,1,2}.pt 2>/dev/null | wc -l)
        echo "CHECK ${arm} s${seed}: ${n}/3 epochs" >> "${STAGE}"
        [ "${n}" -ge 3 ] || missing=$(( missing + 1 ))
    done
done

log "E41 BATCH1 DONE missing_epoch_files=${missing}; EVAL NOT LAUNCHED (separate step)"
echo "training outputs: ${TRAIN}; per-job logs: ${LOGS}; stage: ${STAGE}"
