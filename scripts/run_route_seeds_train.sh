#!/usr/bin/env bash
# Train the extra seeds that route P2's pass-rate study and route P3's
# lag sweep need.
#
# Why this is cheap
# -----------------
# The weight of an epoch does not depend on how many epochs the run was
# going to do.  There is no learning-rate schedule in ``scripts/train.py``
# (the optimiser is a plain AdamW at a fixed learning rate), the order of
# the training batches is fixed by the seed, and the state the optimiser
# starts an epoch with depends only on the epochs before it.  So a
# three-epoch run produces exactly the ``epoch_00``, ``epoch_01`` and
# ``epoch_02`` that a twenty-epoch run produces.
#
# That is not an argument, it is a measurement: on the far side, a
# one-epoch run and a twenty-epoch run agree byte for byte on
# ``epoch_00.pt``, and a three-epoch run agrees with the twenty-epoch run
# on ``epoch_00.pt``, ``epoch_01.pt`` and ``epoch_02.pt``.
#
# The peak of every arm is in the first three epochs (measured, section
# 4.12.1 of the master document), so three epochs are enough to locate
# the peak of a new seed and to measure the pass rate at it.
#
# The judge is the file count: one directory per (arm, seed) under
# ``runs/route_seeds/train``, each holding ``epoch_00.pt`` ..
# ``epoch_02.pt``.  A run whose directory already holds those three
# files is skipped, so the script is safe to run again.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_seeds}"
TRAIN_DIR="${OUT_DIR}/train"
EPOCHS="${EPOCHS:-3}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
# How many trainings at a time.  Four leaves four cards free and keeps
# the epoch time near what a single run costs; eight puts two on every
# card and roughly doubles the epoch time.
PARALLEL="${PARALLEL:-4}"

mkdir -p "${TRAIN_DIR}"

# name  model  lag  seed
JOBS=(
    # Route P3: the lag sweep.  Seed 0 is already trained (runs/route_p3_epochs).
    "assim_lag0 simvp_v2_assimilated {\"obs_lag_steps\":0} 1"
    "assim_lag0 simvp_v2_assimilated {\"obs_lag_steps\":0} 2"
    "assim_lag2 simvp_v2_assimilated {\"obs_lag_steps\":2} 1"
    "assim_lag2 simvp_v2_assimilated {\"obs_lag_steps\":2} 2"
    "assim_lag4 simvp_v2_assimilated {\"obs_lag_steps\":4} 1"
    "assim_lag4 simvp_v2_assimilated {\"obs_lag_steps\":4} 2"
    "assim_lag11 simvp_v2_assimilated {\"obs_lag_steps\":11} 1"
    "assim_lag11 simvp_v2_assimilated {\"obs_lag_steps\":11} 2"
    # Route P2: the pass rate at the peak, ten seeds per configuration.
    "qpe_0 simvp_v2_qpe_guided {} 3"
    "qpe_0 simvp_v2_qpe_guided {} 4"
    "qpe_0 simvp_v2_qpe_guided {} 5"
    "qpe_0 simvp_v2_qpe_guided {} 6"
    "qpe_0 simvp_v2_qpe_guided {} 7"
    "qpe_0 simvp_v2_qpe_guided {} 8"
    "qpe_0 simvp_v2_qpe_guided {} 9"
    "base_concat simvp_v2_nwp_concat {} 3"
    "base_concat simvp_v2_nwp_concat {} 4"
    "base_concat simvp_v2_nwp_concat {} 5"
    "base_concat simvp_v2_nwp_concat {} 6"
    "base_concat simvp_v2_nwp_concat {} 7"
    "base_concat simvp_v2_nwp_concat {} 8"
    "base_concat simvp_v2_nwp_concat {} 9"
)

echo "=========================================================="
echo " route P2/P3 extra seeds"
echo " ${EPOCHS} epochs per run (the peak is inside them)"
echo " ${PARALLEL} trainings at a time"
echo "=========================================================="

slot=0
for entry in "${JOBS[@]}"; do
    read -r name model kwargs seed <<< "${entry}"
    out_dir="${TRAIN_DIR}/${name}_s${seed}"
    # ``[[ -f dir/*/file ]]`` does not work: the pattern is not expanded
    # inside ``[[ ]]``, so the test is always false.  The glob has to be
    # resolved by the shell first.
    if [[ -n "$(ls "${out_dir}"/*/epoch_02.pt 2>/dev/null)" ]]; then
        echo "  ${name} seed ${seed}: already trained, skipped"
        continue
    fi
    gpu=$(( slot % 8 ))
    slot=$(( slot + 1 ))
    echo "  -> ${name} seed ${seed} on GPU ${gpu}"
    (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS=4 \
        "${PYTHON}" scripts/train_keep_epochs.py \
            --model "${model}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split train \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --device cuda \
            --seed "${seed}" \
            --out-dir "${out_dir}" \
            --loss balanced \
            --alpha-csi "${ALPHA_CSI}" \
            --tier-thresholds ${TIERS} \
            --qpe-weight 0.0 \
            --model-kwargs "${kwargs}" \
            > "${TRAIN_DIR}/${name}_s${seed}.log" 2>&1
    ) &
    while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]]; do sleep 5; done
done
wait

echo "training done; the judge is the file count:"
missing=0
for entry in "${JOBS[@]}"; do
    read -r name model kwargs seed <<< "${entry}"
    n=0
    for ep in 00 01 02; do
        [[ -n "$(ls "${TRAIN_DIR}/${name}_s${seed}"/*/epoch_${ep}.pt 2>/dev/null)" ]] \
            && n=$(( n + 1 ))
    done
    echo "  ${name} seed ${seed}: ${n} / 3"
    [[ "${n}" -eq 3 ]] || missing=$(( missing + 1 ))
done

# ---------------------------------------------------------------------------
# The check that goes with the numbers: ``assim_lag11`` with the observed
# history withheld is the same model as ``qpe_0``.  The master document
# says their weights agree; this says whether that holds for the new seeds
# too, and it costs nothing because the weights are already on disk.
# ---------------------------------------------------------------------------
echo
echo "the no-observation endpoint is qpe_0:"
for seed in 1 2; do
    a_dir="$(ls -dt "${TRAIN_DIR}/assim_lag11_s${seed}"/*/ 2>/dev/null | head -1)"
    b_dir="$(ls -dt "/home/wf/radarNWP/runs/route_p2/seeds/qpe_0_s${seed}"/*/ 2>/dev/null | head -1)"
    if [[ -z "${a_dir}" || -z "${b_dir}" ]]; then
        echo "  seed ${seed}: a directory is missing, skipped"
        continue
    fi
    same=1
    for ep in 00 01 02; do
        a="$(md5sum "${a_dir}epoch_${ep}.pt" 2>/dev/null | cut -d' ' -f1)"
        b="$(md5sum "${b_dir}epoch_${ep}.pt" 2>/dev/null | cut -d' ' -f1)"
        [[ -n "${a}" && "${a}" = "${b}" ]] || same=0
    done
    if [[ "${same}" -eq 1 ]]; then
        echo "  seed ${seed}: identical to qpe_0 at epochs 00, 01, 02"
    else
        echo "  seed ${seed}: DIFFERENT from qpe_0"
    fi
done

echo "=========================================================="
echo " extra seeds done (${missing} runs missing an epoch)."
echo "=========================================================="
