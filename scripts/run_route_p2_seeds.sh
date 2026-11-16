#!/usr/bin/env bash
# Route P2, part 2 only: the seed spread.
#
# This is Part 2 of ``scripts/run_route_p2.sh``, with the Part-1 loop
# left out and one change to the checkpoint handling.  The reason for
# the first change is a cost: Part 1 trains the three curve arms again
# and then scores every kept epoch again, and its scoring loop has no
# "already scored" check -- the skip lives in
# ``scripts/eval_route_p2_curve.sh``, which was written later.  So
# running the whole script would cost about 45 minutes of retraining
# plus roughly 5.5 hours of serial scoring, all of it reproducing
# numbers that are already on disk.
#
# The reason for the second change is the measurement itself.  Part 1
# has just shown that every arm peaks at epoch 0 and that the score
# decays from there -- ``qpe_0`` falls from 0.1303 to 0.0399 over
# twenty epochs, ``base_concat`` from 0.1865 to 0.0218.  Part 2 asks
# whether the difference between the two configurations survives the
# run-to-run noise, and the difference that matters is the one at the
# peak.  ``scripts/train.py`` keeps one checkpoint, ``last.pt``, which
# it rewrites at the end of every epoch: it is the state of the last
# epoch, which is the *worst* epoch by a factor of three to eight.
# Scoring the seeds there would measure the noise of a collapsed
# model and say nothing about the peak.
#
# So the seeds are trained with ``scripts/train_keep_epochs.py``, which
# keeps a copy of every epoch, and each seed is scored at the epochs
# where the peak can be.  The wrapper patches ``torch.save`` rather
# than the training loop, so the training is exactly the code the
# matrix ran; only the bookkeeping of what is written to disk differs.
#
# Everything else -- the models, the hyper-parameters, the seeds, the
# per-window tables, the output paths -- is Part 2 as it stands in
# ``scripts/run_route_p2.sh``.  The one other change is where the
# evaluations run: Part 2 puts all of them on GPU 0, four at a time,
# and route P3 wants GPU 0 for its training.  The evaluations are CPU
# bound (the Hannings stitching runs on the processor), so spreading
# them over GPUs 4 to 7 changes how long they take and nothing about
# what they report -- the same reasoning ``eval_route_p2_curve.sh``
# already relies on when it rotates over all eight cards.
#
# The judge is the file count, not the exit code: ``runs/route_p2/seeds``
# must hold six ``.eval.json`` files and six ``.windows.json`` files
# per scored epoch.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p2}"
EPOCHS="${EPOCHS:-20}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
SEEDS="${SEEDS:-0 1 2}"
# The epochs to score per seed.  Part 1 put every arm's peak at epoch
# 0 and the score fell away from there, so the peak epoch is the one
# that matters.  The last epoch is kept as well: it is what
# ``last.pt`` would have been, it is the epoch the earlier numbers in
# the literature were measured at, and the seed-0 run at that epoch
# must reproduce the Part-1 curve exactly, which is a check on the
# whole pipeline end to end.
SCORE_EPOCHS="${SCORE_EPOCHS:-00 19}"

mkdir -p "${OUT_DIR}"

SEED_DIR="${OUT_DIR}/seeds"
mkdir -p "${SEED_DIR}"

echo "=========================================================="
echo " P2 part 2: ${SEEDS} seeds per configuration"
echo " the criterion and the curve are Part 1's job"
echo " scoring epochs: ${SCORE_EPOCHS}"
echo "=========================================================="

for entry in "qpe_0 simvp_v2_qpe_guided" "base_concat simvp_v2_nwp_concat"; do
    read -r name model <<< "${entry}"
    for seed in ${SEEDS}; do
        # Cards 0 to 3 are route P3's while the two run together, so the
        # seeds take 4, 5 and 6.  Which card a run lands on changes how
        # long it takes and nothing about what it measures.
        gpu=$(( (seed + 4) % 8 ))
        log="${SEED_DIR}/${name}_s${seed}.log"
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
                --out-dir "${SEED_DIR}/${name}_s${seed}" \
                --loss balanced \
                --alpha-csi "${ALPHA_CSI}" \
                --tier-thresholds ${TIERS} \
                > "${log}" 2>&1
        ) &
        # Three at a time: eight concurrent runs on eight cards would put
        # three per card and the epoch time would triple.
        while [[ $(jobs -rp | wc -l) -ge 3 ]]; do sleep 5; done
    done
done
wait

echo "training done; scoring every seed with the per-window tables"

slot=0
for entry in "qpe_0 simvp_v2_qpe_guided" "base_concat simvp_v2_nwp_concat"; do
    read -r name model <<< "${entry}"
    for seed in ${SEEDS}; do
        run_dir="$(ls -dt "${SEED_DIR}/${name}_s${seed}"/*/ 2>/dev/null | head -1)"
        [[ -n "${run_dir}" ]] || continue
        for ep in ${SCORE_EPOCHS}; do
            ck="${run_dir}epoch_${ep}.pt"
            [[ -f "${ck}" ]] || continue
            out="${SEED_DIR}/${name}_s${seed}_ep${ep}.eval.json"
            # A report that parses is a report that is finished; anything
            # else is scored again.
            if [[ -f "${out}" ]] && "${PYTHON}" -c "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null; then
                echo "  ${name} seed ${seed} epoch ${ep}: already scored, skipped"
                continue
            fi
            gpu=$(( 5 + slot % 3 ))
            slot=$(( slot + 1 ))
            (
                cd "${REPO_ROOT}"
                CUDA_VISIBLE_DEVICES="${gpu}" \
                "${PYTHON}" scripts/evaluate.py \
                    --model "${model}" \
                    --checkpoint "${ck}" \
                    --data-root "${DATA_ROOT}" \
                    --packed --split test \
                    --batch-size "${BATCH_SIZE}" --num-workers "${WORKERS}" \
                    --patch-size "${PATCH_SIZE}" --slide-stride "${SLIDE_STRIDE}" \
                    --thresholds ${TIERS} \
                    --device cuda \
                    --out-json "${out}" \
                    --per-window-out "${SEED_DIR}/${name}_s${seed}_ep${ep}.windows.json" \
                    > "${SEED_DIR}/${name}_s${seed}_ep${ep}.eval.log" 2>&1
            ) &
            while [[ $(jobs -rp | wc -l) -ge 4 ]]; do sleep 5; done
        done
    done
done
wait

echo "the judge is the file count:"
for name in qpe_0 base_concat; do
    for seed in ${SEEDS}; do
        for ep in ${SCORE_EPOCHS}; do
            n=0
            [[ -f "${SEED_DIR}/${name}_s${seed}_ep${ep}.eval.json" ]] && n=$(( n + 1 ))
            [[ -f "${SEED_DIR}/${name}_s${seed}_ep${ep}.windows.json" ]] && n=$(( n + 1 ))
            echo "  ${name} seed ${seed} epoch ${ep}: ${n} / 2"
        done
    done
done

echo "=========================================================="
echo " P2 part 2 done. Next: tools/_analyze_p2.py"
echo "=========================================================="
