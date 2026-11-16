#!/usr/bin/env bash
# Score the route P2 seed checkpoints that are ready, and skip the rest.
#
# ``scripts/run_route_p2_seeds.sh`` trains all six seed runs first and
# scores them afterwards.  The two waves of training do not finish at the
# same time, and the scoring is the slow part, so waiting for the last
# training to finish leaves the machine idle.
#
# This script scores whatever is on disk and leaves the rest alone.  It
# is the same evaluation as Part 2's -- the same model, the same
# checkpoint, the same thresholds, the same per-window tables -- so a
# report produced here is the same report Part 2 would have produced.
# A report that already exists and parses is left alone, which is what
# makes this safe to run while Part 2 is still going.
#
# A lock directory keeps two scorers off the same file: the run script
# and this one can both be alive, and a file that is half written must
# not be read as if it were finished.
#
# The judge is the file count: for every arm and every epoch, one
# ``.eval.json`` and one ``.windows.json``.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
SCORE_EPOCHS="${SCORE_EPOCHS:-00 19}"
# How many evaluations at a time.  The measurement in the master
# document says eight is the best the machine does: the verification is
# bound by memory bandwidth, not by the number of cards, so more
# concurrency does not go faster.
PARALLEL="${PARALLEL:-6}"
LOCK_DIR="${OUT_DIR}/.eval_locks"

mkdir -p "${LOCK_DIR}"

SEED_DIR="${OUT_DIR}/seeds"
mkdir -p "${SEED_DIR}"

running=0
for entry in "qpe_0 simvp_v2_qpe_guided" "base_concat simvp_v2_nwp_concat"; do
    read -r name model <<< "${entry}"
    for seed in 0 1 2; do
        run_dir="$(ls -dt "${SEED_DIR}/${name}_s${seed}"/*/ 2>/dev/null | head -1)"
        [[ -n "${run_dir}" ]] || continue
        for ep in ${SCORE_EPOCHS}; do
            ck="${run_dir}epoch_${ep}.pt"
            [[ -f "${ck}" ]] || continue
            out="${SEED_DIR}/${name}_s${seed}_ep${ep}.eval.json"
            win="${SEED_DIR}/${name}_s${seed}_ep${ep}.windows.json"
            if [[ -f "${out}" ]] && "${PYTHON}" -c \
                "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null; then
                echo "  ${name} seed ${seed} epoch ${ep}: already scored"
                continue
            fi
            lock="${LOCK_DIR}/${name}_s${seed}_ep${ep}"
            if ! mkdir "${lock}" 2>/dev/null; then
                echo "  ${name} seed ${seed} epoch ${ep}: another scorer has it"
                continue
            fi
            gpu=$(( 4 + running % 4 ))
            (
                cd "${REPO_ROOT}"
                CUDA_VISIBLE_DEVICES="${gpu}" \
                "${PYTHON}" scripts/evaluate.py \
                    --model "${model}" \
                    --checkpoint "${ck}" \
                    --data-root "${DATA_ROOT}" \
                    --packed --split test \
                    --batch-size "${BATCH_SIZE}" --num-workers "${WORKERS}" \
                    --patch-size 64 --slide-stride 64 \
                    --thresholds ${TIERS} \
                    --device cuda \
                    --out-json "${out}" \
                    --per-window-out "${win}" \
                    > "${out%.json}.log" 2>&1
                rmdir "${lock}" 2>/dev/null
            ) &
            running=$(( running + 1 ))
            while [[ $(jobs -rp | wc -l) -ge ${PARALLEL} ]]; do sleep 5; done
        done
    done
done
wait

echo "the judge is the file count:"
for name in qpe_0 base_concat; do
    for seed in 0 1 2; do
        for ep in ${SCORE_EPOCHS}; do
            n=0
            [[ -f "${SEED_DIR}/${name}_s${seed}_ep${ep}.eval.json" ]] && n=$(( n + 1 ))
            [[ -f "${SEED_DIR}/${name}_s${seed}_ep${ep}.windows.json" ]] && n=$(( n + 1 ))
            echo "  ${name} seed ${seed} epoch ${ep}: ${n} / 2"
        done
    done
done
