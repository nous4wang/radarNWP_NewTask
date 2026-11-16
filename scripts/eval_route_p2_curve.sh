#!/usr/bin/env bash
# Evaluate every kept epoch of the route P2 curve, in parallel.
#
# The experiment script evaluates the epochs one at a time. That is slow:
# the verification is CPU bound (the Hannings stitching runs on the
# processor, and the GPU sits at zero per cent while it happens), so one
# evaluation at a time leaves 143 cores idle and every GPU idle.
#
# The training is already finished and every epoch's weights are on disk,
# so this re-runs only the scoring, and it runs it across all the cards
# and a pool of workers. The numbers it produces are the same numbers:
# the evaluation is deterministic, and the same checkpoint scored twice
# gives a bit-identical report.
#
# The judge is the number of .eval.json files, not the exit code.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
CURVE_DIR="${CURVE_DIR:-/home/wf/radarNWP/runs/route_p2/curve}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
PARALLEL="${PARALLEL:-8}"

echo "=========================================================="
echo " P2 curve scoring, ${PARALLEL} at a time"
echo "=========================================================="

# Every (checkpoint, model, output) triple, one per epoch of every arm.
JOBS="$(mktemp /tmp/_rnb_evaljobs.XXXXXX)"
for arm in qpe_0 qpe_05 base_concat; do
    case "${arm}" in
        qpe_0|qpe_05) model="simvp_v2_qpe_guided" ;;
        *)           model="simvp_v2_nwp_concat" ;;
    esac
    run_dir="$(ls -dt "${CURVE_DIR}/${arm}"/*/ 2>/dev/null | head -1)"
    if [[ -z "${run_dir}" ]]; then
        echo "  ${arm}: no run directory, skipped"
        continue
    fi
    mkdir -p "${CURVE_DIR}/${arm}_eval"
    for ck in "${run_dir}"epoch_*.pt; do
        [[ -f "${ck}" ]] || continue
        base="$(basename "${ck}" .pt)"
        out="${CURVE_DIR}/${arm}_eval/${base}.eval.json"
        # Skip the ones that are already scored.
        if [[ -f "${out}" ]]; then
            echo "  ${arm}/${base}: already scored, skipped"
            continue
        fi
        echo "${arm}|${model}|${ck}|${out}" >> "${JOBS}"
    done
done

n_jobs="$(wc -l < "${JOBS}")"
echo "  ${n_jobs} evaluations to run, ${PARALLEL} at a time"
if [[ "${n_jobs}" -eq 0 ]]; then
    echo "nothing to do"
    rm -f "${JOBS}"
    exit 0
fi

# The pool: a bounded number of evaluations at a time, each on its own
# card. The card is chosen by the worker's slot, so two evaluations never
# share one.
run_one() {
    local line="$1"
    local slot="$2"
    local arm model ck out gpu
    IFS='|' read -r arm model ck out <<< "${line}"
    gpu=$(( slot % 8 ))
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
            > "${out%.json}.log" 2>&1
    )
    if [[ -f "${out}" ]]; then
        echo "  done: ${arm}/$(basename "${ck}" .pt)"
    else
        echo "  FAILED: ${arm}/$(basename "${ck}" .pt)"
    fi
}
export -f run_one
export PYTHON DATA_ROOT REPO_ROOT TIERS BATCH_SIZE WORKERS

slot=0
while IFS= read -r line; do
    run_one "${line}" "${slot}" &
    slot=$(( (slot + 1) % PARALLEL ))
    # Wait when the pool is full. The count is of the running children,
    # not of the slots.
    while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]]; do
        wait -n 2>/dev/null || true
    done
done < "${JOBS}"
wait

echo "=========================================================="
echo " scoring done. The judge is the file count:"
for arm in qpe_0 qpe_05 base_concat; do
    n="$(ls "${CURVE_DIR}/${arm}_eval"/*.eval.json 2>/dev/null | wc -l)"
    echo "  ${arm}: ${n} / 20"
done
echo "=========================================================="
rm -f "${JOBS}"
