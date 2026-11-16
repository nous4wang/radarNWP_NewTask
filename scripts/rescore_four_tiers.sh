#!/usr/bin/env bash
# Score the three stages again, after the output-path defect was fixed.
#
# The training of all three stages is finished and every checkpoint is on
# disk, so only the scoring is repeated. The reports that were produced
# before the fix are not kept: the output path was keyed on the run
# directory, whose name is the wall clock at launch, so the seeds that
# started in the same second overwrote one another and the set that came
# back was both short and unlabelled -- sixteen evaluations produced six
# reports, and which seed each of the six belonged to was not recoverable.
#
# The stages are scored one after another. They all want the whole
# machine: the verification is bound by the processor and the memory
# bandwidth, and running them together would make each of them slower.
#
# The judge is the number of reports written, not the exit code.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/runs/four_tiers}"

cd "${REPO_ROOT}"

echo "=========================================================="
echo " clearing the reports written under the defective paths"
echo "=========================================================="
for d in r5_soft_fss r4_lagrangian r2_decoupled; do
    if [ -d "${LOG_DIR}/${d}" ]; then
        rm -rf "${LOG_DIR}/${d}"/eval "${LOG_DIR}/${d}"/eval_CORRUPT_basename_collision
        echo "  ${d}: cleared"
    fi
done

score() {
    local name="$1"; shift
    local dir="$1"; shift
    echo "=========================================================="
    echo " scoring ${name}"
    echo "=========================================================="
    "${PYTHON}" tools/_eval_four_tiers.py "${LOG_DIR}/${dir}" --epochs 1
    echo "SCORED_${name}=$?"
}

score "stage2" "r5_soft_fss"
score "stage3" "r4_lagrangian"
score "stage5" "r2_decoupled"

echo "=========================================================="
echo " what the advection model does with the physics"
echo "=========================================================="
# The score says whether the model is better. It does not say whether the
# model uses the advection at all: a gate that sits at zero means the
# model has decided the flow is not worth using, and its score has to be
# read as the score of the residual arm.
"${PYTHON}" tools/_probe_gate.py "${LOG_DIR}/r4_lagrangian" \
    --data-root "${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}" --packed \
    --split test --batches 6 --batch-size 8 \
    --out-json "${LOG_DIR}/gate_probe.json"
echo "GATE_PROBE_EXIT=$?"

echo "=========================================================="
echo " the reports that exist now"
echo "=========================================================="
for d in r5_soft_fss r4_lagrangian r2_decoupled; do
    n="$(ls "${LOG_DIR}/${d}"/eval/*/*.eval.json 2>/dev/null | wc -l)"
    w="$(ls "${LOG_DIR}/${d}"/eval/*/*.windows.json 2>/dev/null | wc -l)"
    echo "  ${d}: ${n} reports, ${w} window tables"
done
echo "RESCORE_DONE=0"
