#!/usr/bin/env bash
# The four tiers, run in order, with the third and fifth waiting for the second.
#
# The stages are chained rather than run side by side because they all
# want every card: the scoring is bound by the processor and the memory
# bandwidth, and the training wants a card each. Running them together
# makes each of them slower than if it had the machine to itself.
#
# The order is the one the review settled on. The second stage does not
# depend on the Oracle's answer; the third does, and its criteria were
# revised after the Oracle came back; the fifth needs the decoupling to
# be in place, which it is.
#
# The judge for each stage is the number of reports written, not the exit
# code of the driver.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/runs/four_tiers}"

cd "${REPO_ROOT}"

wait_for_stage2() {
    echo "waiting for the second stage to finish"
    while ! grep -q "STAGE2_DONE" "${LOG_DIR}/stage2b.log" 2>/dev/null; do
        if ! ps -eo args | grep -q "[e]val_four_tiers"; then
            # the scoring process is gone but the marker was never written:
            # it died, and waiting for it would wait for ever
            if ! grep -q "STAGE2_DONE" "${LOG_DIR}/stage2b.log" 2>/dev/null; then
                echo "the second stage is not running and left no marker"
                return 1
            fi
        fi
        sleep 60
    done
    echo "the second stage is done"
    return 0
}

echo "=========================================================="
echo " waiting for the second stage"
echo "=========================================================="
if ! wait_for_stage2; then
    echo "stopping here: the second stage did not finish"
    exit 1
fi

echo "=========================================================="
echo " stage three: the advection model and its ablation arms"
echo "=========================================================="
bash scripts/run_four_tiers_stage3.sh
echo "STAGE3_DONE=$?"

echo "=========================================================="
echo " stage five: the decoupled lag sweep and the missing-frame pairing"
echo "=========================================================="
bash scripts/run_four_tiers_stage5.sh
echo "STAGE5_DONE=$?"

echo "=========================================================="
echo " all four tiers have run"
echo "=========================================================="
for d in r5_soft_fss r4_lagrangian r2_decoupled; do
    n="$(ls "${LOG_DIR}/${d}"/eval/*/*.eval.json 2>/dev/null | wc -l)"
    echo "  ${d}: ${n} reports"
done
