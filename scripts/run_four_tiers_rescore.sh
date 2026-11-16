#!/usr/bin/env bash
# Re-score the stages whose scoring was written into colliding directories,
# and measure what the advection model does with the physics.
#
# The scorer used to key its output on the run directory's name, and the
# training stamps that name with the wall clock. Ten seeds of one arm
# therefore fell into a handful of names, and the second arm to finish
# overwrote the first one's report. The scorer now keys the output on the
# arm directory above it, so this re-scores what was lost.
#
# The judge is the number of reports written, not the exit code.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/runs/four_tiers}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"

cd "${REPO_ROOT}"

echo "=========================================================="
echo " waiting for the fifth stage to release the machine"
echo "=========================================================="
# The scoring is bound by the processor and the memory bandwidth, and
# the training wants a card each. Running the two together makes each
# of them slower than if it had the machine to itself.
while pgrep -f "train_keep_epochs.py" > /dev/null 2>&1; do
    sleep 60
done
echo " the training has finished"

# The fifth stage's own scoring failed on a stale copy of the scorer,
# and its missing-frame pairing is still running: it scores the same
# twenty-three cases twice, once with the interpolation repaired and
# once with it switched off, and that is the size of a full scoring.
# The pairing is not waited for here. It uses four workers, the
# machine has a hundred and forty-four cores, and the fifty-eight
# evaluations below are the long pole. Waiting for the pairing first
# would add an hour to everything and change no number.
echo " the missing-frame pairing is still running; it is not waited for"

echo "=========================================================="
echo " stage five: scoring the lag sweep"
echo "=========================================================="
"${PYTHON}" tools/_eval_four_tiers.py runs/four_tiers/r2_decoupled --epochs 1
echo "STAGE5_REDO_EXIT=$?"
n5="$(find runs/four_tiers/r2_decoupled/eval -name '*.eval.json' 2>/dev/null | wc -l)"
echo "  reports under the fifth stage: ${n5}"

echo "=========================================================="
echo " stage two: re-scoring the fractions arm"
echo "=========================================================="
"${PYTHON}" tools/_eval_four_tiers.py runs/four_tiers/r5_soft_fss --epochs 1
echo "STAGE2_REDO_EXIT=$?"
n2="$(find runs/four_tiers/r5_soft_fss/eval -name '*.eval.json' 2>/dev/null | wc -l)"
echo "  reports under the second stage: ${n2}"

echo "=========================================================="
echo " stage three: re-scoring the advection arms"
echo "=========================================================="
"${PYTHON}" tools/_eval_four_tiers.py runs/four_tiers/r4_lagrangian --epochs 1
echo "STAGE3_REDO_EXIT=$?"
n3="$(find runs/four_tiers/r4_lagrangian/eval -name '*.eval.json' 2>/dev/null | wc -l)"
echo "  reports under the third stage: ${n3}"

echo "=========================================================="
echo " the gate weight and the mass ratio"
echo "=========================================================="
"${PYTHON}" tools/_probe_gate.py runs/four_tiers/r4_lagrangian \
    --data-root "${DATA_ROOT}" --packed --split test \
    --batches 6 --batch-size 8 \
    --out-json runs/four_tiers/gate_probe.json
echo "GATE_PROBE_EXIT=$?"

echo "=========================================================="
echo " everything is scored"
echo "=========================================================="
for d in r5_soft_fss r4_lagrangian r2_decoupled; do
    n="$(find "${LOG_DIR}/${d}/eval" -name '*.eval.json' 2>/dev/null | wc -l)"
    echo "  ${d}: ${n} reports"
done
