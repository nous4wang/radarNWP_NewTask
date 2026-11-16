#!/usr/bin/env bash
# E42 -- structure metrics for the fixed A2' arm (S1, the primary criterion).
#
# The metrics come from tools/_e41_struct_metrics.py, which IMPORTS the
# section 2 probe's own implementation rather than copying it, so the four
# structure quantities of S1 are measured with the same code that measured
# the label.  Nothing here trains or evaluates a model.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
OUT=/home/wf/radarNWP/runs/e42

"${PY}" tools/_e41_struct_metrics.py \
    --fields "${OUT}/fields" \
    --tags A2p_s0_ep00 A2p_s1_ep00 A2p_s2_ep00 \
    --out "${OUT}/struct_metrics.json"
rc=$?
echo "E42 STRUCT METRICS rc=${rc}"
exit ${rc}
