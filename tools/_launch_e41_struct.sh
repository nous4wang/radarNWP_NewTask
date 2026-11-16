#!/usr/bin/env bash
# E41 -- structure metrics on the dumped forecast fields (S1, primary criterion).
#
# The metrics come from tools/_e41_struct_metrics.py, which IMPORTS the
# section 2 probe's own implementation rather than copying it, so the four
# structure quantities of S1 are measured with the same code that measured
# the label.  Nothing here trains or evaluates a model.
set -uo pipefail
cd /home/wf/radarNWP || exit 1
export PYTHONPATH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
OUT=/home/wf/radarNWP/runs/e41

"${PY}" tools/_e41_struct_metrics.py \
    --fields "${OUT}/fields" \
    --tags B0_s0_ep00 B0_s1_ep00 B0_s2_ep00 \
           A1_s0_ep00 A1_s1_ep00 A1_s2_ep00 \
           A2_s0_ep00 A2_s1_ep00 A2_s2_ep00 \
    --out "${OUT}/struct_metrics.json"
rc=$?
echo "STRUCT METRICS rc=${rc}"
exit ${rc}
