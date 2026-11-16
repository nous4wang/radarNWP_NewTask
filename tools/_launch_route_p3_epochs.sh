#!/usr/bin/env bash
# Launch the route P3 peak re-measurement on the remote.
#
# Separate from the experiment script, for the reason recorded in
# MASTER.md: a fixed temporary path is overwritten by a second launch,
# and bash reads a script lazily by byte offset, so the first launch
# continues from the middle of the second one and dies silently in the
# middle.  The name here carries the time and the process id.
#
# The four arms take GPUs 0 to 3.  The route P2 seed evaluations take
# GPUs 4 to 7 at the same time, so the two do not compete for a card.
# The evaluations are CPU bound, so they are the part that would
# contend; four training runs and four evaluations is what the machine
# has room for.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p3_epochs}"
LOG="${LOG:-/home/wf/radarNWP/runs/route_p3_epochs_driver.log}"

mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${LOG}")"

STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT="/tmp/_rnb_p3e_${STAMP}_$$.sh"

cat > "${SCRIPT}" <<EOF
#!/usr/bin/env bash
cd "${REPO_ROOT}"
export PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
export OUT_DIR="${OUT_DIR}"
bash scripts/run_route_p3_epochs.sh
echo "P3E_EXIT=\$?"
EOF
chmod +x "${SCRIPT}"

{
    echo "=== $(date '+%F %T') launching the P3 peak re-measurement ==="
    echo "script: ${SCRIPT}"
    echo "out:   ${OUT_DIR}"
} >> "${LOG}"

setsid nohup "${SCRIPT}" >> "${LOG}" 2>&1 < /dev/null &
PID=$!
echo "launched pid ${PID}, script ${SCRIPT}" >> "${LOG}"

sleep 20
if [[ -d "${OUT_DIR}" ]]; then
    echo "OK: the output directory exists" >> "${LOG}"
else
    echo "WARNING: ${OUT_DIR} is not there yet" >> "${LOG}"
fi
echo "launcher returned" >> "${LOG}"
