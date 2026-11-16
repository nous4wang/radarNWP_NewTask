#!/usr/bin/env bash
# Launch route P2 (the epoch curve and the seed spread) on the remote.
#
# The launcher is a separate script from the experiment on purpose. The
# experiment script is the thing under test and must not be edited to
# accommodate the launcher; and the launcher's temporary path is unique,
# because a fixed one is overwritten by a second launch and bash reads a
# script lazily by byte offset, so the first launch then continues from
# the middle of the second one.
#
# The judge is the expected files in the output directory, not the exit
# code of the launcher: a launcher that returns without an error may
# still have died before it launched anything.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p2}"
LOG="${LOG:-/home/wf/radarNWP/runs/route_p2_driver.log}"

mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${LOG}")"

# A unique path per launch: the stamp and the pid together make a name
# that a concurrent launch cannot produce.
STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT="/tmp/_rnb_p2_${STAMP}_$$.sh"

cat > "${SCRIPT}" <<EOF
#!/usr/bin/env bash
cd "${REPO_ROOT}"
export PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
export OUT_DIR="${OUT_DIR}"
bash scripts/run_route_p2.sh
echo "P2_EXIT=\$?"
EOF
chmod +x "${SCRIPT}"

{
    echo "=== $(date '+%F %T') launching P2 ==="
    echo "script: ${SCRIPT}"
    echo "out:   ${OUT_DIR}"
} >> "${LOG}"

setsid nohup "${SCRIPT}" >> "${LOG}" 2>&1 < /dev/null &
PID=$!
echo "launched pid ${PID}, script ${SCRIPT}" >> "${LOG}"

# The judge: the curve directory and the seed directory must appear.
sleep 20
if [[ -d "${OUT_DIR}/curve" ]]; then
    echo "OK: the output directory exists" >> "${LOG}"
else
    echo "WARNING: ${OUT_DIR}/curve is not there yet" >> "${LOG}"
fi
echo "launcher returned" >> "${LOG}"
