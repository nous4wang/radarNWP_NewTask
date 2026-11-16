#!/usr/bin/env bash
# Launch route P3 (the observation-assimilation lag sweep) on the remote.
#
# Separate from the P2 launcher and separate from the experiment script,
# for the reason recorded in MASTER.md: a fixed temporary path is
# overwritten by a second launch, and bash reads a script lazily by byte
# offset, so the first launch continues from the middle of the second one
# and dies silently in the middle.
#
# P3 runs on four cards. P2's curve part uses three and its seed part
# three at a time, so the two overlap badly if started together. They are
# therefore launched at different times: P2 now, P3 when the curve is done.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWP/runs/route_p3}"
LOG="${LOG:-/home/wf/radarNWP/runs/route_p3_driver.log}"

mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${LOG}")"

STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT="/tmp/_rnb_p3_${STAMP}_$$.sh"

cat > "${SCRIPT}" <<EOF
#!/usr/bin/env bash
cd "${REPO_ROOT}"
export PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
export OUT_DIR="${OUT_DIR}"
bash scripts/run_route_p3.sh
echo "P3_EXIT=\$?"
EOF
chmod +x "${SCRIPT}"

{
    echo "=== $(date '+%F %T') launching P3 ==="
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
