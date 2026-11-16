#!/usr/bin/env bash
# Launch the route experiments on the server, detached from the session.
#
# The launcher is a separate file because the transport writes the
# script it is given to the far side and runs it in its own session;
# a command typed into an interactive shell dies with the connection.
set -uo pipefail

cd /home/wf/radarNWP || exit 1

rm -rf runs/route_r1r2 runs/route_baseline
mkdir -p runs/route_r1r2 runs/route_baseline

export PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
export DATA_ROOT=/home/wf/data/radarNWP/BJ_mmap
export EPOCHS=20
export PATCH_SIZE=64
export SLIDE_STRIDE=64
export BATCH_SIZE=16
export WORKERS=6
export TIERS="0.1 0.5 2.0 4.0"
export ALPHA_CSI=0.5
export QPE_WEIGHT=0.5

# The route matrix takes the first seven cards, one arm each. The
# baseline re-run takes the last one and runs its two models one after
# the other, because it is a confirmation rather than the comparison:
# the exact null check already showed the data layer moved nothing.
export GPUS="0 1 2 3 4 5 6"
bash scripts/run_route_r1r2.sh > runs/route_r1r2/driver.log 2>&1

export GPUS_0=7
export GPUS_1=7
bash scripts/run_route_baseline.sh > runs/route_baseline/driver.log 2>&1

echo "ALL_DONE"
