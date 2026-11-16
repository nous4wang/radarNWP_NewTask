#!/usr/bin/env bash
# The baseline arms on their own, detached.
#
# On 2026-11-02 the route launcher was killed between the two halves:
# it had written its script to a fixed path, a later launch overwrote
# that path, and bash -- which reads a script by byte offset -- resumed
# in the middle of the new file. The seven route arms survived because
# they had already been forked; the baseline arms, which start after the
# route arms, never began. This file is that second half, on its own.
#
# The experiment script itself is not touched: it is the same
# scripts/run_route_baseline.sh that the route launcher would have run.
# Only the card assignment differs -- the route launcher put both
# baselines on the last card because the seven route arms had taken the
# other seven, and every card is free now.
set -uo pipefail

cd /home/wf/radarNWP || exit 1
mkdir -p runs/route_baseline

export PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
export DATA_ROOT=/home/wf/data/radarNWP/BJ_mmap
export EPOCHS=20
export PATCH_SIZE=64
export SLIDE_STRIDE=64
export BATCH_SIZE=16
export WORKERS=6
export TIERS="0.1 0.5 2.0 4.0"
export ALPHA_CSI=0.5
export GPUS_0=0
export GPUS_1=1

bash scripts/run_route_baseline.sh
