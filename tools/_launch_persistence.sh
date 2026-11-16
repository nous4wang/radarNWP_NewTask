#!/usr/bin/env bash
# Produce the persistence reference on a free card, detached.
#
# The reference needs no training and no checkpoint, but it does have to
# be produced by the same code as the models, on the same split, with
# the same verifier and the same stitching -- otherwise it is not
# comparable with them.
set -uo pipefail

cd /home/wf/radarNWP || exit 1
mkdir -p runs/reference

CUDA_VISIBLE_DEVICES="${GPU:-7}" \
/home/wf/.conda/envs/radar_nwp/bin/python tools/_persistence_report.py
