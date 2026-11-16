#!/bin/bash
# E44 -- the two diagnostic read-outs that the S1 verdicts require.
#
# A1: the tail fell while the background did not flatten.  The probe
# replays the trained forward path and compares three compositions of the
# SAME body (as trained / full bypass / E43's form), so it can say
# whether the soft gate's softness shaved the tail or the projection did.
#
# A3: the field is needle-shaped (tail_max_over_p99 200, field maximum
# 306 mm, hf_share 0.58 against the truth's 0.0126).  The probe takes
# real crops around real spikes and evaluates the connectivity term with
# and without the spike, so it can say whether the term is gameable.
#
# The A1 half needs the GPU (it runs the model); the A3 half is numpy on
# the dumped fields.  One job, one GPU.
#
#SBATCH --job-name=e44-diag
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err
set -eo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/wf/radarNWP

python tools/_e44_diag_probe.py \
    --runs /home/wf/radarNWP/runs/e44 \
    --data-root /home/wf/data/radarNWP/BJ_mmap \
    --a1-arm A1 --a1-model lr_sparse_bg \
    --a3-arm A3 \
    --epoch 00 \
    --threshold 2.0 \
    --batch-size 8 --num-workers 4 \
    --scan-windows 80 --max-spikes 25 --spike-mm 20.0 --crop 64 \
    --out /home/wf/radarNWP/runs/e44/diag_probe.json

echo "DIAG DONE $(date -Is)"
