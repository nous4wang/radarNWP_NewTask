#!/bin/bash
#SBATCH --job-name=e43-probe
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err
set -eo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "PROBE job=${SLURM_JOB_ID} host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.ones(1024, device="cuda:0")
    print("alloc_ok", float((x @ x).sum()))
PY
