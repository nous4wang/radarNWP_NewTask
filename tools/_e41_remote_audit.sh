#!/usr/bin/env bash
# E41 remote readiness audit -- READ ONLY. Writes nothing on the server.
# Run from the workstation with:
#   python tools/remote.py --script tools/_e41_remote_audit.sh
set -uo pipefail
cd /home/wf/radarNWP || { echo "REPO_MISSING"; exit 1; }
PY=/home/wf/.conda/envs/radar_nwp/bin/python

echo "=== FILES (name size sha256[:16]) ==="
for f in \
  models/struct_hurdle/__init__.py \
  models/struct_hurdle/model.py \
  radar_nwp_bench/metrics/struct_losses.py \
  configs/models/struct_hurdle.yaml \
  scripts/train.py \
  scripts/train_keep_epochs.py \
  scripts/evaluate.py \
  tests/test_struct_hurdle.py \
  tools/_s0_equivalence.py \
  tools/_struct_probe_one_case.py ; do
  if [ -f "$f" ]; then
    printf 'OK      %-56s %10s %s\n' "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-16)"
  else
    printf 'MISSING %s\n' "$f"
  fi
done

echo "=== TRAIN LOSS CHOICES ON REMOTE ==="
grep -n 'hurdle\|uot' scripts/train.py | head -8

echo "=== IMPORT SMOKE ==="
PYTHONPATH=/home/wf/radarNWP "$PY" -c "import torch; from radar_nwp_bench.models import build_model, list_models; from radar_nwp_bench.metrics.struct_losses import HurdleNLL, UnbalancedSinkhornLoss; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available()); print('REGISTERED', 'struct_hurdle' in list_models()); m = build_model('struct_hurdle', head_mode='hurdle'); print('BUILD', type(m).__name__)" 2>&1 | tail -8

echo "=== DATA ==="
ls -d /home/wf/data/radarNWP/BJ_mmap 2>&1
ls /home/wf/data/radarNWP/BJ_mmap 2>/dev/null | head -5

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -10

echo "=== DISK ==="
df -h /home/wf | tail -1

echo "=== CRON ==="
crontab -l 2>/dev/null | head -15

echo "=== TRAIN/EVAL PROCS ALREADY RUNNING ==="
ps -eo pid,etime,cmd | grep -E 'train.py|train_keep_epochs|evaluate.py|xmrig|miner' | grep -v grep | head -20

echo "=== END AUDIT ==="
