"""Check the remote score-aware matrix: training progress, eval count, instances."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture  # noqa: E402

MODELS = [
    "simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_early_concat", "simvp_radar_only", "convlstm",
    "predrnn", "earthformer", "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
]

TRAIN = (
    "cd /home/wf/radarNWP && echo '=== TRAIN ===' && for m in "
    + " ".join(MODELS)
    + "; do printf '%-24s ' $m; "
    "grep -E '^epoch [0-9]+:' runs/score_matrix/$m.log 2>/dev/null | tail -1 || echo '(not started)'; "
    "done"
)

EVAL = (
    "cd /home/wf/radarNWP && echo '=== EVAL FILES ===' && "
    "ls -la runs/score_matrix/*.eval.json 2>/dev/null | awk '{print $5, $9}'"
)

INST = (
    "cd /home/wf/radarNWP && echo '=== INSTANCES ===' && "
    "ps -eo pid,pgid,etime,args | grep '[r]un_score_matrix' | awk '{print $1, $2, $3}' ; "
    "echo '=== DRIVER PIDS ===' ; "
    "ps -eo pid,pgid,etime,args | grep '[r]un_score_matrix.sh' | awk '{print $1, $2, $3, $4}'"
)

GPU = (
    "cd /home/wf/radarNWP && echo '=== GPU ===' && "
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\\n' ' ' ; echo ; "
    "echo '=== DISK ===' ; df -h /home/wf | tail -1"
)

LOG = "cd /home/wf/radarNWP && echo '=== DRIVER LOG ===' && tail -40 runs/score_matrix_driver.log 2>/dev/null"

client = connect()
try:
    for label, cmd in (("TRAIN", TRAIN), ("EVAL", EVAL), ("INST", INST), ("GPU", GPU), ("LOG", LOG)):
        try:
            out = capture(client, cmd, timeout=180)
        except Exception as exc:  # noqa: BLE001
            out = f"<ERROR {exc}>"
        print(f"----- {label} -----")
        print(out)
finally:
    client.close()
