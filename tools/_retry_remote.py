"""Patient SSH retry: the server may have a temporary auth lockout."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture  # noqa: E402

CMD = (
    "cd /home/wf/radarNWP && echo '=== TRAIN ===' && for m in "
    "simvp_v2 simvp_v2_nwp_concat simvp_v2_quantum simvp_v2_nwp_xattn "
    "simvp_v2_nwp_delta simvp_early_concat simvp_radar_only convlstm "
    "predrnn earthformer metnet3 diffcast cascast_nwp mocast pw_foucast "
    "fusioncast mmf_rnn furecast; do printf '%-24s ' $m; "
    "grep -cE '^epoch 19:' runs/score_matrix/$m.log 2>/dev/null || echo 0; done; "
    "echo '=== EVAL ==='; ls runs/score_matrix/*.eval.json 2>/dev/null | wc -l; "
    "echo '=== INST ==='; ps -eo pid,pgid,etime,args | grep '[r]un_score_matrix' "
    "| awk '{print $1, $2, $3}'; "
    "echo '=== GPU ==='; nvidia-smi --query-gpu=index,memory.used "
    "--format=csv,noheader | tr '\\n' ' '; echo; "
    "echo '=== SUMMARY ==='; sed -n '/=== the summary ===/,$p' "
    "runs/score_matrix_driver.log 2>/dev/null | head -60"
)

for attempt in range(1, 31):
    try:
        client = connect(retries=2)
        try:
            out = capture(client, CMD, timeout=300)
            print(f"=== SUCCESS on attempt {attempt} ===")
            print(out)
            sys.exit(0)
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001
        print(f"attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
    time.sleep(60)

print("=== GAVE UP after 30 attempts ===")
sys.exit(1)
