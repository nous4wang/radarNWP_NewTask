"""Upload the parallel-eval script and launch it detached on the server.

The launch goes through setsid so the evaluation survives the SSH channel
closing.  The script takes its own lock, so a double execution is a
no-op rather than two workers racing on the same checkpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture, upload  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LOCAL = REPO / "scripts" / "run_score_matrix_eval.sh"
REMOTE = "/home/wf/radarNWP/scripts/run_score_matrix_eval.sh"

client = connect()
try:
    upload(client, LOCAL, REMOTE)
    # The script is uploaded as-is; make sure bash sees LF endings.
    capture(client, f"sed -i 's/\\r$//' {REMOTE} && chmod +x {REMOTE} && bash -n {REMOTE} && echo SYNTAX_OK")
    out = capture(
        client,
        "cd /home/wf/radarNWP && setsid nohup bash scripts/run_score_matrix_eval.sh "
        "> /home/wf/radarNWP/runs/score_matrix_eval.log 2>&1 < /dev/null & echo LAUNCHED",
    )
    print(out)
    print(capture(client, "ps -eo pid,pgid,etime,args | grep '[r]un_score_matrix_eval.sh'"))
finally:
    client.close()
