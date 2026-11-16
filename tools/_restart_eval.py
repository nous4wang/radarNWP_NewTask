"""Restart the evaluation sweep with the improved scheduler.

The running instance used a two-pass schedule: it evaluated whatever was
ready, then waited for the driver's own phase 2 to finish before it
touched the models that were still training.  The new version polls, so
a model that finishes training late is picked up immediately.

The kill targets the sweep's own process group by pgid.  The training
sweep has its own pgid and is never touched.
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
    # Find the sweep's process group; the driver has a different one.
    listing = capture(
        client,
        "ps -eo pid,pgid,args | grep '[b]ash scripts/run_score_matrix_eval.sh' | awk '{print $1, $2}'",
    )
    print("before:\n" + listing)

    pgids = set()
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            pgids.add(int(parts[1]))

    train_pgids = set()
    for line in capture(
        client, "ps -eo pid,pgid,args | grep '[b]ash scripts/run_score_matrix.sh' | awk '{print $1, $2}'"
    ).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            train_pgids.add(int(parts[1]))

    for pgid in pgids:
        if pgid in train_pgids:
            print(f"  refusing to kill pgid {pgid}: belongs to the training sweep")
            continue
        print(f"  killing eval process group {pgid}")
        capture(client, f"kill -- -{pgid} 2>/dev/null; sleep 2; kill -9 -- -{pgid} 2>/dev/null; echo killed")

    capture(client, "rm -rf /home/wf/radarNWP/runs/score_matrix/.eval_lock")
    capture(client, "sleep 3")

    upload(client, LOCAL, REMOTE)
    print(capture(client, f"sed -i 's/\\r$//' {REMOTE} && chmod +x {REMOTE} && bash -n {REMOTE} && echo SYNTAX_OK"))
    print(capture(
        client,
        "cd /home/wf/radarNWP && setsid nohup bash scripts/run_score_matrix_eval.sh "
        "> /home/wf/radarNWP/runs/score_matrix_eval.log 2>&1 < /dev/null & echo LAUNCHED",
    ))
    print("after:\n" + capture(
        client, "ps -eo pid,pgid,etime,args | grep '[r]un_score_matrix_eval.sh'"
    ))
    print("training sweep untouched:\n" + capture(
        client, "ps -eo pid,pgid,etime,args | grep '[r]un_score_matrix.sh' | awk '{print $1, $2, $3}'"
    ))
finally:
    client.close()
