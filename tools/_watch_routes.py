"""Report the state of the route P2 and P3 runs on the remote.

The judge for every stage is the expected file, not the exit code of the
launcher: a launcher that returns without an error may have died before
it launched anything. This reads the files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import capture, connect  # noqa: E402

ROOT = "/home/wf/radarNWP/runs"


def main() -> None:
    client = connect()
    try:
        cmd = (
            "echo '### GPU'; "
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu "
            "--format=csv,noheader; "
            "echo '### processes'; "
            "ps -eo etime,args | grep -E '[t]rain_keep_epochs|[t]rain.py|"
            "[e]valuate.py' | head -12; "
            f"echo '### P2 curve evals'; "
            f"ls {ROOT}/route_p2/curve/*_eval/*.eval.json 2>/dev/null | wc -l; "
            f"echo '### P2 seed evals'; "
            f"ls {ROOT}/route_p2/seeds/*.eval.json 2>/dev/null | wc -l; "
            f"echo '### P2 seed window tables'; "
            f"ls {ROOT}/route_p2/seeds/*.windows.json 2>/dev/null | wc -l; "
            f"echo '### P3 evals'; "
            f"ls {ROOT}/route_p3/*.eval.json 2>/dev/null | wc -l; "
            f"echo '### curve training progress'; "
            f"for f in {ROOT}/route_p2/curve/*.log; do "
            f"echo \"-- $f\"; grep -E '^epoch' \"$f\" | tail -2; done; "
            f"echo '### seed training progress'; "
            f"for f in {ROOT}/route_p2/seeds/*.log; do "
            f"echo \"-- $f\"; grep -E '^epoch' \"$f\" | tail -1; done; "
            f"echo '### P3 training progress'; "
            f"for f in {ROOT}/route_p3/*.log; do "
            f"echo \"-- $f\"; grep -E '^epoch' \"$f\" | tail -1; done"
        )
        print(capture(client, cmd, timeout=180))
    finally:
        client.close()


if __name__ == "__main__":
    main()
