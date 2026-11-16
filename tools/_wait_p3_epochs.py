"""Wait for the route P3 peak re-measurement, then report.

The judge is the file count: four arms and two epochs give eight
``.eval.json`` files under ``runs/route_p3_epochs``.  The script polls
until they are all there and then stops -- it changes nothing on the
server.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import capture, connect  # noqa: E402

OUT = "/home/wf/radarNWP/runs/route_p3_epochs"
ARMS = ("assim_lag0", "assim_lag2", "assim_lag4", "assim_lag11")
EPOCHS = ("00", "19")
WANT = len(ARMS) * len(EPOCHS)


def main() -> int:
    deadline = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 10800)
    client = connect()
    try:
        while True:
            out = capture(
                client,
                f"ls {OUT}/*.eval.json 2>/dev/null | wc -l; "
                f"grep -c '^epoch 19' {OUT}/*.log 2>/dev/null | "
                f"grep -c ':1' || true",
                timeout=120,
            )
            parts = out.split()
            n = int(parts[0]) if parts else 0
            print(f"{time.strftime('%H:%M:%S')}  eval.json {n}/{WANT}", flush=True)
            if n >= WANT:
                print("ALL PRESENT", flush=True)
                return 0
            if time.time() > deadline:
                print("TIMEOUT", flush=True)
                return 1
            time.sleep(180)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
