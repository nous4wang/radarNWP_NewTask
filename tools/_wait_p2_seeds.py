"""Wait for the route P2 seed evaluations to finish, then report.

The judge is the file count on the far side: six arms and two epochs
give twelve ``.eval.json`` and twelve ``.windows.json``.  The script
polls until they are all there and then stops -- it changes nothing on
the server.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import capture, connect  # noqa: E402

SEED_DIR = "/home/wf/radarNWP/runs/route_p2/seeds"
ARMS = ("qpe_0", "base_concat")
SEEDS = ("0", "1", "2")
EPOCHS = ("00", "19")
WANT = len(ARMS) * len(SEEDS) * len(EPOCHS)


def count(client) -> tuple[int, int]:
    cmd = (
        f"ls {SEED_DIR}/*.eval.json 2>/dev/null | wc -l; "
        f"ls {SEED_DIR}/*.windows.json 2>/dev/null | wc -l"
    )
    out = capture(client, cmd, timeout=120)
    parts = out.split()
    a = int(parts[0]) if parts else 0
    b = int(parts[1]) if len(parts) > 1 else 0
    return a, b


def main() -> int:
    deadline = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 7200)
    client = connect()
    try:
        while True:
            a, b = count(client)
            print(f"{time.strftime('%H:%M:%S')}  eval.json {a}/{WANT}"
                  f"   windows.json {b}/{WANT}", flush=True)
            if a >= WANT and b >= WANT:
                print("ALL PRESENT", flush=True)
                return 0
            if time.time() > deadline:
                print("TIMEOUT", flush=True)
                return 1
            time.sleep(120)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
