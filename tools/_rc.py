"""Local helper: run a remote command via tools/remote.py with retries.

Usage:  /c/Users/nous/anaconda3/python.exe tools/_rc.py "<shell command>"
"""
import sys
import time

sys.path.insert(0, "tools")
from remote import connect, capture  # noqa: E402

cmd = sys.argv[1]
last = None
for attempt in range(6):
    try:
        c = connect()
        try:
            out = capture(c, cmd)
        finally:
            c.close()
        sys.stdout.write(out)
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last = exc
        time.sleep(3 + attempt * 2)
sys.stderr.write(f"FAILED after retries: {last}\n")
sys.exit(1)
