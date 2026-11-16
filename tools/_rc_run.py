"""One-shot helper: run a shell command on the radar benchmark server.

    python tools/_rc_run.py "<shell command>"

Kept tiny on purpose: every call opens a fresh SSH session (the server drops
banners often enough that reuse is not worth it) and prints the remote stdout.
"""
from __future__ import annotations

import sys
import time

from remote import capture, connect


def main() -> int:
    cmd = sys.argv[1]
    for attempt in range(5):
        try:
            client = connect()
            try:
                out = capture(client, cmd)
            finally:
                client.close()
            sys.stdout.write(out)
            return 0
        except Exception as exc:  # noqa: BLE001 - banner drops are expected
            print(f"[retry {attempt + 1}] {exc}", file=sys.stderr)
            time.sleep(5)
    print("[failed] could not run remote command", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
