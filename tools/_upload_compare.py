"""Upload a local file to the server (no double execution of anything)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture, upload  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

FILES = [
    (REPO / "scripts" / "compare_score_matrix.py", "/home/wf/radarNWP/scripts/compare_score_matrix.py"),
]

client = connect()
try:
    for local, remote in FILES:
        upload(client, local, remote)
    print(capture(client, "ls -la /home/wf/radarNWP/scripts/compare_score_matrix.py"))
finally:
    client.close()
