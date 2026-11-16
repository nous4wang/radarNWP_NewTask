"""One-shot remote status probe for the score-aware matrix.

Reads a shell snippet, runs it on the server without a PTY, and writes the
output to a local file so long logs survive the SSH channel.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture  # noqa: E402

SNIPPET = Path(__file__).with_name("_score_status.sh").read_text(encoding="utf-8")

client = connect()
try:
    out = capture(client, SNIPPET, timeout=180)
finally:
    client.close()

dest = Path(__file__).with_name("_score_status.out")
dest.write_text(out, encoding="utf-8")
print(out)
print(f"[written] {dest}")
