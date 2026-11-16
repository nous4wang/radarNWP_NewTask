"""Run a remote command and save the output locally (long output safe)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture  # noqa: E402

cmd = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None

client = connect()
try:
    text = capture(client, cmd)
finally:
    client.close()

if out:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[written] {out}  ({len(text)} chars)")
else:
    print(text)
