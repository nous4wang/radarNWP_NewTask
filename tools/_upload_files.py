"""Upload changed files to the remote by base64.

Git Bash rewrites anything that looks like a path into a Windows path, so a
plain copy over the shell channel mangles it. Base64 keeps the bytes
intact and the payload is split into chunks small enough for the shell to
echo back without truncating.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import capture, connect  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/home/wf/radarNWP"


def upload(rel: str, executable: bool = False) -> None:
    payload = base64.b64encode((REPO / rel).read_bytes()).decode("ascii")
    rp = f"{REMOTE_ROOT}/{rel}"
    tmp = rp + ".tmp_up"
    parts = [payload[i: i + 4000] for i in range(0, len(payload), 4000)]
    sc = [f"rm -f {tmp}", f"touch {tmp}"]
    for p in parts:
        sc.append(f"printf '%s' '{p}' >> {tmp}")
    sc.append(f"base64 -d {tmp} > {rp}.dec && mv {rp}.dec {rp} && rm -f {tmp}")
    if executable:
        sc.append(f"chmod +x {rp}")
    client = connect()
    try:
        out = capture(client, " && ".join(sc), timeout=600)
        if out.strip():
            print(rel, "->", out.strip()[:120])
        size = capture(client, f"wc -c < {rp}", timeout=120).strip()
        local = (REPO / rel).stat().st_size
        print(f"{rel}: local={local} remote={size}")
    finally:
        client.close()


if __name__ == "__main__":
    for rel in sys.argv[1:]:
        upload(rel, executable=rel.endswith(".sh"))
