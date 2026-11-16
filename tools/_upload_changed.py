"""Upload the changed files to the remote, and verify them there.

The transport is base64 over the shell channel rather than an SFTP put,
because the Git Bash path conversion rewrites anything that looks like a
POSIX path before it reaches the shell. Each file is written to a
temporary name and moved into place, so a half-written file never sits
at the path the training run reads.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import capture, connect  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/home/wf/radarNWP"

FILES = [
    "scripts/train.py",
    "scripts/evaluate.py",
    "radar_nwp_bench/data/dataset.py",
    "radar_nwp_bench/metrics/precipitation.py",
]


def main() -> None:
    client = connect()
    try:
        for rel in FILES:
            local = REPO / rel
            payload = base64.b64encode(local.read_bytes()).decode("ascii")
            remote_path = f"{REMOTE_ROOT}/{rel}"
            tmp = f"{remote_path}.upload_tmp"
            # base64 in chunks: a single 200 KB line can exceed the
            # command length the shell accepts.
            chunk = 4000
            pieces = [payload[i:i + chunk] for i in range(0, len(payload), chunk)]
            script = [f"rm -f {tmp}", f"touch {tmp}"]
            for piece in pieces:
                script.append(f"printf '%s' '{piece}' >> {tmp}")
            script.append(
                f"base64 -d {tmp} > {remote_path}.decoded && "
                f"mv {remote_path}.decoded {remote_path} && rm -f {tmp}"
            )
            out = capture(client, " && ".join(script), timeout=300)
            if out.strip():
                print(f"{rel}: {out.strip()[:200]}")

            # Verify: the byte count and a marker must match.
            local_size = local.stat().st_size
            check = capture(
                client,
                f"wc -c < {remote_path}",
                timeout=60,
            ).strip()
            marker = capture(
                client,
                f"grep -c 'CUBLAS_WORKSPACE_CONFIG' {remote_path} "
                f"|| true",
                timeout=60,
            ).strip()
            print(f"{rel}: local={local_size} remote={check} marker={marker}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
