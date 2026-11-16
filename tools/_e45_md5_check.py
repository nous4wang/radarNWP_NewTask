"""E45 round-20: remote vs local md5 cross-check.

Reads remote md5sum output straight from tools/remote.py stdout (no temp file on
/tmp -- Git Bash /tmp is invisible to the Windows python build) and compares it
with the local tools/_evals/e45/ directory.

VERDICT: PASS  => every remote json has a local twin with identical md5.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(REPO, "tools", "_evals", "e45")
REMOTE_CMD = "cd /home/wf/radarNWP && md5sum runs/e45/*.json"

fails: list[str] = []
notes: list[str] = []


def remote_md5() -> dict[str, str]:
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "remote.py"), "--", REMOTE_CMD],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"remote.py failed rc={proc.returncode}: {proc.stderr[-800:]}")
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 32:
            out[os.path.basename(parts[1])] = parts[0]
    return out


def local_md5() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in sorted(os.listdir(LOCAL_DIR)):
        if not name.endswith(".json"):
            continue
        h = hashlib.md5()
        with open(os.path.join(LOCAL_DIR, name), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
    return out


def main() -> int:
    rem = remote_md5()
    loc = local_md5()
    notes.append(f"remote json count = {len(rem)}, local json count = {len(loc)}")
    if not rem:
        fails.append("remote md5 extraction produced 0 records (extraction failed, not 'no data')")
    if not loc:
        fails.append("local md5 extraction produced 0 records (extraction failed, not 'no data')")

    for name in sorted(rem):
        if name not in loc:
            fails.append(f"remote-only (not downloaded): {name}")
            continue
        if rem[name] != loc[name]:
            fails.append(f"md5 mismatch: {name} remote={rem[name]} local={loc[name]}")
        else:
            notes.append(f"OK  {name}  {rem[name]}")
    for name in sorted(loc):
        if name not in rem:
            notes.append(f"LOCAL-ONLY (not on remote listing): {name}")

    # every local file must be matched by exactly one remote record
    matched = sum(1 for n in rem if n in loc and rem[n] == loc[n])
    notes.append(f"matched pairs = {matched}")
    if rem and matched != len(rem):
        fails.append(f"matched pairs {matched} != remote records {len(rem)}")

    print("\n".join(notes))
    if fails:
        print("FAILURES:")
        print("\n".join("  - " + f for f in fails))
        print("VERDICT: FAIL")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
