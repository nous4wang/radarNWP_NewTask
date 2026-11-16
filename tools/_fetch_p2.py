"""Fetch the route P2 evaluations into the layout _analyze_p2.py expects.

The analyzer reads

    <root>/curve/<arm>_eval/epoch_NN.eval.json
    <root>/seeds/<arm>_s<seed>.eval.json
    <root>/seeds/<arm>_s<seed>.windows.json

so the files are put exactly there, and the curve directory is refreshed
on every run while the seed directory only takes what is new. Nothing is
deleted: a file that is already local is left alone, and a file that is
missing remotely is simply absent.

The judge for the whole route is the file count on the far side; this
only moves the numbers to the local disk so that the analysis can run
without a round trip.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect  # noqa: E402

ROOT = "/home/wf/radarNWP/runs/route_p2"
LOCAL_ROOT = Path(__file__).parent / "_evals" / "route_p2"
ARMS = ("qpe_0", "qpe_05", "base_concat")
SEED_ARMS = ("qpe_0", "base_concat")
SEEDS = ("0", "1", "2")

# (remote directory, remote prefix, local directory, local prefix)
JOBS: list[tuple[str, str, Path, str]] = []
for arm in ARMS:
    JOBS.append(
        (f"{ROOT}/curve/{arm}_eval", "epoch_",
         LOCAL_ROOT / "curve" / f"{arm}_eval", "epoch_")
    )
for arm in SEED_ARMS:
    for seed in SEEDS:
        JOBS.append(
            (f"{ROOT}/seeds", f"{arm}_s{seed}_ep",
             LOCAL_ROOT / "seeds", f"{arm}_s{seed}_ep")
        )


def pull(sftp, remote_dir: str, remote_prefix: str,
         local_dir: Path, local_prefix: str) -> int:
    """Copy every matching file from one remote directory."""
    try:
        entries = sftp.listdir(remote_dir)
    except OSError:
        return 0
    local_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in sorted(entries):
        if not entry.startswith(remote_prefix):
            continue
        # The seed directory holds the logs and the run directories too;
        # only the two report files are wanted.
        if not entry.endswith((".eval.json", ".windows.json")):
            continue
        dest = local_dir / entry
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            data = sftp.open(f"{remote_dir}/{entry}", "rb").read()
        except OSError as exc:
            print(f"  skip {entry}: {exc}")
            continue
        dest.write_bytes(data)
        n += 1
    return n


def main() -> int:
    last = None
    for attempt in range(8):
        try:
            client = connect(retries=3)
            try:
                sftp = client.open_sftp()
                try:
                    for remote_dir, rprefix, local_dir, lprefix in JOBS:
                        got = pull(sftp, remote_dir, rprefix, local_dir, lprefix)
                        total = len(list(local_dir.glob(lprefix + "*")))
                        print(f"{remote_dir}: {got} new, {total} local")
                finally:
                    sftp.close()
            finally:
                client.close()
            return 0
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            print(f"attempt {attempt + 1} failed: {exc}")
            time.sleep(6 * (attempt + 1))
    print(f"FAILED: {last}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
