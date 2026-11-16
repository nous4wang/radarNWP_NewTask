"""Fetch everything ``tools/_analyze_p2.py`` reads, in the layout it expects.

The analyzer reads

    <root>/curve/<arm>_eval/epoch_NN.eval.json
    <root>/seeds/<arm>_s<seed>_ep<epoch>.eval.json
    <root>/seeds/<arm>_s<seed>_ep<epoch>.windows.json
    <root>/../route_p3/<arm>.eval.json
    <root>/../route_p3/<arm>.log
    <root>/../reference/persistence.eval.json

so the files are put exactly there.  Nothing is deleted: a file that is
already local and non-empty is left alone, and a file that is missing
remotely is simply absent.  The judge for the whole route is the file
count on the far side; this only moves the numbers to the local disk so
that the analysis can run without a round trip.

Route P3 needs its training logs as well as its reports: the analyzer
reads the epoch losses out of the logs to decide whether the observation
reached the training, and that check is the one the whole sweep rests on.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect  # noqa: E402

RUNS = "/home/wf/radarNWP/runs"
LOCAL_ROOT = Path(__file__).parent / "_evals"
ARMS = ("qpe_0", "qpe_05", "base_concat")
SEED_ARMS = ("qpe_0", "base_concat")
SEEDS = ("0", "1", "2")
P3_ARMS = ("assim_lag0", "assim_lag2", "assim_lag4", "assim_lag11")

# (remote directory, remote prefix, local directory, local suffix filter)
JOBS: list[tuple[str, str, Path, tuple[str, ...]]] = []
for arm in ARMS:
    JOBS.append(
        (f"{RUNS}/route_p2/curve/{arm}_eval", "epoch_",
         LOCAL_ROOT / "route_p2" / "curve" / f"{arm}_eval", (".eval.json",))
    )
for arm in SEED_ARMS:
    for seed in SEEDS:
        JOBS.append(
            (f"{RUNS}/route_p2/seeds", f"{arm}_s{seed}_ep",
             LOCAL_ROOT / "route_p2" / "seeds", (".eval.json", ".windows.json"))
        )
for arm in P3_ARMS:
    JOBS.append(
        (f"{RUNS}/route_p3", f"{arm}.",
         LOCAL_ROOT / "route_p3", (".eval.json", ".log"))
    )
JOBS.append(
    (f"{RUNS}/reference", "persistence.",
     LOCAL_ROOT / "reference", (".eval.json",))
)


def pull(sftp, remote_dir: str, remote_prefix: str,
         local_dir: Path, suffixes: tuple[str, ...]) -> int:
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
        if not entry.endswith(suffixes):
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
                    for remote_dir, rprefix, local_dir, suffixes in JOBS:
                        got = pull(sftp, remote_dir, rprefix, local_dir, suffixes)
                        total = len(list(local_dir.glob("*")))
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
