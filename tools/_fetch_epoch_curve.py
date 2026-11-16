"""Fetch remote files over SFTP, retrying a dropped SSH banner.

The banner is sometimes lost on connect, so the connection is retried.  The
transfer itself goes over SFTP rather than the exec channel: a long log read
as command output can time out the channel, while a file transfer does not.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect  # noqa: E402

# local-relative-path : remote-absolute-path
JOBS = [
    ("tools/_evals/epoch_curve/eval.tar", "/home/wf/radarNWP/runs/epoch_curve/eval"),
    ("tools/_evals/epoch_curve/epoch_curve.log", "/home/wf/radarNWP/runs/epoch_curve.log"),
    ("tools/_evals/epoch_curve/epoch_curve_eval.log", "/home/wf/radarNWP/runs/epoch_curve_eval.log"),
    ("tools/_evals/epoch_curve/epoch_check.log", "/home/wf/radarNWP/runs/epoch_check.log"),
]


def main() -> int:
    for local, remote in JOBS:
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(8):
            try:
                client = connect(retries=3)
                try:
                    sftp = client.open_sftp()
                    try:
                        if remote.endswith("/"):
                            # a directory: pull every file in it
                            import io
                            import tarfile

                            buf = io.BytesIO()
                            with tarfile.open(fileobj=buf, mode="w") as tar:
                                for entry in sftp.listdir(remote):
                                    full = f"{remote}{entry}"
                                    if sftp.stat(full).st_mode & 0o170000 == 0o040000:
                                        continue
                                    data = sftp.open(full, "rb").read()
                                    info = tarfile.TarInfo(name=entry)
                                    info.size = len(data)
                                    tar.addfile(info, io.BytesIO(data))
                            dest.write_bytes(buf.getvalue())
                        else:
                            dest.write_bytes(sftp.open(remote, "rb").read())
                    finally:
                        sftp.close()
                finally:
                    client.close()
                print(f"ok {remote} -> {dest} ({dest.stat().st_size} bytes)")
                last = None
                break
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                print(f"  attempt {attempt + 1} failed: {exc}")
                time.sleep(6 * (attempt + 1))
        if last is not None:
            print(f"FAILED {remote}: {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
