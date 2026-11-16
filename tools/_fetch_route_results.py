"""Download the route P2 and P3 results for local analysis."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import capture, connect  # noqa: E402

ROOT = "/home/wf/radarNWP/runs"
LOCAL = Path(__file__).resolve().parent / "_evals"

JOBS: list[tuple[str, str]] = []
for arm in ("qpe_0", "qpe_05", "base_concat"):
    for ep in range(20):
        JOBS.append(
            (f"route_p2/curve/{arm}_eval/epoch_{ep:02d}.eval.json",
             f"route_p2/curve/{arm}_eval/epoch_{ep:02d}.eval.json")
        )
for arm in ("qpe_0", "base_concat"):
    for seed in (0, 1, 2):
        for ep in ("00", "19"):
            JOBS.append(
                (f"route_p2/seeds/{arm}_s{seed}_ep{ep}.eval.json",
                 f"route_p2/seeds/{arm}_s{seed}_ep{ep}.eval.json")
            )
            JOBS.append(
                (f"route_p2/seeds/{arm}_s{seed}_ep{ep}.windows.json",
                 f"route_p2/seeds/{arm}_s{seed}_ep{ep}.windows.json")
            )
for lag in (0, 2, 4, 11):
    JOBS.append(
        (f"route_p3/assim_lag{lag}.eval.json",
         f"route_p3/assim_lag{lag}.eval.json")
    )


def main() -> None:
    client = connect()
    ok = miss = 0
    try:
        for remote, local in JOBS:
            out = capture(client, f"cat {ROOT}/{remote} 2>/dev/null", timeout=120)
            if out.strip().startswith(("{", "[")):
                p = LOCAL / local
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(out, encoding="utf-8")
                ok += 1
            else:
                miss += 1
                print(f"MISS {remote}")
    finally:
        client.close()
    print(f"downloaded {ok}, missing {miss}")


if __name__ == "__main__":
    main()
