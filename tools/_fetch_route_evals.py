"""Download the route arms (R1/R2 + retrained baselines) for local analysis."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, capture  # noqa: E402

ARMS = {
    "route_r1r2": [
        "base_concat", "qpe_0", "qpe_05",
        "assim_lag11", "assim_lag4", "assim_lag2", "assim_lag0",
    ],
    "route_baseline": ["simvp_v2", "simvp_v2_nwp_concat"],
    "reference": ["persistence"],
}

client = connect()
try:
    for run, arms in ARMS.items():
        for arm in arms:
            out = capture(
                client,
                f"cat /home/wf/radarNWP/runs/{run}/{arm}.eval.json",
                timeout=120,
            )
            if out.strip().startswith("{"):
                p = Path(__file__).parent / "_evals" / run
                p.mkdir(parents=True, exist_ok=True)
                (p / f"{arm}.eval.json").write_text(out, encoding="utf-8")
                print(f"ok {run}/{arm}")
            else:
                print(f"MISS {run}/{arm}")
finally:
    client.close()
