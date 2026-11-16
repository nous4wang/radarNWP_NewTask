#!/usr/bin/env python
"""E45 round-30 remote read-only probe.

Dumps the raw contingency tables of all runs/e45/evals/*.eval.json so that the
local verifier can recompute CSI/POD/FAR/BIAS from hits/misses/false_alarms and
tie the 21 remote eval files back to the locally aggregated scores_summary.

Read-only: only opens files for reading and prints one JSON blob to stdout.
"""
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "runs", "e45", "evals")


def main():
    paths = sorted(glob.glob(os.path.join(EVAL_DIR, "*.eval.json")))
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            raw = fh.read()
        d = json.loads(raw)
        rows.append(
            {
                "file": os.path.basename(p),
                "md5": hashlib.md5(raw.encode("utf-8")).hexdigest(),
                "n_samples": d.get("n_samples"),
                "top_keys": sorted(d.keys()),
                "thresholds": d.get("thresholds"),
            }
        )
    out = {"eval_dir": EVAL_DIR, "n_files": len(rows), "rows": rows}
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
