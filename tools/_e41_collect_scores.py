"""E41 -- collect the evaluation scores of the finished runs (read-only).

Prints one row per scored run: the verification scores at the thresholds
that the E41 criteria read (CSI and BIAS at 2.0 mm, CSI at 4.0 mm and
0.5 mm, plus POD and FAR at 2.0 mm).  The reference lines are the
master document's anchors: the persistence floor is 0.0924 and the Oracle
upper bound is 0.1653.

It reads only the JSON files that ``scripts/evaluate.py`` wrote.  It
trains nothing, evaluates nothing and writes nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

FLOOR = 0.0924
ORACLE = 0.1653


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", default="/home/wf/radarNWP/runs/e41/evals")
    ap.add_argument("--pattern", default="*.eval.json")
    ap.add_argument("--out", default=None,
                    help="optional JSON file with the same numbers")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.evals, args.pattern))):
        d = json.load(open(p, encoding="utf-8"))
        t = d["thresholds"]

        def g(k, f):
            return t[k][f]

        tag = os.path.basename(p).replace(".eval.json", "")
        rows.append((
            tag,
            g("2.0", "csi"), g("2.0", "bias"), g("4.0", "csi"),
            g("0.5", "csi"), g("2.0", "pod"), g("2.0", "far"),
            d.get("n_samples", -1),
            g("0.5", "bias"), g("4.0", "bias"),
        ))

    hdr = ("tag", "CSI@2.0", "BIAS@2.0", "CSI@4.0", "CSI@0.5",
           "POD@2.0", "FAR@2.0", "n", "BIAS@0.5", "BIAS@4.0")
    print("{:14s} {:>9s} {:>9s} {:>9s} {:>9s} {:>8s} {:>8s} {:>6s} {:>9s} {:>9s}".format(*hdr))
    for r in rows:
        print("{:14s} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:8.4f} {:8.4f} {:6d} {:9.4f} {:9.4f}".format(*r))
    print()
    print(f"anchors: persistence floor {FLOOR}, Oracle upper bound {ORACLE}")
    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"highest CSI@2.0 in this batch: {best[0]} = {best[1]:.4f}")

    if args.out:
        out = {}
        for r in rows:
            out[r[0]] = {
                "csi_at_2.0": r[1], "bias_at_2.0": r[2],
                "csi_at_4.0": r[3], "csi_at_0.5": r[4],
                "pod_at_2.0": r[5], "far_at_2.0": r[6],
                "n_samples": r[7], "bias_at_0.5": r[8], "bias_at_4.0": r[9],
            }
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"scores written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
