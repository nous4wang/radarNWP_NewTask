#!/usr/bin/env python3
"""Read the bias/CSI survey and print the arms that matter.

Separate tool so that the analysis runs on the server without any
quoting hazard: a f-string cannot contain a backslash, and a heredoc
through SSH is exactly where that bites.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    d = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    rows = d["rows"]

    print("=== the highest CSI over the whole history ===")
    for r in sorted(rows, key=lambda x: -x["csi"])[:args.top]:
        print(f"  {r['tag']:46s} CSI {r['csi']:9.6f}  BIAS {r['bias']:8.4f}"
              f"  n={r['n_samples']}")

    print()
    print("=== every base_concat / qpe_0 row ===")
    for r in rows:
        tag = r["tag"]
        if "base_concat" in tag or "qpe_0" in tag:
            print(f"  {tag:46s} CSI {r['csi']:9.6f}  BIAS {r['bias']:8.4f}")

    print()
    print("=== the recent batches (E41..E44) ===")
    for r in rows:
        tag = r["tag"]
        if tag.startswith(("A1_s", "A2_s", "A3_s", "A4_s", "B0_s", "A2p_s")):
            print(f"  {tag:46s} CSI {r['csi']:9.6f}  BIAS {r['bias']:8.4f}")

    print()
    print("=== CSI >= 0.0924 (above the persistence floor) ===")
    above = [r for r in rows if r["csi"] >= 0.0924]
    print(f"  count: {len(above)} of {len(rows)}")
    for r in sorted(above, key=lambda x: -x["csi"]):
        print(f"  {r['tag']:46s} CSI {r['csi']:9.6f}  BIAS {r['bias']:8.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
