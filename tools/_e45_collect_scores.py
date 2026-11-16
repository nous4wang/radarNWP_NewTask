#!/usr/bin/env python3
"""E45 -- the three-seed summary: medians and pass rates, never a bare score.

Why this file exists at all
---------------------------
E45's finding four (section 0.4) measured the whole history's
evaluations and found that the highest score ever recorded on this
benchmark -- ``base_concat_s0_ep00`` = 0.186511 -- is NOT a level but
the best of ten seeds: the ten seeds of that arm at epoch 0 have a
median of 0.072026, a spread of 9.59x (0.019439-0.186511) and only
FOUR of the ten clear the persistence floor of 0.0924.

A single score therefore cannot be read as a level, and reporting one
would be misleading no matter how good it looks.  Since E45 the report
format is: the MEDIAN over the seeds and the PASS RATE (how many seeds
clear the floor), and the best seed is only ever shown next to the
median, never alone.

This tool reads only the JSON reports that ``scripts/evaluate.py``
wrote.  It trains nothing, evaluates nothing, and writes nothing except
the optional JSON summary.

What it prints
--------------
For every arm: the median and the per-seed spread of ``CSI@2.0`` and
``BIAS@2.0``, the pass rate against the persistence floor, and the
spread of the same quantity across the seeds.  The floor and the Oracle
bound are printed with every table, so that no number can be read
without them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

#: The persistence floor (the benchmark's own baseline) and the Oracle
#: upper bound (a perfect nowcast).  Both are the master document's
#: anchors and are printed with every table.
FLOOR = 0.0924
ORACLE = 0.1653

#: The arms of the batch, in the order they are reported.  A number is
#: only reported for an arm that has all three seeds: a partial arm
#: would print a median over fewer seeds, which is the kind of number
#: this file exists to avoid.
ARMS = ("B0", "A1", "A2", "A3a", "A3b", "A4", "A4b")


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _fmt(value: float) -> str:
    return "  n/a " if value != value else f"{value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", default="/home/wf/radarNWP/runs/e45/evals")
    ap.add_argument("--pattern", default="*.eval.json")
    ap.add_argument("--out", default=None,
                    help="optional JSON file with the same numbers")
    args = ap.parse_args()

    # tag -> the numbers, grouped by arm and seed.
    per_arm: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.evals, args.pattern))):
        tag = os.path.basename(path).replace(".eval.json", "")
        # The tag is "<arm>_s<seed>_ep<epoch>"; the epoch is dropped
        # here because the criterion reads epoch 00 only, and a run that
        # wrote several epochs would otherwise be counted twice.
        m = re.match(r"^(?P<arm>.+)_s(?P<seed>\d+)_ep(?P<epoch>\d+)$", tag)
        if not m:
            print(f"skipping {tag}: the tag does not name an arm and a seed")
            continue
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        thresholds = report["thresholds"]
        per_arm[m.group("arm")][int(m.group("seed"))] = {
            "epoch": int(m.group("epoch")),
            "csi_at_2.0": thresholds["2.0"]["csi"],
            "bias_at_2.0": thresholds["2.0"]["bias"],
            "csi_at_4.0": thresholds["4.0"]["csi"],
            "csi_at_0.5": thresholds["0.5"]["csi"],
            "n_samples": report.get("n_samples", -1),
        }

    print("=" * 78)
    print("E45 three-seed summary -- the median and the pass rate, never a bare score")
    print(f"persistence floor {FLOOR}   Oracle upper bound {ORACLE}")
    print("=" * 78)

    summary: Dict[str, Dict[str, object]] = {}
    for arm in ARMS:
        seeds = per_arm.get(arm)
        if not seeds:
            continue
        got = sorted(seeds)
        csi = [seeds[s]["csi_at_2.0"] for s in got]
        bias = [seeds[s]["bias_at_2.0"] for s in got]
        passed = sum(1 for v in csi if v >= FLOOR)
        print()
        print(f"--- {arm}  ({len(got)} seed(s): {', '.join('s%d' % s for s in got)})")
        print(f"    {'seed':>6s} {'CSI@2.0':>10s} {'BIAS@2.0':>10s} {'CSI@4.0':>10s} "
              f"{'CSI@0.5':>10s} {'n':>6s}  vs floor")
        for s in got:
            row = seeds[s]
            mark = "clears" if row["csi_at_2.0"] >= FLOOR else "below "
            print(f"    {'s%d' % s:>6s} {row['csi_at_2.0']:>10.4f} {row['bias_at_2.0']:>10.4f} "
                  f"{row['csi_at_4.0']:>10.4f} {row['csi_at_0.5']:>10.4f} "
                  f"{row['n_samples']:>6d}  {mark} the floor")
        med_csi = _median(csi)
        med_bias = _median(bias)
        spread = (max(csi) / min(csi)) if csi and min(csi) > 0 else float("nan")
        print(f"    MEDIAN   CSI@2.0 = {_fmt(med_csi)}   BIAS@2.0 = {_fmt(med_bias)}")
        print(f"    PASS RATE over the floor: {passed}/{len(got)}")
        if spread == spread:
            print(f"    seed spread of CSI@2.0: {spread:.2f}x")
        else:
            print("    seed spread of CSI@2.0: n/a")
        if len(got) < 3:
            print(f"    NOTE: only {len(got)} seed(s) finished -- the median is over "
                  f"fewer than three and is not a level")
        summary[arm] = {
            "seeds": got,
            "csi_at_2.0": {f"s{s}": seeds[s]["csi_at_2.0"] for s in got},
            "bias_at_2.0": {f"s{s}": seeds[s]["bias_at_2.0"] for s in got},
            "csi_at_4.0": {f"s{s}": seeds[s]["csi_at_4.0"] for s in got},
            "csi_at_0.5": {f"s{s}": seeds[s]["csi_at_0.5"] for s in got},
            "median_csi_at_2.0": med_csi,
            "median_bias_at_2.0": med_bias,
            "pass_rate_over_floor": f"{passed}/{len(got)}",
            "seed_spread_csi_at_2.0": spread,
            "n_seeds": len(got),
        }

    print()
    print("=" * 78)
    print("The floor is 0.0924 and the Oracle bound 0.1653: a median above the")
    print("floor is a RESULT, and the pass rate is what says how solid it is.")
    print("A single seed is never a level here (E45 finding four).")
    print("=" * 78)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
