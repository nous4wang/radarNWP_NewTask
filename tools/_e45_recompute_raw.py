#!/usr/bin/env python3
"""E45 round 35 -- recompute the three-seed table from the RAW eval reports.

Why this file exists
--------------------
Every earlier round of verification checked the *shape* of the E45 verdict
(S3 confidence intervals, median + pass rate present, wording).  Nobody
checked the last link in the chain: whether the medians and pass rates that
are printed in MASTER.md and INDEX.md are the medians and pass rates that
actually fall out of the 21 evaluation reports.

This script closes that link.  It is deliberately independent of
``tools/_e45_collect_scores.py``: it re-derives the per-arm table straight
from ``runs/e45/evals/*.eval.json`` with its own parsing, and then compares
that table against the stored ``runs/e45/scores_summary.json``.  If the two
agree, the summary that was written into the documents is backed by the raw
per-window scores; if they disagree, the documents are wrong and the
disagreement is printed instead of being silently accepted.

It trains nothing, evaluates nothing, writes nothing, and touches no
checkpoint.  It only reads JSON.

Usage (on the remote):
    python tools/_e45_recompute_raw.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

FLOOR = 0.0924
ORACLE = 0.1653

EVALS_DIR = "/home/wf/radarNWP/runs/e45/evals"
STORED_SUMMARY = "/home/wf/radarNWP/runs/e45/scores_summary.json"

ARMS = ("B0", "A1", "A2", "A3a", "A3b", "A4", "A4b")
EPOCH = 0

TAG_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)_ep(?P<epoch>\d+)$")


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def recompute_from_raw(evals_dir: str) -> Dict[str, Dict[str, object]]:
    """Per-arm table derived from the raw evaluation reports."""
    per_arm: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    skipped: List[str] = []
    for path in sorted(glob.glob(os.path.join(evals_dir, "*.eval.json"))):
        tag = os.path.basename(path).replace(".eval.json", "")
        m = TAG_RE.match(tag)
        if not m:
            skipped.append(tag)
            continue
        if int(m.group("epoch")) != EPOCH:
            skipped.append(tag)
            continue
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        thresholds = report["thresholds"]
        per_arm[m.group("arm")][int(m.group("seed"))] = {
            "csi_at_2.0": float(thresholds["2.0"]["csi"]),
            "bias_at_2.0": float(thresholds["2.0"]["bias"]),
            "csi_at_4.0": float(thresholds["4.0"]["csi"]),
            "csi_at_0.5": float(thresholds["0.5"]["csi"]),
            "n_samples": int(report.get("n_samples", -1)),
        }

    table: Dict[str, Dict[str, object]] = {}
    for arm in ARMS:
        seeds = per_arm.get(arm)
        if not seeds:
            continue
        got = sorted(seeds)
        csi = [seeds[s]["csi_at_2.0"] for s in got]
        bias = [seeds[s]["bias_at_2.0"] for s in got]
        passed = sum(1 for v in csi if v >= FLOOR)
        table[arm] = {
            "seeds": got,
            "csi_at_2.0": {f"s{s}": seeds[s]["csi_at_2.0"] for s in got},
            "bias_at_2.0": {f"s{s}": seeds[s]["bias_at_2.0"] for s in got},
            "csi_at_4.0": {f"s{s}": seeds[s]["csi_at_4.0"] for s in got},
            "median_csi_at_2.0": _median(csi),
            "median_bias_at_2.0": _median(bias),
            "pass_rate_over_floor": f"{passed}/{len(got)}",
            "n_seeds": len(got),
            "spread_csi_at_2.0": (max(csi) / min(csi)) if csi and min(csi) > 0 else float("nan"),
        }
    return {"table": table, "skipped": skipped}


def compare(raw: Dict[str, Dict[str, object]], stored_path: str) -> List[str]:
    """Compare the recomputed table with the stored summary.  Returns problems."""
    problems: List[str] = []
    if not os.path.exists(stored_path):
        return [f"stored summary not found: {stored_path}"]
    with open(stored_path, encoding="utf-8") as handle:
        stored = json.load(handle)

    stored_arms = set(stored.keys())
    raw_arms = set(raw.keys())
    if stored_arms != raw_arms:
        problems.append(f"arm sets differ: stored-only={sorted(stored_arms - raw_arms)} "
                        f"raw-only={sorted(raw_arms - stored_arms)}")

    for arm in sorted(raw_arms & stored_arms):
        r, s = raw[arm], stored[arm]
        for field in ("median_csi_at_2.0", "median_bias_at_2.0"):
            rv, sv = float(r[field]), float(s.get(field, float("nan")))
            if not (abs(rv - sv) <= 1e-9):
                problems.append(f"{arm}.{field}: recomputed {rv!r} vs stored {sv!r}")
        if r["pass_rate_over_floor"] != s.get("pass_rate_over_floor"):
            problems.append(f"{arm}.pass_rate_over_floor: recomputed "
                            f"{r['pass_rate_over_floor']} vs stored {s.get('pass_rate_over_floor')!r}")
        if list(r["seeds"]) != list(s.get("seeds", [])):
            problems.append(f"{arm}.seeds: recomputed {r['seeds']} vs stored {s.get('seeds')}")
        # the per-seed values themselves
        for seed_key in r["csi_at_2.0"]:
            for field in ("csi_at_2.0", "bias_at_2.0", "csi_at_4.0"):
                rv = float(r[field][seed_key])
                sv = float(s.get(field, {}).get(seed_key, float("nan")))
                if not (abs(rv - sv) <= 1e-9):
                    problems.append(f"{arm}.{field}.{seed_key}: recomputed {rv!r} vs stored {sv!r}")
    return problems


def main() -> int:
    print("=" * 78)
    print("E45 round 35 -- the three-seed table recomputed from the RAW eval reports")
    print(f"persistence floor {FLOOR}   Oracle upper bound {ORACLE}")
    print("=" * 78)

    raw = recompute_from_raw(EVALS_DIR)
    table = raw["table"]
    print(f"\nraw eval files read from {EVALS_DIR}")
    if raw["skipped"]:
        print(f"skipped (wrong epoch or unparsable tag): {raw['skipped']}")

    print()
    print(f"{'arm':>5s} {'seeds':>8s} {'MEDIAN CSI@2.0':>15s} {'MEDIAN BIAS@2.0':>16s} "
          f"{'pass':>6s} {'spread':>8s}")
    for arm in ARMS:
        if arm not in table:
            print(f"{arm:>5s}   -- absent --")
            continue
        row = table[arm]
        spread = row["spread_csi_at_2.0"]
        spread_txt = f"{spread:.2f}x" if spread == spread else "n/a"
        print(f"{arm:>5s} {len(row['seeds']):>8d} {row['median_csi_at_2.0']:>15.6f} "
              f"{row['median_bias_at_2.0']:>16.6f} {row['pass_rate_over_floor']:>6s} {spread_txt:>8s}")
        for seed in row["seeds"]:
            key = f"s{seed}"
            print(f"      {key:>4s} csi2.0={row['csi_at_2.0'][key]:.6f}  "
                  f"bias2.0={row['bias_at_2.0'][key]:.6f}  csi4.0={row['csi_at_4.0'][key]:.6f}")

    print()
    print("-" * 78)
    print("cross-check against the stored summary the documents were written from")
    problems = compare(table, STORED_SUMMARY)
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        print()
        print("VERDICT: FAIL")
        return 1

    print(f"stored summary: {STORED_SUMMARY}")
    print("every arm's median CSI@2.0, median BIAS@2.0, per-seed values and pass "
          "rate agree with the raw reports")
    print()
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
