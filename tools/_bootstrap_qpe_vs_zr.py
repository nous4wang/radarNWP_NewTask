#!/usr/bin/env python3
"""Paired bootstrap: a neural QPE arm vs the analytical Z-R persistence arm.

Both arms must already have been scored on the SAME 538 test windows with
``--per-window-out`` (scripts/evaluate.py and tools/_scan_zr_k.py write the
same JSON shape).  The resampling unit is the evaluation window, not the
pixel: CSI is a ratio of accumulated counts, so windows are the independent
units.  The same window indices are drawn for both arms, which is what makes
the comparison paired -- the difference is computed on the same weather.

If several seeds are supplied per arm (``--a`` / ``--b`` each take one or
more paths, paired by position) the seed is resampled as a second level,
exactly as in tools/_bootstrap_seeds_final.py.  With one seed the interval
measures evaluation-window sampling uncertainty only; it does NOT measure
training-seed variance.  §4.18.3 S2-E and §24 are explicit about that.

Run this on the remote: the per-window tables are ~318 kB and the SSH
channel truncates at ~256 kB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TIERS = ("0.1", "0.5", "2.0", "4.0")
N_BOOT = 10000
RNG_SEED = 20261106


def load_windows(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # scripts/evaluate.py wraps the table; a bare list is also accepted.
    rows = data["windows"] if isinstance(data, dict) else data
    if not rows:
        raise SystemExit(f"no windows in {path}")
    return rows


def counts(rows: list[dict], tier: str) -> np.ndarray:
    out = np.zeros((len(rows), 4), dtype=np.float64)
    for i, w in enumerate(rows):
        c = w[tier]
        out[i] = (
            c["hits"], c["misses"], c["false_alarms"],
            c["correct_negatives"],
        )
    return out


def csi(h: float, m: float, f: float) -> float:
    d = h + m + f
    return h / d if d > 0 else float("nan")


def bias(h: float, m: float, f: float) -> float:
    d = h + m
    return (h + f) / d if d > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", nargs="+", required=True,
                    help="per-window tables of the neural QPE arm")
    ap.add_argument("--b", nargs="+", required=True,
                    help="per-window tables of the analytical Z-R arm")
    ap.add_argument("--name-a", default="physical_qpe")
    ap.add_argument("--name-b", default="zr_k")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    if len(args.a) != len(args.b):
        raise SystemExit("--a and --b must have the same number of seeds")
    pairs = []
    for pa, pb in zip(args.a, args.b):
        ra, rb = load_windows(pa), load_windows(pb)
        if len(ra) != len(rb):
            raise SystemExit(
                f"window count differs: {pa}={len(ra)} vs {pb}={len(rb)}"
            )
        pairs.append((ra, rb, pa, pb))

    print("=" * 92)
    print(f"paired bootstrap | {args.name_a} minus {args.name_b} | "
          f"{args.n_boot} resamples | {len(pairs)} seed(s)")
    print("=" * 92)
    print(f"arm A: {', '.join(args.a)}")
    print(f"arm B: {', '.join(args.b)}")

    # A pairing sanity check that costs nothing: the truth field is the same
    # in both arms, so hits+misses (observed rain pixels) must match window
    # by window.  If it does not, the two tables do not describe the same
    # windows and no paired interval is meaningful.
    for tier in TIERS:
        worst = 0.0
        for ra, rb, _, _ in pairs:
            ca, cb = counts(ra, tier), counts(rb, tier)
            worst = max(worst, float(np.max(np.abs(
                (ca[:, 0] + ca[:, 1]) - (cb[:, 0] + cb[:, 1])
            ))))
        status = "OK -- same truth field" if worst == 0.0 else \
                 f"MISMATCH (max |Delta| = {worst:.0f}) -- pairing INVALID"
        print(f"  truth pairing @ {tier:>4} mm: {status}")
        if worst != 0.0:
            return 2

    for tier in TIERS:
        per_seed = []
        for ra, rb, _, _ in pairs:
            ca, cb = counts(ra, tier), counts(rb, tier)
            per_seed.append((ca, cb))

        n_win = min(len(a) for a, _ in per_seed)
        rng = np.random.default_rng(RNG_SEED)
        diffs = np.empty(args.n_boot)
        for i in range(args.n_boot):
            si = rng.integers(0, len(per_seed), len(per_seed))
            wi = rng.integers(0, n_win, n_win)
            da = db = 0.0
            for k in si:
                ca, cb = per_seed[k]
                da += csi(*ca[wi].sum(axis=0)[:3])
                db += csi(*cb[wi].sum(axis=0)[:3])
            diffs[i] = da - db

        point_a = np.mean([csi(*a.sum(axis=0)[:3]) for a, _ in per_seed])
        point_b = np.mean([csi(*b.sum(axis=0)[:3]) for _, b in per_seed])
        bias_a = np.mean([bias(*a.sum(axis=0)[:3]) for a, _ in per_seed])
        bias_b = np.mean([bias(*b.sum(axis=0)[:3]) for _, b in per_seed])
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        crosses = lo <= 0.0 <= hi
        print()
        print(f"  CSI@{tier}   {args.name_a} {point_a:.4f}  vs  "
              f"{args.name_b} {point_b:.4f}  ->  {point_a - point_b:+.4f}")
        print(f"    BIAS        {args.name_a} {bias_a:.4f}  vs  "
              f"{args.name_b} {bias_b:.4f}")
        print(f"    95 % paired interval : [{lo:+.4f}, {hi:+.4f}]")
        print("    verdict              : " + (
            "includes zero -- the two arms cannot be separated"
            if crosses else
            "excludes zero -- the difference exceeds window sampling noise"
        ))
        print("    NOTE: with 1 seed this interval does NOT cover "
              "training-seed variance (§24, 9.59x peak-end band).")
    print()
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
