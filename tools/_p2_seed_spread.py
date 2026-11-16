"""The seed spread, measured where the criterion is written against.

The variance band in the master document (1.78x) was measured at the
last epoch, on two pairs.  This measures the spread again, per arm, at
both the peak epoch and the last epoch, over three seeds each -- and it
asks the question the criterion is actually written against: how often
does an arm clear the persistence floor when the seed is left to chance?

The pass rate matters more than the mean.  A configuration whose mean
clears the floor but whose seed spread puts one run below it has not
learned to place the heavy rain; it has learned it on two runs out of
three.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FLOOR = 0.0924
PYSTEPS = 0.1014
TIERS = ["0.1", "0.5", "2.0", "4.0"]
ARMS = ("qpe_0", "base_concat")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_rows(seed_dir: Path, arm: str):
    rows = []
    for f in sorted(seed_dir.glob(f"{arm}_s*_ep*.eval.json")):
        stem = f.name[: -len(".eval.json")]
        tail = stem.split("_s", 1)[1]
        seed, ep = tail.split("_ep")
        rows.append((seed, ep, load(f)))
    return rows


def main() -> None:
    root = Path(sys.argv[1])
    seed_dir = root / "seeds"

    print("=" * 96)
    print("1. the seed spread, per arm, per epoch")
    print("=" * 96)
    print(f"  {'arm':14s}{'ep':>5}{'seeds':>7}{'min':>9}{'max':>9}"
          f"{'mean':>9}{'spread':>9}{'band 1.78x':>12}")
    spreads = {}
    for arm in ARMS:
        rows = seed_rows(seed_dir, arm)
        by_ep = {}
        for seed, ep, rep in rows:
            by_ep.setdefault(ep, []).append(rep["thresholds"]["2.0"]["csi"])
        for ep in sorted(by_ep):
            vals = by_ep[ep]
            lo, hi = min(vals), max(vals)
            mean = sum(vals) / len(vals)
            ratio = hi / lo if lo > 0 else float("nan")
            spreads[(arm, ep)] = ratio
            print(f"  {arm:14s}{ep:>5}{len(vals):>7}{lo:>9.4f}{hi:>9.4f}"
                  f"{mean:>9.4f}{ratio:>8.2f}x"
                  f"{'exceeds' if ratio > 1.78 else 'within':>12}")
    print()
    print("  The band in the master document (1.78x) was measured at the")
    print("  last epoch from two pairs.  The spread at the peak epoch is")
    print("  the one that matters for the criterion.")

    print()
    print("=" * 96)
    print("2. how often does a seed clear the persistence floor?")
    print("=" * 96)
    print(f"  {'arm':14s}{'ep':>5}{'per-seed CSI@2.0':>34}"
          f"{'pass rate':>13}{'mean':>9}")
    for arm in ARMS:
        rows = seed_rows(seed_dir, arm)
        by_ep = {}
        for seed, ep, rep in rows:
            by_ep.setdefault(ep, []).append(
                (seed, rep["thresholds"]["2.0"]["csi"]))
        for ep in sorted(by_ep):
            items = sorted(by_ep[ep])
            n_pass = sum(1 for _, v in items if v > FLOOR)
            mean = sum(v for _, v in items) / len(items)
            detail = " ".join(f"s{s}={v:.4f}" for s, v in items)
            print(f"  {arm:14s}{ep:>5}{detail:>34}"
                  f"{f'{n_pass}/{len(items)}':>13}{mean:>9.4f}")

    print()
    print("=" * 96)
    print("3. the same question at the other thresholds")
    print("=" * 96)
    for ep in ("00", "19"):
        print(f"\n  epoch {ep}")
        print(f"  {'arm':14s}{'thr':>6}{'min':>9}{'max':>9}{'mean':>9}"
              f"{'spread':>9}{'above floor':>13}")
        for arm in ARMS:
            rows = [r for r in seed_rows(seed_dir, arm) if r[1] == ep]
            if not rows:
                continue
            for th in TIERS:
                vals = [r[2]["thresholds"][th]["csi"] for r in rows]
                lo, hi = min(vals), max(vals)
                mean = sum(vals) / len(vals)
                ratio = hi / lo if lo > 0 else float("nan")
                n_pass = sum(1 for v in vals if v > FLOOR)
                print(f"  {arm:14s}{th:>6}{lo:>9.4f}{hi:>9.4f}{mean:>9.4f}"
                      f"{ratio:>8.2f}x{f'{n_pass}/{len(vals)}':>13}")

    print()
    print("=" * 96)
    print("4. the reading")
    print("=" * 96)
    q00 = spreads.get(("qpe_0", "00"))
    b00 = spreads.get(("base_concat", "00"))
    q19 = spreads.get(("qpe_0", "19"))
    b19 = spreads.get(("base_concat", "19"))
    print(f"  the peak epoch   : qpe_0 spread {q00:.2f}x, "
          f"base_concat spread {b00:.2f}x")
    print(f"  the last epoch   : qpe_0 spread {q19:.2f}x, "
          f"base_concat spread {b19:.2f}x")
    print()
    print("  The single-run band of 1.78x was measured at the last epoch.")
    print("  At the peak epoch the spread is "
          f"{max(q00, b00):.2f}x, so a single-run comparison at the")
    print("  peak cannot decide anything either.")


if __name__ == "__main__":
    main()
