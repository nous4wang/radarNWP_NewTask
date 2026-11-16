"""Read the multi-seed evaluations and report the two open questions.

Section 9 of the master document listed two things the multi-seed runs
were supposed to settle: the pass rate of the arms over the persistence
floor at the peak of the curve, and whether the lag sweep reproduces
across seeds. This reads the evaluations that came back and answers both.

The persistence floor is 0.0924 at the 2.0 mm tier.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

FLOOR = 0.0924
PYSTEPS = 0.1014
TIERS = ("0.1", "0.5", "2.0", "4.0")


def csi(path: Path, tier: str = "2.0") -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["thresholds"][tier]["csi"])


def bias(path: Path, tier: str = "0.5") -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["thresholds"][tier]["bias"])


def collect(evals: Path) -> dict:
    """Group the evaluations by arm and seed."""
    table = defaultdict(dict)
    for path in sorted(evals.glob("*.eval.json")):
        stem = path.name[: -len(".eval.json")]
        arm, seed, epoch = stem.rsplit("_", 2)
        table[arm][(int(seed[1:]), int(epoch[2:]))] = path
    return table


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test for a >= b in the first margin.

    The table is [[a, b], [c, d]]: the first row is the arm that was
    expected to pass more often, the second row is the reference arm.
    """
    from math import comb

    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c

    def prob(x: int) -> float:
        return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)

    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        if x >= a:
            total += prob(x)
    return min(1.0, total)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools/_evals/route_seeds")
    evals = root / "evals"
    if not evals.is_dir():
        evals = root
    table = collect(evals)
    print(f"read {sum(len(v) for v in table.values())} evaluations from {evals}")
    print()
    print("=" * 96)
    print("question 1: the pass rate over the persistence floor at the peak of the curve")
    print("=" * 96)
    print()
    print(f"the floor is {FLOOR:.4f} at the 2.0 mm tier; pysteps is {PYSTEPS:.4f}")
    print()
    print("  arm          epoch   n   passes   values (CSI@2.0)")
    for arm in ("qpe_0", "base_concat"):
        for epoch in (0, 1, 2):
            pairs = [(s, p) for (s, e), p in table.get(arm, {}).items() if e == epoch]
            if not pairs:
                continue
            pairs.sort()
            vals = [csi(p) for _, p in pairs]
            passes = sum(1 for v in vals if v > FLOOR)
            over_p = sum(1 for v in vals if v > PYSTEPS)
            spread = max(vals) / min(vals) if min(vals) > 0 else float("nan")
            shown = " ".join(f"{v:.4f}" for v in vals)
            print(
                f"  {arm:<13} {epoch:<6} {len(vals):<3} {passes}/{len(vals)}"
                f"   spread {spread:.2f}x   {shown}"
            )
            print(
                f"  {'':<13} {'':<6} {'':<3} over pysteps: {over_p}/{len(vals)}"
            )
    print()

    q0 = {s: csi(p) for (s, e), p in table.get("qpe_0", {}).items() if e == 0}
    b0 = {s: csi(p) for (s, e), p in table.get("base_concat", {}).items() if e == 0}
    if q0 and b0:
        seeds = sorted(set(q0) & set(b0))
        q_pass = sum(1 for s in seeds if q0[s] > FLOOR)
        b_pass = sum(1 for s in seeds if b0[s] > FLOOR)
        q_over = sum(1 for s in seeds if q0[s] > PYSTEPS)
        b_over = sum(1 for s in seeds if b0[s] > PYSTEPS)
        print(f"  paired over {len(seeds)} seeds, at epoch 0")
        print(f"    qpe_0      over the floor {q_pass}/{len(seeds)}   over pysteps {q_over}/{len(seeds)}")
        print(f"    base_concat over the floor {b_pass}/{len(seeds)}   over pysteps {b_over}/{len(seeds)}")
        p = fisher_exact_greater(q_pass, len(seeds) - q_pass, b_pass, len(seeds) - b_pass)
        print(f"    one-sided Fisher (greater): p = {p:.4f}")
        print()
        print("  paired difference per seed (qpe_0 minus base_concat):")
        diffs = [q0[s] - b0[s] for s in seeds]
        for s, d in zip(seeds, diffs):
            print(f"    seed {s}: {q0[s]:.4f} - {b0[s]:.4f} = {d:+.4f}")
        print(f"    mean {sum(diffs)/len(diffs):+.4f}   median {sorted(diffs)[len(diffs)//2]:+.4f}")
        print()

    print("=" * 96)
    print("question 2: does the lag sweep reproduce across seeds")
    print("=" * 96)
    print()
    print("  arm            seed   ep00    ep01    ep02    ep00 vs ep02")
    lag_arms = [a for a in table if a.startswith("assim_lag")]
    for arm in sorted(lag_arms):
        seeds = sorted({s for s, _ in table[arm]})
        for seed in seeds:
            row = []
            for epoch in (0, 1, 2):
                p = table[arm].get((seed, epoch))
                row.append(csi(p) if p else float("nan"))
            ratio = row[0] / row[2] if row[2] > 0 else float("nan")
            print(
                f"  {arm:<14} {seed:<6} {row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f}"
                f"   {ratio:6.2f}x"
            )
        print()

    print("  the ordering of the arms at epoch 0, per seed:")
    for seed in sorted({s for a in lag_arms for s, _ in table[a]}):
        row = []
        for arm in ("assim_lag0", "assim_lag2", "assim_lag4", "assim_lag11"):
            p = table.get(arm, {}).get((seed, 0))
            row.append((arm, csi(p) if p else float("nan")))
        best = max(row, key=lambda x: x[1])
        order = " > ".join(f"{a.replace('assim_', '')}({v:.4f})" for a, v in row)
        print(f"    seed {seed}: {order}   best = {best[0]}")
    print()

    print("  bias at 0.5 mm, epoch 0 (persistence sits at 0.9048):")
    for arm in ("qpe_0", "base_concat") + tuple(sorted(lag_arms)):
        rows = []
        for (s, e), p in sorted(table.get(arm, {}).items()):
            if e == 0:
                rows.append(bias(p))
        if rows:
            print(f"    {arm:<16} " + " ".join(f"{v:.3f}" for v in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
