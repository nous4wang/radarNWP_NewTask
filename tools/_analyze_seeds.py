"""Analyse the extra route P2/P3 seed runs.

Two questions are answered here, and they are answered from the files
rather than from the launcher's exit code.

1. The pass rate at the peak.  Route P2 compared three seeds of the
   QPE-QPF split against three seeds of the plain concatenation, and the
   two pass rates (3/3 against 2/3) could not be told apart: the Fisher
   exact test gives p = 1.0.  Ten seeds per configuration is what the
   master document asks for.  For each seed the peak is the best of the
   epochs that were scored, and the criterion is the persistence floor.

2. The lag sweep.  Route P3 measured one seed of four arms and the
   master document says the result cannot support a quantitative
   conclusion, because the spread of a single run at the peak reaches
   2.87x.  Three seeds per arm is the number the master document asks
   for, and the question it answers is whether the order of the arms is
   the same from one seed to the next.

The per-window tables are read as well, because the bootstrap resamples
the windows: the sampling uncertainty and the training uncertainty are
different things and only the second one is the variance band.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FLOOR = 0.0924
TIERS = ["0.1", "0.5", "2.0", "4.0"]
P2_ARMS = ("qpe_0", "base_concat")
P3_ARMS = ("assim_lag0", "assim_lag2", "assim_lag4", "assim_lag11")
# The arms of the lag sweep in the order the delay increases.
P3_ORDER = ("assim_lag0", "assim_lag2", "assim_lag4", "assim_lag11")
P3_LABEL = {
    "assim_lag0": "0 min",
    "assim_lag2": "12 min",
    "assim_lag4": "24 min",
    "assim_lag11": "no observation",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csi_of(path: Path, thr: str = "2.0") -> float:
    return load(path)["thresholds"][thr]["csi"]


def read_arm(evals: Path, arm: str) -> dict[tuple[str, str], Path]:
    """Every (seed, epoch) report of one arm, keyed by them."""
    out: dict[tuple[str, str], Path] = {}
    for f in sorted(evals.glob(f"{arm}_s*_ep*.eval.json")):
        stem = f.name[: -len(".eval.json")]
        tail = stem.split("_s", 1)[1]
        seed, ep = tail.split("_ep")
        out[(seed, ep)] = f
    return out


def counts_of(path: Path, thr: str = "2.0") -> np.ndarray:
    """The per-window contingency counts, as an (n, 3) array."""
    rows = load(path)["windows"]
    out = np.zeros((len(rows), 3))
    for i, w in enumerate(rows):
        c = w[thr]
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out


def csi_of_counts(counts: np.ndarray) -> float:
    denom = counts[:, 0].sum() + counts[:, 1].sum() + counts[:, 2].sum()
    return counts[:, 0].sum() / denom if denom > 0 else float("nan")


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """The one-sided Fisher exact p-value of a 2x2 table.

    ``a`` and ``b`` are the two counts of the first row, ``c`` and ``d``
    those of the second.  The p-value is the probability of the first
    row's first count being at least ``a``, under the hypergeometric
    distribution that the table's margins imply.
    """
    from math import comb

    n = a + b + c + d
    row1, col1 = a + b, a + c
    total = 0.0
    p = 0.0
    for x in range(max(0, col1 - (n - row1)), min(row1, col1) + 1):
        prob = comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
        if x >= a:
            p += prob
        total += prob
    return p / total if total else float("nan")


def bootstrap_two_level(
    seed_files: dict[str, Path],
    arm_a: str,
    arm_b: str,
    thr: str = "2.0",
    draws: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict:
    """The difference ``arm_a - arm_b`` with a two-level bootstrap.

    Every draw picks a seed for each arm and a set of windows, and both
    arms are scored on the same windows, because they were scored on the
    same windows in the same order.
    """
    rng = rng or np.random.default_rng(0)
    seeds_a = sorted(s for s in seed_files if s in arm_a)
    counts_a = {s: counts_of(seed_files[s], thr) for s in seeds_a}
    counts_b = {s: counts_of(seed_files[s], thr) for s in counts_b}
    seeds_b = sorted(counts_b)
    if not seeds_a or not seeds_b:
        return {}
    n_win = counts_a[seeds_a[0]].shape[0]
    diffs = []
    for _ in range(draws):
        sa = seeds_a[rng.integers(len(seeds_a))]
        sb = seeds_b[rng.integers(len(seeds_b))]
        idx = rng.integers(0, n_win, n_win)
        diffs.append(
            csi_of_counts(counts_a[sa][idx]) - csi_of_counts(counts_b[sb][idx])
        )
    diffs = np.array(diffs)
    mean_a = float(np.mean([csi_of_counts(counts_a[s]) for s in seeds_a]))
    mean_b = float(np.mean([csi_of_counts(counts_b[s]) for s in seeds_b]))
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": mean_a - mean_b,
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
    }


def pass_rate(evals: Path) -> None:
    print()
    print("=" * 96)
    print("1. the pass rate at the peak (criterion: CSI@2.0 > "
          f"{FLOOR}, the persistence floor)")
    print("=" * 96)
    peaks: dict[str, dict[str, tuple[str, float]]] = {}
    for arm in P2_ARMS:
        rows = read_arm(evals, arm)
        if not rows:
            print(f"\n{arm}: no evaluations found in {evals}")
            continue
        print(f"\n--- {arm} ---")
        print(f"{'seed':>6}{'ep':>5}{'CSI@0.1':>10}{'CSI@0.5':>10}"
              f"{'CSI@2.0':>10}{'CSI@4.0':>10}   note")
        best: dict[str, tuple[str, float]] = {}
        for (seed, ep), path in sorted(rows.items()):
            t = load(path)["thresholds"]
            vals = {th: t[th]["csi"] for th in TIERS}
            note = ""
            if seed not in best or vals["2.0"] > best[seed][1]:
                best[seed] = (ep, vals["2.0"])
                note = "peak of this seed"
            print(f"{seed:>6}{ep:>5}{vals['0.1']:>10.4f}{vals['0.5']:>10.4f}"
                  f"{vals['2.0']:>10.4f}{vals['4.0']:>10.4f}   {note}")
        peaks[arm] = best
        passed = sum(1 for _, v in best.values() if v > FLOOR)
        print(f"  the peak of each seed:")
        for seed in sorted(best):
            ep, v = best[seed]
            flag = "above the floor" if v > FLOOR else "BELOW the floor"
            print(f"    seed {seed}: epoch {ep}, CSI@2.0 = {v:.4f}  {flag}")
        vals = [v for _, v in best.values()]
        print(f"  PASS RATE: {passed}/{len(vals)}")
        if len(vals) >= 2 and min(vals) > 0:
            print(f"  the peak CSI@2.0 across seeds: min {min(vals):.4f} "
                  f"max {max(vals):.4f}, ratio {max(vals) / min(vals):.2f}x")
        if vals:
            print(f"  mean of the peaks: {sum(vals) / len(vals):.4f} "
                  f"({'above' if sum(vals) / len(vals) > FLOOR else 'below'}"
                  f" the floor {FLOOR})")

    if len(peaks) == 2:
        print()
        print("  the two pass rates, side by side:")
        table = {}
        for arm in P2_ARMS:
            best = peaks[arm]
            passed = sum(1 for _, v in best.values() if v > FLOOR)
            table[arm] = (passed, len(best) - passed)
            print(f"    {arm:>14}: {passed}/{len(best)} above the floor")
        (a_pass, a_fail), (b_pass, b_fail) = (
            table[P2_ARMS[0]], table[P2_ARMS[1]])
        p = fisher_exact_2x2(a_pass, a_fail, b_pass, b_fail)
        print(f"  Fisher exact (one-sided): p = {p:.4f}")
        print(f"  -> {'the two configurations are told apart' if p < 0.05 else 'the two configurations are NOT told apart'}"
              f" at the 5% level.")


def lag_sweep(evals: Path) -> None:
    print()
    print("=" * 96)
    print("2. the lag sweep: is the order of the arms the same from seed to seed?")
    print("=" * 96)
    data: dict[str, dict[tuple[str, str], float]] = {}
    for arm in P3_ARMS:
        rows = read_arm(evals, arm)
        data[arm] = {k: csi_of(v) for k, v in rows.items()}
    seeds = sorted({s for arm in P3_ARMS for (s, _) in data[arm]})
    if not seeds:
        print("no evaluations found")
        return
    epochs = sorted({e for arm in P3_ARMS for (_, e) in data[arm]})

    for ep in epochs:
        print(f"\n--- epoch {ep} ---")
        print(f"{'arm':>16}{'delay':>16}{'CSI@2.0':>10}   order")
        vals = {}
        for arm in P3_ORDER:
            v = data[arm].get((seeds[0], ep))
            if v is None:
                continue
            vals[arm] = v
        for arm in P3_ORDER:
            if arm in vals:
                print(f"{arm:>16}{P3_LABEL[arm]:>16}{vals[arm]:>10.4f}")
        order = sorted(vals, key=lambda a: -vals[a])
        print("  the order, best first: " + " > ".join(
            f"{a.replace('assim_lag', 'lag')}" for a in order))
        # The physically expected order is that the score falls as the
        # observation gets later.
        expected = [a for a in P3_ORDER if a in vals]
        by_delay = sorted(expected, key=lambda a: P3_ORDER.index(a))
        monotone = all(
            vals[by_delay[i]] >= vals[by_delay[i + 1]]
            for i in range(len(by_delay) - 1)
        )
        print(f"  monotone in the delay: {monotone}")

    print()
    print("=" * 96)
    print("   per seed, at the peak epoch of that seed")
    print("=" * 96)
    per_seed_order = {}
    for seed in seeds:
        best_ep = {}
        for arm in P3_ARMS:
            cands = [data[arm][(seed, e)] for e in epochs if (seed, e) in data[arm]]
            if cands:
                best_ep[arm] = max(cands)
        if len(best_ep) < 2:
            continue
        order = sorted(best_ep, key=lambda a: -best_ep[a])
        per_seed_order[seed] = order
        print(f"\n  seed {seed}:")
        for arm in P3_ORDER:
            if arm in best_ep:
                print(f"    {arm:>16} ({P3_LABEL[arm]:>14}): {best_ep[arm]:.4f}")
        print("    order: " + " > ".join(a.replace("assim_lag", "lag") for a in order))
        obs = best_ep.get("assim_lag0")
        nobs = best_ep.get("assim_lag11")
        if obs and nobs:
            print(f"    the observation is worth {obs / nobs:.2f}x")
        by_delay = [a for a in P3_ORDER if a in best_ep]
        mono = all(best_ep[by_delay[i]] >= best_ep[by_delay[i + 1]]
                   for i in range(len(by_delay) - 1))
        print(f"    monotone in the delay: {mono}")

    orders = {tuple(v) for v in per_seed_order.values()}
    print()
    print(f"  the order of the arms is the same in {len(orders)} distinct "
          f"form(s) over {len(per_seed_order)} seeds:")
    for o in orders:
        print("    " + " > ".join(a.replace("assim_lag", "lag") for a in o))
    if len(orders) == 1 and len(per_seed_order) > 1:
        print("  -> the order does not depend on the seed.")
    elif len(per_seed_order) > 1:
        print("  -> the order DEPENDS on the seed, so the single-seed lag")
        print("     curve cannot be read as a property of the lag.")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/wf/radarNWP/runs/route_seeds/evals")
    print(f"reading {root}")
    pass_rate(root)
    lag_sweep(root)


if __name__ == "__main__":
    main()
