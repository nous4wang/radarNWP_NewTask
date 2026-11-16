"""Read a stage of the four-tier work and report what it says.

The numbers that decide anything here are not single scores. Section 4.13
measured a spread of 9.59x over ten seeds at the peak of the curve, so a
single run is worth almost nothing and a three-seed range is not a
statistic at all. Everything below therefore reports the median, the
spread, the rate at which the arm clears the persistence floor, and a
two-level bootstrap interval over the seeds and the evaluation windows
together.

The bootstrap has to run where the window tables are: they are about
three hundred kilobytes each and the SSH channel truncates at about two
hundred and fifty, so the copy that comes down would not parse.

Usage:
    python tools/_analyze_four_tiers.py <root> --stage 2
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FLOOR = 0.0924
PYSTEPS = 0.1014
N_BOOT = 10000
RNG_SEED = 20261103
TIERS = ("0.1", "0.5", "2.0", "4.0")

#: The names the arms are written under, per stage.
STAGE_ARMS: dict[int, tuple[str, ...]] = {
    2: ("fss35", "scale3", "scale5"),
    3: ("full", "adv", "res"),
    5: ("lag0", "lag2", "lag4", "lag11"),
}

#: Where the ten seeds of the baseline live, and how they are named.
#:
#: The four-tier runs write their reports under a directory per arm and
#: seed; the earlier route runs wrote flat files into one directory with
#: the epoch in the name. Both are read here.
DEFAULT_BASELINE_DIR = "/home/wf/radarNWP/runs/route_seeds/evals"

#: The arm each stage is judged against, and the epoch of that arm.
#:
#: The second stage changes the objective of the concatenation model, so
#: its control is the concatenation model under the balanced objective.
#: The third stage changes the extrapolation of the inversion model, so
#: its control is the inversion model itself.
BASELINE_ARM = {2: "base_concat", 3: "qpe_0", 5: ""}
BASELINE_EPOCH = {2: 0, 3: 0, 5: 0}

_ARM_RE = re.compile(r"^(?P<arm>[A-Za-z]+[0-9]*)_s(?P<seed>\d+)$")


def csi_of(h: float, m: float, f: float) -> float:
    d = h + m + f
    return h / d if d > 0 else float("nan")


def window_counts(path: Path, tier: str) -> np.ndarray | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    rows = data.get("windows")
    if not rows:
        return None
    out = np.zeros((len(rows), 3))
    for i, w in enumerate(rows):
        c = w[tier]
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out


def report_path(run_dir: Path, epoch: int) -> Path:
    """The report of one epoch, wherever the scorer put it.

    The scorer writes ``<stem>.eval.json`` and its window table beside
    it under the name ``<stem>.eval.json.windows.json``. The window
    table is found by suffix rather than by replacing ``.eval.json``:
    the two names differ, and a replaced name points at a file that was
    never written.
    """
    stem = f"epoch_{epoch:02d}"
    return run_dir / f"{stem}.eval.json"


def window_path(report: Path) -> Path:
    """The window table that belongs to a report.

    Two naming conventions coexist in this project:

    * the four-tier runs append ``.windows.json`` to the whole report
      name, giving ``epoch_00.eval.json.windows.json``;
    * the earlier route runs name the table ``<stem>.windows.json`` where
      ``<stem>`` is the report without ``.eval.json``, giving
      ``base_concat_s0_ep00.windows.json``.

    Return whichever of the two actually exists, falling back to the
    four-tier name so a genuinely missing table still surfaces as such.
    """
    main = report.with_name(report.name + ".windows.json")
    if main.exists():
        return main
    alt = report.with_name(
        report.with_suffix("").with_suffix("").name + ".windows.json"
    )
    if alt.exists():
        return alt
    return main


def collect(root: Path, arms: tuple[str, ...]) -> dict:
    """arm -> (seed, epoch) -> report path.

    The arm of a run is the name of the directory it was trained in:
    ``fss35_s0`` is the fractions arm at seed zero. The scorer writes
    its reports under ``<root>/eval/<arm>_s<seed>/``, and that name is
    the one thing about a run that is unique -- the run directory itself
    is stamped with the wall clock, so several runs share one.
    """
    table: dict = defaultdict(dict)
    eval_root = root / "eval"
    for arm_dir in sorted(eval_root.glob("*_s*")):
        m = _ARM_RE.match(arm_dir.name)
        if not m:
            continue
        arm = m.group("arm")
        if arm not in arms:
            continue
        seed = int(m.group("seed"))
        for report in sorted(arm_dir.glob("epoch_*.eval.json")):
            try:
                # report is ``epoch_NN.eval.json``: strip both suffixes
                # (``.json`` then ``.eval``) before splitting on ``_``.
                epoch = int(report.with_suffix("").with_suffix("").name.split("_")[1])
            except (IndexError, ValueError):
                continue
            table[arm][(seed, epoch)] = report
    return table


def collect_baseline(directory: Path, arm: str, epoch: int) -> dict:
    """seed -> report path, for the baseline arm."""
    table: dict = {}
    for report in sorted(directory.glob(f"{arm}_s*_ep{epoch:02d}.eval.json")):
        m = re.search(r"_s(\d+)_ep(\d+)", report.with_suffix("").with_suffix("").name)
        if not m:
            continue
        table[int(m.group(1))] = report
    return table


def read_csi(report: Path, tier: str) -> float | None:
    try:
        data = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return float(data["thresholds"][tier]["csi"])
    except (KeyError, TypeError, ValueError):
        return None


def describe(table: dict, arms: tuple[str, ...], epochs: tuple[int, ...]) -> None:
    print("=" * 92)
    print("the arms, per epoch: median, spread, and the rate over the floor")
    print("=" * 92)
    print()
    print(f"  the persistence floor is {FLOOR:.4f} and pysteps is {PYSTEPS:.4f}")
    print()
    print(f"  {'arm':<10} {'ep':<4} {'n':<4} {'median':>8} {'mean':>8} "
          f"{'spread':>8} {'over floor':>11} {'over pysteps':>13}")
    for arm in arms:
        for epoch in epochs:
            vals = []
            for (seed, e), report in table.get(arm, {}).items():
                if e != epoch:
                    continue
                csi = read_csi(report, "2.0")
                if csi is not None:
                    vals.append(csi)
            if not vals:
                continue
            vals.sort()
            med = vals[len(vals) // 2]
            spread = vals[-1] / vals[0] if vals[0] > 0 else float("nan")
            over = sum(1 for v in vals if v > FLOOR)
            over_p = sum(1 for v in vals if v > PYSTEPS)
            print(
                f"  {arm:<10} {epoch:<4} {len(vals):<4} {med:>8.4f}"
                f" {float(np.mean(vals)):>8.4f} {spread:>7.2f}x"
                f" {over:>6}/{len(vals):<4} {over_p:>8}/{len(vals):<4}"
            )
    print()


def paired_bootstrap(table: dict, arm_a: str, arm_b: str,
                     epochs: tuple[int, ...], tiers: tuple[str, ...]) -> None:
    """The two-level bootstrap: the seed and the window, together."""
    print("=" * 92)
    print(f"the paired bootstrap: {arm_a} minus {arm_b}")
    print("=" * 92)
    print()
    seeds_a = {s for s, _ in table.get(arm_a, {})}
    seeds_b = {s for s, _ in table.get(arm_b, {})}
    seeds = sorted(seeds_a & seeds_b)
    if not seeds:
        print(f"  no seed is shared by {arm_a} and {arm_b}")
        print()
        return

    for epoch in epochs:
        for tier in tiers:
            per_seed = []
            for s in seeds:
                ra = table[arm_a].get((s, epoch))
                rb = table[arm_b].get((s, epoch))
                if ra is None or rb is None:
                    per_seed = []
                    break
                ca = window_counts(window_path(ra), tier)
                cb = window_counts(window_path(rb), tier)
                if ca is None or cb is None:
                    per_seed = []
                    break
                n = min(len(ca), len(cb))
                per_seed.append((ca[:n], cb[:n]))
            if not per_seed:
                print(f"  ep{epoch:02d} CSI@{tier}: window tables missing, skipped")
                continue

            n_win = min(len(a) for a, _ in per_seed)
            point_a = float(np.mean([csi_of(*a.sum(axis=0)) for a, _ in per_seed]))
            point_b = float(np.mean([csi_of(*b.sum(axis=0)) for _, b in per_seed]))

            rng = np.random.default_rng(RNG_SEED)
            diffs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                si = rng.integers(0, len(per_seed), len(per_seed))
                wi = rng.integers(0, n_win, n_win)
                da = db = 0.0
                for k in si:
                    ca, cb = per_seed[k]
                    da += csi_of(*ca[wi].sum(axis=0))
                    db += csi_of(*cb[wi].sum(axis=0))
                diffs[i] = da - db
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            crosses = lo <= 0.0 <= hi
            print(f"  ep{epoch:02d} CSI@{tier}   ({len(seeds)} seeds)")
            print(f"    mean over seeds : {point_a:.4f} - {point_b:.4f} = "
                  f"{point_a - point_b:+.4f}")
            print(f"    95 % interval    : [{lo:+.4f}, {hi:+.4f}]")
            print("    verdict         : " + (
                "includes zero -- the arms cannot be separated"
                if crosses else
                "excludes zero -- the difference exceeds the sampling uncertainty"
            ))
            print()


def baseline_compare(table: dict, arm: str, baseline_dir: Path,
                     baseline_arm: str, baseline_epoch: int,
                     epochs: tuple[int, ...]) -> None:
    """The arm against the ten seeds of the baseline arm.

    The baseline is read from the earlier route runs, which wrote flat
    files; the arm under test is read from the four-tier directory. The
    two are paired by seed, and the window tables are paired by window
    index, because both were scored over the same test split.
    """
    if not baseline_arm:
        return
    base = collect_baseline(baseline_dir, baseline_arm, baseline_epoch)
    if not base:
        print(f"  the baseline arm {baseline_arm} was not found in {baseline_dir}")
        print()
        return

    print("=" * 92)
    print(f"the arm against the baseline: {arm} minus {baseline_arm} "
          f"(ep{baseline_epoch:02d})")
    print("=" * 92)
    print()

    for epoch in epochs:
        seeds = sorted(
            s for s, _ in table.get(arm, {}) if s in base
        )
        if not seeds:
            print(f"  ep{epoch:02d}: no seed is shared with the baseline")
            continue

        arm_vals = []
        base_vals = []
        for s in seeds:
            ra = table[arm].get((s, epoch))
            if ra is None:
                continue
            ca = read_csi(ra, "2.0")
            cb = read_csi(base[s], "2.0")
            if ca is None or cb is None:
                continue
            arm_vals.append(ca)
            base_vals.append(cb)
        if not arm_vals:
            print(f"  ep{epoch:02d}: no readable scores")
            continue

        def stats(vals: list[float]) -> tuple[float, float, float, int]:
            s = sorted(vals)
            med = s[len(s) // 2]
            spread = s[-1] / s[0] if s[0] > 0 else float("nan")
            over = sum(1 for v in s if v > FLOOR)
            return med, float(np.mean(s)), spread, over

        med_a, mean_a, spread_a, over_a = stats(arm_vals)
        med_b, mean_b, spread_b, over_b = stats(base_vals)
        print(f"  ep{epoch:02d} CSI@2.0   ({len(arm_vals)} seeds)")
        print(f"    {arm:<12} median {med_a:.4f}  mean {mean_a:.4f}  "
              f"spread {spread_a:.2f}x  over the floor {over_a}/{len(arm_vals)}")
        print(f"    {baseline_arm:<12} median {med_b:.4f}  mean {mean_b:.4f}  "
              f"spread {spread_b:.2f}x  over the floor {over_b}/{len(base_vals)}")
        print(f"    median difference : {med_a - med_b:+.4f}")
        print()

        # The paired bootstrap over the seeds and the windows together.
        for tier in ("0.1", "0.5", "2.0", "4.0"):
            per_seed = []
            for s in seeds:
                ra = table[arm].get((s, epoch))
                if ra is None:
                    per_seed = []
                    break
                ca = window_counts(window_path(ra), tier)
                cb = window_counts(window_path(base[s]), tier)
                if ca is None or cb is None:
                    per_seed = []
                    break
                n = min(len(ca), len(cb))
                per_seed.append((ca[:n], cb[:n]))
            if not per_seed:
                print(f"  ep{epoch:02d} CSI@{tier}: window tables missing, skipped")
                continue

            n_win = min(len(a) for a, _ in per_seed)
            point_a = float(np.mean([csi_of(*a.sum(axis=0)) for a, _ in per_seed]))
            point_b = float(np.mean([csi_of(*b.sum(axis=0)) for _, b in per_seed]))

            rng = np.random.default_rng(RNG_SEED)
            diffs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                si = rng.integers(0, len(per_seed), len(per_seed))
                wi = rng.integers(0, n_win, n_win)
                da = db = 0.0
                for k in si:
                    ca, cb = per_seed[k]
                    da += csi_of(*ca[wi].sum(axis=0))
                    db += csi_of(*cb[wi].sum(axis=0))
                diffs[i] = da - db
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            crosses = lo <= 0.0 <= hi
            print(f"  ep{epoch:02d} CSI@{tier}   ({len(seeds)} seeds)")
            print(f"    mean over seeds : {point_a:.4f} - {point_b:.4f} = "
                  f"{point_a - point_b:+.4f}")
            print(f"    95 % interval    : [{lo:+.4f}, {hi:+.4f}]")
            print("    verdict         : " + (
                "includes zero -- the arms cannot be separated"
                if crosses else
                "excludes zero -- the difference exceeds the sampling uncertainty"
            ))
            print()


def order_across_seeds(table: dict, arms: tuple[str, ...],
                       epoch: int) -> None:
    """The order of the arms, per seed.

    A curve whose order changes with the seed cannot support a
    quantitative claim, and saying so is the answer rather than a
    failure to find one.
    """
    print("=" * 92)
    print(f"the order of the arms at ep{epoch:02d}, per seed")
    print("=" * 92)
    print()
    seeds = sorted({s for arm in arms for s, e in table.get(arm, {})
                    if e == epoch})
    print(f"  {'seed':<6} {'order (best to worst)':<50} {'best':<8}")
    for s in seeds:
        scored = []
        for arm in arms:
            report = table.get(arm, {}).get((s, epoch))
            if report is None:
                continue
            csi = read_csi(report, "2.0")
            if csi is None:
                continue
            scored.append((csi, arm))
        if not scored:
            continue
        scored.sort(reverse=True)
        order = " > ".join(f"{arm}({csi:.4f})" for csi, arm in scored)
        print(f"  {s:<6} {order:<50} {scored[0][1]:<8}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stage", type=int, required=True, choices=[2, 3, 5])
    ap.add_argument("--epochs", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--tiers", nargs="*", default=list(TIERS))
    ap.add_argument(
        "--baseline-dir", default=DEFAULT_BASELINE_DIR,
        help="where the ten seeds of the baseline arm were scored",
    )
    ap.add_argument(
        "--baseline-arm", default=None,
        help="the arm to compare against; the default is the one the "
             "criterion of this stage names",
    )
    ap.add_argument(
        "--baseline-epoch", type=int, default=None,
        help="the epoch of the baseline arm",
    )
    ap.add_argument(
        "--pair",
        nargs=2,
        default=None,
        help="the two arms to compare; the default compares every arm "
             "with the first one",
    )
    args = ap.parse_args()

    root = Path(args.root)
    arms = STAGE_ARMS[args.stage]
    table = collect(root, arms)
    n = sum(len(v) for v in table.values())
    print(f"read {n} evaluations from {root}")
    print()

    describe(table, arms, tuple(args.epochs))

    baseline_arm = args.baseline_arm
    if baseline_arm is None:
        baseline_arm = BASELINE_ARM[args.stage]
    baseline_epoch = args.baseline_epoch
    if baseline_epoch is None:
        baseline_epoch = BASELINE_EPOCH[args.stage]
    baseline_dir = Path(args.baseline_dir)

    if baseline_arm:
        primary = arms[0]
        baseline_compare(
            table, primary, baseline_dir, baseline_arm, baseline_epoch,
            tuple(args.epochs),
        )

    if args.pair:
        pairs = [tuple(args.pair)]
    else:
        pairs = [(arms[0], a) for a in arms[1:]]
    for a, b in pairs:
        paired_bootstrap(table, a, b, tuple(args.epochs), tuple(args.tiers))

    order_across_seeds(table, arms, tuple(args.epochs)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
