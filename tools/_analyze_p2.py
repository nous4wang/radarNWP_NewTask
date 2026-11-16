"""Read route P2: the epoch curve, the seed spread, the bootstrap interval.

Three questions, three sections.

1. Where does each configuration peak, and does the peak cross the
   persistence floor (CSI@2.0 = 0.0924)? That floor is the criterion that
   was written down before the run.
2. How far apart are the seeds of one configuration? This is the run-to-run
   noise, measured rather than assumed.
3. Does the difference between the two configurations survive both? The
   bootstrap resamples the evaluation windows, which is the sampling
   uncertainty; the seed spread is the training uncertainty. A difference
   has to clear both to be a difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FLOOR = 0.0924          # persistence CSI@2.0, the strong-rain floor
PYSTEPS = 0.1014        # the physical extrapolation baseline
TIERS = ["0.1", "0.5", "2.0", "4.0"]
N_BOOT = 10000
RNG_SEED = 12345


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def epoch_of(path: Path) -> int | None:
    """The epoch number in ``epoch_07.eval.json``.

    ``Path.stem`` of that name is ``epoch_07.eval``, not ``epoch_07``:
    the suffix is stripped once, so the trailing ``.eval`` survives and
    the integer parse fails.  The name is split on the underscore and
    the remainder is cut at the first dot.
    """
    parts = path.name.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].split(".")[0])
    except (ValueError, IndexError):
        return None


def read_curve(curve_dir: Path) -> None:
    print("=" * 96)
    print("1. the epoch curve (seed 0)")
    print("=" * 96)
    print(f"criterion: CSI@2.0 > {FLOOR} (the persistence floor)")
    for arm in ("qpe_0", "qpe_05", "base_concat"):
        d = curve_dir / f"{arm}_eval"
        if not d.exists():
            print(f"\n{arm}: no evaluations in {d}")
            continue
        rows = []
        for f in sorted(d.glob("epoch_*.eval.json")):
            epoch = epoch_of(f)
            if epoch is None:
                continue
            rep = load(f)
            t = rep["thresholds"]
            rows.append((epoch, {th: t[th]["csi"] for th in TIERS}))
        if not rows:
            print(f"\n{arm}: no evaluations found")
            continue
        rows.sort()
        best = max(rows, key=lambda r: r[1]["2.0"])
        print(f"\n--- {arm} ---")
        print(f"{'ep':>4}{'CSI@0.1':>10}{'CSI@0.5':>10}{'CSI@2.0':>10}{'CSI@4.0':>10}   note")
        for epoch, t in rows:
            note = ""
            if t["2.0"] > FLOOR:
                note = "above the floor"
            elif t["2.0"] > PYSTEPS:
                note = "above pysteps"
            mark = "  <- peak" if epoch == best[0] else ""
            print(f"{epoch:>4}{t['0.1']:>10.4f}{t['0.5']:>10.4f}"
                  f"{t['2.0']:>10.4f}{t['4.0']:>10.4f}   {note}{mark}")
        peak = best[1]
        print(f"  peak CSI@2.0 = {peak['2.0']:.4f} at epoch {best[0]}"
              f"   (floor {FLOOR}, pysteps {PYSTEPS})")
        verdict = "PASS" if peak["2.0"] > FLOOR else "FAIL"
        print(f"  criterion: {verdict}")


def read_seeds(seed_dir: Path) -> None:
    """The seed spread of the peak.

    The seeds are scored at several epochs, because the peak is what the
    comparison is about and the peak is not at the last epoch.  Each
    seed's best epoch is reported, and the spread of those bests is the
    run-to-run noise of the quantity the criterion is written against.
    """
    print()
    print("=" * 96)
    print("2. the seed spread (the run-to-run noise)")
    print("=" * 96)
    for arm in ("qpe_0", "base_concat"):
        rows = []
        for f in sorted(seed_dir.glob(f"{arm}_s*_ep*.eval.json")):
            # qpe_0_s1_ep02.eval.json -> seed 1, epoch 2
            stem = f.name[: -len(".eval.json")]
            tail = stem.split("_s", 1)[1]
            seed, ep = tail.split("_ep")
            rep = load(f)
            t = rep["thresholds"]
            rows.append((seed, ep, {th: t[th]["csi"] for th in TIERS}))
        if not rows:
            print(f"\n{arm}: no seed evaluations found")
            continue
        print(f"\n--- {arm} ---")
        print(f"{'seed':>6}{'ep':>5}{'CSI@0.1':>10}{'CSI@0.5':>10}"
              f"{'CSI@2.0':>10}{'CSI@4.0':>10}   note")
        best = {}
        for seed, ep, t in sorted(rows, key=lambda r: (r[0], r[1])):
            note = "peak of this seed" if seed not in best else ""
            if seed not in best or t["2.0"] > best[seed][1]["2.0"]:
                best[seed] = (ep, t)
            print(f"{seed:>6}{ep:>5}{t['0.1']:>10.4f}{t['0.5']:>10.4f}"
                  f"{t['2.0']:>10.4f}{t['4.0']:>10.4f}   {note}")
        print("  the peak of each seed:")
        for seed in sorted(best):
            ep, t = best[seed]
            flag = "above the floor" if t["2.0"] > FLOOR else ""
            print(f"    seed {seed}: epoch {ep}, CSI@2.0 = {t['2.0']:.4f} "
                  f"{flag}")
        vals = [t["2.0"] for _, t in best.values()]
        if len(vals) >= 2 and min(vals) > 0:
            print(f"  the peak CSI@2.0 across seeds: min {min(vals):.4f} "
                  f"max {max(vals):.4f}, ratio {max(vals) / min(vals):.2f}x")
        if vals:
            print(f"  mean of the peaks: {sum(vals) / len(vals):.4f} "
                  f"({'above' if sum(vals) / len(vals) > FLOOR else 'below'} "
                  f"the floor {FLOOR})")


def _counts_of(path: Path, thr: str) -> np.ndarray:
    """The per-window contingency counts of one seed, as an (n, 3) array.

    The columns are hits, misses and false alarms -- the three numbers
    a critical success index is made of.
    """
    rows = load(path)["windows"]
    out = np.zeros((len(rows), 3))
    for i, w in enumerate(rows):
        c = w[thr]
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out


def _csi(counts: np.ndarray) -> float:
    """The critical success index of a set of contingency counts."""
    denom = counts[:, 0].sum() + counts[:, 1].sum() + counts[:, 2].sum()
    return counts[:, 0].sum() / denom if denom > 0 else float("nan")


def _seed_files(seed_dir: Path, arm: str, ep: str | None = None):
    """The per-window tables of one arm, optionally at one epoch."""
    out = []
    for f in sorted(seed_dir.glob(f"{arm}_s*_ep*.windows.json")):
        stem = f.name[: -len(".windows.json")]
        tail = stem.split("_s", 1)[1]
        seed, this_ep = tail.split("_ep")
        if ep is not None and this_ep != ep:
            continue
        out.append((seed, this_ep, f))
    return out


def bootstrap_pair(seed_dir: Path, a: str, b: str, thr: str = "2.0",
                   ep: str | None = None) -> None:
    """The sampling uncertainty and the training uncertainty, separately.

    Two things have to clear for a difference to be a difference, and
    they are not the same thing:

    1. the sampling uncertainty -- which evaluation windows happened to
       be in the test set.  The bootstrap resamples the windows, with
       the same indices for both arms, because both arms were scored on
       the same windows in the same order; a paired resample is the
       right one when the two arms share the windows.
    2. the training uncertainty -- which seed the run happened to get.
       That is the seed spread of section 2, and it is what the
       variance band is.

    The first is measured per seed, so that a difference that only
    holds for one seed is visible as such.  The second is measured by
    the two-level bootstrap: every resample draws a seed for each arm
    as well as a set of windows, which is the distribution of the
    difference for a randomly trained pair of runs.
    """
    print()
    print("=" * 96)
    label = f"3. the bootstrap: {a} minus {b} on CSI@{thr}"
    if ep is not None:
        label += f" at epoch {ep}"
    print(label)
    print("=" * 96)

    a_files = _seed_files(seed_dir, a, ep)
    b_files = _seed_files(seed_dir, b, ep)
    if not a_files or not b_files:
        print(f"{a}: no per-window tables found" if not a_files
              else f"{b}: no per-window tables found")
        return

    ca = [_counts_of(f, thr) for _, _, f in a_files]
    cb = [_counts_of(f, thr) for _, _, f in b_files]
    n = min(min(len(c) for c in ca), min(len(c) for c in cb))
    if n == 0:
        print("no windows to resample")
        return

    print(f"  windows per seed : {n}")
    print(f"  seeds per arm    : {len(ca)} / {len(cb)}")

    rng = np.random.default_rng(RNG_SEED)

    # ---- part one: the sampling uncertainty, per seed -------------------
    print()
    print(f"  {thr} -- the sampling uncertainty (windows resampled, seed fixed)")
    print(f"  {'pair':>12}{'observed':>12}{'95 % interval':>26}{'excludes 0':>13}")
    all_exclude = True
    for i, (xa, xb) in enumerate(zip(ca, cb)):
        point = _csi(xa[:n]) - _csi(xb[:n])
        diffs = np.empty(N_BOOT)
        for k in range(N_BOOT):
            idx = rng.integers(0, n, n)
            diffs[k] = _csi(xa[idx]) - _csi(xb[idx])
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        excludes = not (lo <= 0.0 <= hi)
        all_exclude = all_exclude and excludes
        print(f"  seed {i:>6}{point:>+12.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]"
              f"{('yes' if excludes else 'NO'):>13}")
    if len(ca) > 1:
        print(f"  every seed's interval excludes zero: "
              f"{'yes' if all_exclude else 'no'}")

    # ---- part two: the two-level bootstrap ------------------------------
    # A seed is drawn for each arm on every resample, so the interval
    # carries the run-to-run noise as well as the sampling noise.
    point_a = float(np.mean([_csi(c[:n]) for c in ca]))
    point_b = float(np.mean([_csi(c[:n]) for c in cb]))
    diffs = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, n, n)
        sa = ca[rng.integers(0, len(ca))]
        sb = cb[rng.integers(0, len(cb))]
        diffs[k] = _csi(sa[idx]) - _csi(sb[idx])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    print()
    print(f"  {thr} -- windows and seeds both resampled ({N_BOOT} draws)")
    print(f"  mean over seeds: {a} = {point_a:.4f}, {b} = {point_b:.4f}")
    print(f"  observed difference (seed means): {point_a - point_b:+.4f}")
    print(f"  95 % interval                     : [{lo:+.4f}, {hi:+.4f}]")
    if lo <= 0.0 <= hi:
        print("  verdict: the interval includes zero. The difference between")
        print("           these two configurations is not larger than the")
        print("           noise of choosing a seed and a set of windows.")
    else:
        print("  verdict: the interval excludes zero. The difference is")
        print("           larger than both sources of noise together.")

    # ---- the ratio, which is what the variance band is written in ------
    ratios = []
    for xa, xb in zip(ca, cb):
        num, den = _csi(xa[:n]), _csi(xb[:n])
        if num > 0 and den > 0:
            ratios.append(num / den)
    if len(ratios) > 1:
        print()
        print(f"  the ratio {a}/{b} per seed: "
              + ", ".join(f"{r:.2f}x" for r in ratios))
        print(f"  spread {max(ratios) / min(ratios):.2f}x -- compare with the")
        print(f"  measured variance band of 1.78x (MASTER section 4.10.3)")


def read_lead_times(curve_dir: Path) -> None:
    """How the score decays with the lead time, at each arm's peak.

    The peak value on its own does not say whether the model is
    extrapolating or repeating.  A model that repeats what it already
    saw scores well at the first lead time and falls off afterwards; a
    model that extrapolates holds its score.  The decay ratio between
    the first and the last lead time is what separates the two, so it
    is reported next to the peak.
    """
    print()
    print("=" * 96)
    print("4. the lead-time decay at each arm's peak")
    print("=" * 96)
    for arm in ("qpe_0", "qpe_05", "base_concat"):
        d = curve_dir / f"{arm}_eval"
        if not d.exists():
            continue
        rows = []
        for f in sorted(d.glob("epoch_*.eval.json")):
            epoch = epoch_of(f)
            if epoch is None:
                continue
            rep = load(f)
            leads = rep.get("lead_csi") or {}
            if not leads:
                continue
            rows.append((epoch, rep, leads))
        if not rows:
            print(f"\n{arm}: no lead-time scores found")
            continue
        peak = max(rows, key=lambda r: r[1]["thresholds"]["2.0"]["csi"])
        epoch, rep, leads = peak
        print(f"\n--- {arm} (peak epoch {epoch}, "
              f"CSI@2.0 = {rep['thresholds']['2.0']['csi']:.4f}) ---")
        print(f"{'lead':>6}{'CSI@0.1':>10}{'CSI@0.5':>10}{'CSI@2.0':>10}"
              f"{'CSI@4.0':>10}")
        keys = sorted(leads, key=lambda k: int(k))
        for k in keys:
            t = leads[k]
            print(f"{k:>6}{t['0.1']:>10.4f}{t['0.5']:>10.4f}"
                  f"{t['2.0']:>10.4f}{t['4.0']:>10.4f}")
        for th in TIERS:
            first, last = leads[keys[0]][th], leads[keys[-1]][th]
            if first > 0:
                print(f"  CSI@{th}: lead {keys[0]} = {first:.4f} -> "
                      f"lead {keys[-1]} = {last:.4f}, "
                      f"decay {first / last:.2f}x")


def read_p3(root: Path) -> None:
    """The observation-assimilation lag sweep of route P3.

    The four arms consume the ground observation with a different delay,
    so the curve says what is left of the benefit when the observation
    is late.  The check that goes with it is the epoch-0 training loss:
    if the four arms agree there, the observation still does not reach
    the training and the sweep measures nothing.
    """
    p3 = root / "route_p3"
    if not p3.exists():
        p3 = root.parent / "route_p3"
    if not p3.exists():
        return
    print()
    print("=" * 96)
    print("5. the assimilation lag sweep (route P3)")
    print("=" * 96)
    arms = [("assim_lag0", "no delay"),
            ("assim_lag2", "12 min late"),
            ("assim_lag4", "24 min late"),
            ("assim_lag11", "no observation")]
    header = (f"{'arm':16s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
              f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}{'n':>7s}")
    print(header)
    print("-" * len(header))
    for name, label in arms:
        path = p3 / f"{name}.eval.json"
        if not path.exists():
            print(f"{label:16s}(not finished)")
            continue
        rep = load(path)
        t = rep["thresholds"]
        g = lambda th, key: t.get(th, {}).get(key, float("nan"))
        print(f"{label:16s}"
              + "".join(f"{g(th, 'csi'):9.3f}" for th in TIERS)
              + f"{g('0.5', 'far'):9.3f}{g('0.5', 'bias'):9.3f}"
              + f"{rep.get('n_samples', 0):7d}")

    losses = {}
    for name, _ in arms:
        log = p3 / f"{name}.log"
        if not log.exists():
            continue
        picked = {}
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith("epoch ") and ": loss=" in line:
                try:
                    epoch = int(line.split()[1].rstrip(":"))
                    picked[epoch] = float(line.split("loss=")[1].split()[0])
                except (ValueError, IndexError):
                    continue
        losses[name] = picked
        if picked:
            print(f"  {name:16s} ep{min(picked)}={picked[min(picked)]:.4f}"
                  f"  ep{max(picked)}={picked[max(picked)]:.4f}")
    ep0 = [v.get(0) for v in losses.values() if v.get(0) is not None]
    if len(ep0) >= 2:
        spread = max(ep0) - min(ep0)
        if spread < 1e-3:
            print("\n  FAIL: the epoch-0 losses still agree to a thousandth.")
            print("  The observation does not reach the training, so this")
            print("  sweep measures nothing.")
        else:
            print(f"\n  PASS: the epoch-0 losses differ by {spread:.4f}, so "
                  f"the lag")
            print("        changes what the model trains on.")
            print("  the order of the losses by lag (more usable observation")
            print("  should cost less):")
            ordered = sorted(
                ((v[0], name) for name, v in losses.items() if 0 in v))
            for value, name in ordered:
                print(f"    {name:16s} {value:.4f}")

    # The curve itself: the score against the lag, in the order of the
    # lag.  This is what the sweep is for -- how much of the benefit
    # survives when the observation is late.
    print()
    print("  the lag curve (CSI@2.0 against the delay):")
    print(f"  {'lag':>10}{'minutes late':>15}{'CSI@2.0':>10}{'CSI@0.5':>10}")
    curve = []
    for name, label in arms:
        path = p3 / f"{name}.eval.json"
        if not path.exists():
            continue
        rep = load(path)
        t = rep["thresholds"]
        lag = int(name.split("lag")[-1])
        curve.append((lag, label, t["2.0"]["csi"], t["0.5"]["csi"]))
    for lag, label, csi2, csi05 in sorted(curve):
        print(f"  {lag:>10}{lag * 6:>15}{csi2:>10.4f}{csi05:>10.4f}")
    if len(curve) >= 2:
        # The largest lag is the arm that has no observation at all, and
        # the smallest lag is the one that has it all.  Reading them the
        # other way round turns a gain into a loss, so the two ends are
        # named explicitly rather than by position.
        no_obs = max(curve, key=lambda c: c[0])
        full = min(curve, key=lambda c: c[0])
        if full[2] > 0:
            print(f"  the score is {full[2]:.4f} with the observation "
                  f"available ({full[1]})")
            print(f"  and {no_obs[2]:.4f} with no observation at all "
                  f"({no_obs[1]})")
            print(f"  the observation is worth {full[2] / no_obs[2]:.2f}x "
                  f"at CSI@2.0")
        middle = [c for c in curve if c is not full and c is not no_obs]
        for lag, label, csi2, csi05 in middle:
            print(f"  a delay of {lag * 6} min leaves {csi2:.4f}, "
                  f"which is {'below' if csi2 < no_obs[2] else 'above'} "
                  f"the no-observation arm")


def read_tail(curve_dir: Path, reference: Path | None) -> None:
    """The long lead times, against the persistence reference.

    The overall score does not say whether a model is extrapolating or
    repeating what it already saw.  At the long lead times the
    persistence reference is the thing to beat: it copies the last
    observed reflectivity and holds it, so it has no skill at all at
    lead nine beyond what the last frame carries.  A model that is
    above it there is doing something the reference cannot do.

    This is the comparison the master document claims for ``qpe_0`` --
    that it is the only arm in the matrix that beats persistence at
    L8 and L9 -- so the claim is checked here rather than repeated.
    """
    print()
    print("=" * 96)
    print("4b. the long lead times against the persistence reference")
    print("=" * 96)
    if reference is None or not reference.exists():
        print("  no persistence reference found")
        return
    ref = load(reference)
    ref_leads = ref.get("lead_scores") or {}
    if not ref_leads:
        print("  the reference has no per-lead scores")
        return

    for arm in ("qpe_0", "qpe_05", "base_concat"):
        d = curve_dir / f"{arm}_eval"
        if not d.exists():
            continue
        rows = []
        for f in sorted(d.glob("epoch_*.eval.json")):
            epoch = epoch_of(f)
            if epoch is None:
                continue
            rep = load(f)
            leads = rep.get("lead_csi") or {}
            if leads:
                rows.append((epoch, rep["thresholds"]["2.0"]["csi"], leads))
        if not rows:
            continue

        # The peak epoch is the one the criterion is written against.
        peak = max(rows, key=lambda r: r[1])
        epoch, csi2, leads = peak

        # How many epochs clear the reference at L9 at all.
        above9 = sum(1 for _, _, ld in rows
                     if ld.get("9", {}).get("2.0", 0.0)
                     > ref_leads["9"]["2.0"]["csi"])

        print(f"\n--- {arm} ---")
        print(f"  the peak is epoch {epoch} (CSI@2.0 = {csi2:.4f})")
        print(f"  {'lead':>6}{'this arm':>11}{'persistence':>13}{'margin':>10}")
        for lead in map(str, range(10)):
            if lead not in leads or lead not in ref_leads:
                continue
            mine = leads[lead]["2.0"]
            theirs = ref_leads[lead]["2.0"]["csi"]
            print(f"  {lead:>6}{mine:>11.4f}{theirs:>13.4f}{mine - theirs:>+10.4f}")
        print(f"  epochs above the reference at L9: {above9} / {len(rows)}")

        # The decay ratio: how much of the score survives to the far
        # lead.  A model that repeats what it saw falls off quickly; one
        # that extrapolates holds on.
        l0, l9 = leads["0"]["2.0"], leads["9"]["2.0"]
        if l9 > 0:
            print(f"  decay from L0 to L9: {l0 / l9:.2f}x")
        rl0 = ref_leads["0"]["2.0"]["csi"]
        rl9 = ref_leads["9"]["2.0"]["csi"]
        if rl9 > 0:
            print(f"  the persistence reference decays {rl0 / rl9:.2f}x over "
                  f"the same span")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/wf/radarNWP/runs/route_p2"
    )
    curve_dir = root / "curve"
    seed_dir = root / "seeds"
    reference = root.parent / "reference" / "persistence.eval.json"
    read_curve(curve_dir)
    read_lead_times(curve_dir)
    read_tail(curve_dir, reference)
    read_seeds(seed_dir)
    # The bootstrap runs at every epoch the seeds were scored at.  The
    # peak epoch is the one the criterion is written against; the last
    # epoch is the one the Part-1 curve is compared with.
    for ep in ("00", "19"):
        bootstrap_pair(seed_dir, "qpe_0", "base_concat", ep=ep)
    read_p3(root)


if __name__ == "__main__":
    main()
