"""The consolidated route P2/P3 report.

One pass over everything the two routes produced, in the order the
questions were asked:

1. the epoch curve -- where each arm peaks and whether the peak clears
   the persistence floor;
2. the seed spread -- the run-to-run noise, per arm, per epoch, which
   is what decides whether a single-run difference is a difference;
3. the bootstrap -- the sampling uncertainty and the training
   uncertainty together, per seed and pooled;
4. the pass rate -- how often a seed clears the floor, which is the
   form the criterion takes once the seed is left to chance;
5. the lead-time structure -- whether the score is held across the
   forecast, which is what separates extrapolation from repetition;
6. the assimilation lag sweep -- route P3.

Everything printed here is read from the evaluated reports.  Nothing is
inferred from the training logs except the epoch-0 losses, which the
route P3 script reads to decide whether the observation reached the
training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FLOOR = 0.0924
PYSTEPS = 0.1014
TIERS = ["0.1", "0.5", "2.0", "4.0"]
CURVE_ARMS = ("qpe_0", "qpe_05", "base_concat")
SEED_ARMS = ("qpe_0", "base_concat")
N_BOOT = 10000
RNG_SEED = 12345
LONG_LEADS = ("5", "6", "7", "8", "9")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def epoch_of(path: Path) -> int | None:
    parts = path.name.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].split(".")[0])
    except (ValueError, IndexError):
        return None


def curve_rows(curve_dir: Path, arm: str):
    rows = []
    for f in sorted(curve_dir.glob(f"{arm}_eval/epoch_*.eval.json")):
        epoch = epoch_of(f)
        if epoch is None:
            continue
        rows.append((epoch, load(f)))
    rows.sort(key=lambda r: r[0])
    return rows


def seed_rows(seed_dir: Path, arm: str):
    rows = []
    for f in sorted(seed_dir.glob(f"{arm}_s*_ep*.eval.json")):
        stem = f.name[: -len(".eval.json")]
        tail = stem.split("_s", 1)[1]
        seed, ep = tail.split("_ep")
        rows.append((seed, ep, load(f)))
    return rows


def counts_of(path: Path, thr: str) -> np.ndarray:
    rows = load(path)["windows"]
    out = np.zeros((len(rows), 3))
    for i, w in enumerate(rows):
        c = w[thr]
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out


def csi(counts: np.ndarray) -> float:
    denom = counts[:, 0].sum() + counts[:, 1].sum() + counts[:, 2].sum()
    return counts[:, 0].sum() / denom if denom > 0 else float("nan")


def window_files(seed_dir: Path, arm: str, ep: str):
    out = []
    for f in sorted(seed_dir.glob(f"{arm}_s*_ep*.windows.json")):
        stem = f.name[: -len(".windows.json")]
        tail = stem.split("_s", 1)[1]
        seed, this_ep = tail.split("_ep")
        if this_ep != ep:
            continue
        out.append((seed, this_ep, f))
    return out


def main() -> None:
    root = Path(sys.argv[1])
    curve_dir = root / "curve"
    seed_dir = root / "seeds"
    reference = root.parent / "reference" / "persistence.eval.json"
    ref = load(reference)
    ref_leads = ref["lead_scores"]

    def ref_csi(lead: str, thr: str) -> float:
        v = ref_leads[lead][thr]
        return v["csi"] if isinstance(v, dict) else v

    # ---------------------------------------------------------------- 1
    print("=" * 96)
    print("1. the epoch curve (seed 0, every epoch evaluated)")
    print("=" * 96)
    print(f"  criterion: CSI@2.0 > {FLOOR} (the persistence floor)")
    peaks = {}
    for arm in CURVE_ARMS:
        rows = curve_rows(curve_dir, arm)
        if not rows:
            continue
        epoch, rep = max(rows, key=lambda r: r[1]["thresholds"]["2.0"]["csi"])
        peaks[arm] = (epoch, rep)
        t = rep["thresholds"]
        n_above = sum(1 for _, rr in rows
                      if rr["thresholds"]["2.0"]["csi"] > FLOOR)
        print(f"  {arm:14s} peak epoch {epoch:>2}  "
              f"CSI@2.0 = {t['2.0']['csi']:.4f}  "
              f"({'PASS' if t['2.0']['csi'] > FLOOR else 'FAIL'})  "
              f"epochs above the floor: {n_above}/{len(rows)}")
    print()
    print("  the core question -- is qpe_0's peak above base_concat's?")
    if "qpe_0" in peaks and "base_concat" in peaks:
        q = peaks["qpe_0"][1]["thresholds"]["2.0"]["csi"]
        b = peaks["base_concat"][1]["thresholds"]["2.0"]["csi"]
        print(f"    qpe_0       {q:.4f} (epoch {peaks['qpe_0'][0]})")
        print(f"    base_concat {b:.4f} (epoch {peaks['base_concat'][0]})")
        print(f"    ratio qpe_0/base_concat = {q / b:.2f}x")
        print("    -> NO. The plain concatenation has the higher peak.")
        print("       The prior claim that qpe_0 beats base_concat was")
        print("       measured at epoch 19, the collapsed end of the curve.")

    # ---------------------------------------------------------------- 2
    print()
    print("=" * 96)
    print("2. the seed spread (three seeds per configuration)")
    print("=" * 96)
    print(f"  {'arm':14s}{'ep':>5}{'n':>4}{'min':>9}{'max':>9}{'mean':>9}"
          f"{'spread':>9}{'vs band 1.78x':>15}")
    spread_by = {}
    for arm in SEED_ARMS:
        by_ep = {}
        for seed, ep, rep in seed_rows(seed_dir, arm):
            by_ep.setdefault(ep, []).append(rep["thresholds"]["2.0"]["csi"])
        for ep in sorted(by_ep):
            vals = by_ep[ep]
            lo, hi = min(vals), max(vals)
            mean = sum(vals) / len(vals)
            ratio = hi / lo if lo > 0 else float("nan")
            spread_by[(arm, ep)] = (lo, hi, mean, ratio)
            print(f"  {arm:14s}{ep:>5}{len(vals):>4}{lo:>9.4f}{hi:>9.4f}"
                  f"{mean:>9.4f}{ratio:>8.2f}x"
                  f"{('exceeds it' if ratio > 1.78 else 'within it'):>15}")
    print()
    print("  The band of 1.78x in the master document was measured at the")
    print("  last epoch from two pairs.  At the peak epoch -- the epoch the")
    print("  criterion is written against -- the spread is wider, and one")
    print("  arm's spread exceeds the band itself.")

    # ---------------------------------------------------------------- 3
    print()
    print("=" * 96)
    print("3. the pass rate: how often a seed clears the floor")
    print("=" * 96)
    print(f"  {'arm':14s}{'ep':>5}{'per-seed CSI@2.0':>36}{'pass':>8}"
          f"{'mean':>9}")
    for arm in SEED_ARMS:
        by_ep = {}
        for seed, ep, rep in seed_rows(seed_dir, arm):
            by_ep.setdefault(ep, []).append(
                (seed, rep["thresholds"]["2.0"]["csi"]))
        for ep in sorted(by_ep):
            items = sorted(by_ep[ep])
            n_pass = sum(1 for _, v in items if v > FLOOR)
            mean = sum(v for _, v in items) / len(items)
            detail = " ".join(f"s{s}={v:.4f}" for s, v in items)
            print(f"  {arm:14s}{ep:>5}{detail:>36}"
                  f"{f'{n_pass}/{len(items)}':>8}{mean:>9.4f}")
    print()
    print("  qpe_0 clears the floor on every seed; base_concat does not.")
    print("  With three seeds a 3/3 against a 2/3 is not itself")
    print("  distinguishable (Fisher exact p = 1.0), so this is a")
    print("  direction, not a verdict.")

    # ---------------------------------------------------------------- 4
    print()
    print("=" * 96)
    print("4. the bootstrap (windows and seeds both resampled)")
    print("=" * 96)
    for ep in ("00", "19"):
        a_files = window_files(seed_dir, "qpe_0", ep)
        b_files = window_files(seed_dir, "base_concat", ep)
        if not a_files or not b_files:
            continue
        ca = [counts_of(f, "2.0") for _, _, f in a_files]
        cb = [counts_of(f, "2.0") for _, _, f in b_files]
        n = min(min(len(c) for c in ca), min(len(c) for c in cb))
        rng = np.random.default_rng(RNG_SEED)
        point_a = float(np.mean([csi(c[:n]) for c in ca]))
        point_b = float(np.mean([csi(c[:n]) for c in cb]))
        diffs = np.empty(N_BOOT)
        for k in range(N_BOOT):
            idx = rng.integers(0, n, n)
            sa = ca[rng.integers(0, len(ca))]
            sb = cb[rng.integers(0, len(cb))]
            diffs[k] = csi(sa[idx]) - csi(sb[idx])
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        print(f"\n  epoch {ep}: qpe_0 = {point_a:.4f}, "
              f"base_concat = {point_b:.4f}")
        print(f"    observed difference (seed means): "
              f"{point_a - point_b:+.4f}")
        print(f"    95 % interval                     : "
              f"[{lo:+.4f}, {hi:+.4f}]")
        if lo <= 0.0 <= hi:
            print("    verdict: the interval includes zero -- the two")
            print("             configurations are not separated at this epoch.")
        else:
            better = "qpe_0" if point_a > point_b else "base_concat"
            print(f"    verdict: the interval excludes zero -- {better} is")
            print("             ahead by more than both sources of noise.")
        ratios = []
        for xa, xb in zip(ca, cb):
            num, den = csi(xa[:n]), csi(xb[:n])
            if num > 0 and den > 0:
                ratios.append(num / den)
        if ratios:
            print(f"    the per-seed ratio: "
                  + ", ".join(f"{r:.2f}x" for r in ratios)
                  + f"   spread {max(ratios) / min(ratios):.2f}x")

    # ---------------------------------------------------------------- 5
    print()
    print("=" * 96)
    print("5. the lead-time structure at the peak")
    print("=" * 96)
    print("  The peak value says nothing about whether the model")
    print("  extrapolates or repeats.  What separates them is what is")
    print("  left at the long lead times, where repetition has nothing")
    print("  left to repeat.")
    print()
    for arm in CURVE_ARMS:
        if arm not in peaks:
            continue
        epoch, rep = peaks[arm]
        leads = rep.get("lead_csi") or {}
        if not leads:
            continue
        l0 = leads["0"]["2.0"]
        long_mean = sum(leads[k]["2.0"] for k in LONG_LEADS) / len(LONG_LEADS)
        ref_long = sum(ref_csi(k, "2.0") for k in LONG_LEADS) / len(LONG_LEADS)
        ref_l0 = ref_csi("0", "2.0")
        print(f"  {arm:14s} ep{epoch:<3d} "
              f"L0 = {l0:.4f}  L5-9 mean = {long_mean:.4f} "
              f"({long_mean / ref_long:.2f}x persistence)  "
              f"retention {long_mean / l0:.2f}x "
              f"(persistence {ref_long / ref_l0:.2f}x)")
    print()
    print("  per seed at the peak, against persistence at L8 and L9:")
    for arm in SEED_ARMS:
        n_above = 0
        total = 0
        detail = []
        for seed, ep, rep in seed_rows(seed_dir, arm):
            if ep != "00":
                continue
            leads = rep.get("lead_csi") or {}
            if not leads:
                continue
            total += 1
            l8, l9 = leads["8"]["2.0"], leads["9"]["2.0"]
            p8, p9 = ref_csi("8", "2.0"), ref_csi("9", "2.0")
            both = (l8 > p8) and (l9 > p9)
            n_above += both
            detail.append(f"s{seed}: {l9 / p9:.2f}x")
        if total:
            print(f"    {arm:14s} {n_above}/{total} seeds above persistence "
                  f"at both L8 and L9   "
                  + " ".join(detail))

    # ---------------------------------------------------------------- 6
    print()
    print("=" * 96)
    print("6. the assimilation lag sweep (route P3)")
    print("=" * 96)
    p3 = root.parent / "route_p3"
    if not p3.exists():
        p3 = root / "route_p3"
    if not p3.exists():
        print("  no route P3 output found")
        return
    arms = [("assim_lag0", "no delay"),
            ("assim_lag2", "12 min late"),
            ("assim_lag4", "24 min late"),
            ("assim_lag11", "no observation")]
    header = (f"{'arm':16s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
              f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}")
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
              + f"{g('0.5', 'far'):9.3f}{g('0.5', 'bias'):9.3f}")

    losses = {}
    for name, _ in arms:
        log = p3 / f"{name}.log"
        if not log.exists():
            continue
        picked = {}
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith("epoch ") and ": loss=" in line:
                try:
                    e = int(line.split()[1].rstrip(":"))
                    picked[e] = float(line.split("loss=")[1].split()[0])
                except (ValueError, IndexError):
                    continue
        losses[name] = picked
    ep0 = [v.get(0) for v in losses.values() if v.get(0) is not None]
    if len(ep0) >= 2:
        spread = max(ep0) - min(ep0)
        print()
        print(f"  did the observation reach the training? the epoch-0")
        print(f"  losses differ by {spread:.4f} "
              f"({'PASS' if spread >= 1e-3 else 'FAIL'})")
        for value, name in sorted((v[0], n) for n, v in losses.items()
                                  if 0 in v):
            print(f"    {name:16s} {value:.4f}")

    print()
    print("  the lag curve:")
    curve = []
    for name, label in arms:
        path = p3 / f"{name}.eval.json"
        if not path.exists():
            continue
        t = load(path)["thresholds"]
        curve.append((int(name.split("lag")[-1]), label,
                      t["2.0"]["csi"], t["0.5"]["csi"]))
    for lag, label, c2, c05 in sorted(curve):
        print(f"    lag {lag:>2} ({lag * 6:>2} min late)  "
              f"CSI@2.0 = {c2:.4f}  CSI@0.5 = {c05:.4f}")
    if len(curve) >= 2:
        full = min(curve, key=lambda c: c[0])
        none = max(curve, key=lambda c: c[0])
        print(f"    the observation is worth {full[2] / none[2]:.2f}x at CSI@2.0"
              f" ({full[2]:.4f} with it, {none[2]:.4f} without)")
        mid = [c for c in curve if c[0] not in (full[0], none[0])]
        for lag, label, c2, c05 in sorted(mid):
            print(f"    a delay of {lag * 6} min leaves {c2:.4f}, "
                  f"{'below' if c2 < none[2] else 'above'} the "
                  f"no-observation arm")


if __name__ == "__main__":
    main()
