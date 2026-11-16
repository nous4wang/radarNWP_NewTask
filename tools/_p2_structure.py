"""Read the route P2 curve: the peak, and the structure that separates arms.

The analyzer answers "where is the peak and does it clear the floor".
This answers the question underneath it: the arms differ in *shape*, not
just in height, and the shape is what separates extrapolation from
repetition.  Three views of the same checkpoint set:

1. the peak, per arm, per threshold;
2. the lead-time profile at the peak, against the persistence reference,
   with the decay ratio computed on the same span;
3. the long lead times (L8, L9), which is where repetition dies and
   extrapolation does not.

Everything here is read from the evaluated reports, so every number is
one the evaluation actually produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FLOOR = 0.0924
PYSTEPS = 0.1014
TIERS = ["0.1", "0.5", "2.0", "4.0"]
ARMS = ("qpe_0", "qpe_05", "base_concat")


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


def curve_rows(curve_dir: Path, arm: str) -> list[tuple[int, dict]]:
    d = curve_dir / f"{arm}_eval"
    rows = []
    for f in sorted(d.glob("epoch_*.eval.json")):
        epoch = epoch_of(f)
        if epoch is None:
            continue
        rows.append((epoch, load(f)))
    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    root = Path(sys.argv[1])
    curve_dir = root / "curve"
    reference = root.parent / "reference" / "persistence.eval.json"

    ref = load(reference) if reference.exists() else None
    ref_leads = (ref or {}).get("lead_scores") or {}

    print("=" * 96)
    print("A. the peak of each arm (seed 0, every epoch evaluated)")
    print("=" * 96)
    print(f"{'arm':14s}{'peak ep':>9}" + "".join(f"{'CSI@' + t:>10}" for t in TIERS)
          + f"{'floor':>9}{'pysteps':>10}")
    peaks = {}
    for arm in ARMS:
        rows = curve_rows(curve_dir, arm)
        if not rows:
            print(f"{arm:14s}(no evaluations)")
            continue
        epoch, rep = max(rows, key=lambda r: r[1]["thresholds"]["2.0"]["csi"])
        peaks[arm] = (epoch, rep)
        t = rep["thresholds"]
        print(f"{arm:14s}{epoch:>9}"
              + "".join(f"{t[th]['csi']:>10.4f}" for th in TIERS)
              + f"{FLOOR:>9.4f}{PYSTEPS:>10.4f}")
    print()
    print(f"  criterion (CSI@2.0 > {FLOOR}, the persistence floor):")
    for arm, (epoch, rep) in peaks.items():
        v = rep["thresholds"]["2.0"]["csi"]
        print(f"    {arm:14s} epoch {epoch:>2}  CSI@2.0 = {v:.4f}  "
              f"{'PASS' if v > FLOOR else 'FAIL'}"
              f"{'  (also above pysteps)' if v > PYSTEPS else ''}")
    if len(peaks) >= 2:
        q = peaks["qpe_0"][1]["thresholds"]["2.0"]["csi"]
        b = peaks["base_concat"][1]["thresholds"]["2.0"]["csi"]
        print(f"  qpe_0 / base_concat at the peak = {q / b:.2f}x "
              f"(the measured variance band is 1.78x)")

    print()
    print("=" * 96)
    print("B. the lead-time profile at the peak, against persistence")
    print("=" * 96)
    for arm in ARMS:
        if arm not in peaks:
            continue
        epoch, rep = peaks[arm]
        leads = rep.get("lead_csi") or {}
        if not leads:
            continue
        print(f"\n--- {arm} (peak epoch {epoch}, "
              f"CSI@2.0 = {rep['thresholds']['2.0']['csi']:.4f}) ---")
        print(f"  {'lead':>5}{'CSI@2.0':>10}{'persistence':>13}{'ratio':>9}")
        for k in sorted(leads, key=int):
            mine = leads[k]["2.0"]
            theirs = ref_leads.get(k, {}).get("2.0", {}).get("csi")
            ratio = f"{mine / theirs:.2f}x" if theirs else "-"
            t = f"{theirs:.4f}" if theirs is not None else "-"
            print(f"  {k:>5}{mine:>10.4f}{t:>13}{ratio:>9}")
        if "0" in leads and "9" in leads:
            l0, l9 = leads["0"]["2.0"], leads["9"]["2.0"]
            print(f"  decay L0 -> L9: {l0 / l9:.2f}x", end="")
            if ref_leads and "0" in ref_leads and "9" in ref_leads:
                r0 = ref_leads["0"]["2.0"]["csi"]
                r9 = ref_leads["9"]["2.0"]["csi"]
                print(f"   (persistence decays {r0 / r9:.2f}x over the same span)"
                      f"   -> {'SLOWER than persistence' if l0 / l9 < r0 / r9 else 'faster than persistence'}")
            else:
                print()

    print()
    print("=" * 96)
    print("C. the long lead times: where repetition dies")
    print("=" * 96)
    print(f"{'arm':14s}{'L8 / pers':>18}{'L9 / pers':>18}"
          f"{'L9 CSI@2.0':>14}{'ep above L9':>14}")
    for arm in ARMS:
        if arm not in peaks:
            continue
        epoch, rep = peaks[arm]
        leads = rep.get("lead_csi") or {}
        rows = curve_rows(curve_dir, arm)
        if not leads or not ref_leads:
            continue
        r8 = ref_leads["8"]["2.0"]["csi"]
        r9 = ref_leads["9"]["2.0"]["csi"]
        above = sum(1 for _, rr in rows
                    if (rr.get("lead_csi") or {}).get("9", {}).get("2.0", 0.0) > r9)
        print(f"{arm:14s}"
              f"{leads['8']['2.0']:>10.4f} ({leads['8']['2.0'] / r8:.2f}x)"
              f"{leads['9']['2.0']:>10.4f} ({leads['9']['2.0'] / r9:.2f}x)"
              f"{leads['9']['2.0']:>14.4f}{above:>10d} /{len(rows):>3d}")

    print()
    print("=" * 96)
    print("D. the epoch at which each arm is at its best, and how fast it dies")
    print("=" * 96)
    for arm in ARMS:
        rows = curve_rows(curve_dir, arm)
        if not rows:
            continue
        epoch, rep = peaks[arm]
        peak = rep["thresholds"]["2.0"]["csi"]
        last = rows[-1][1]["thresholds"]["2.0"]["csi"]
        n_above = sum(1 for _, rr in rows
                      if rr["thresholds"]["2.0"]["csi"] > FLOOR)
        print(f"  {arm:14s} peak {peak:.4f} @ ep{epoch:<3d} "
              f"-> ep{rows[-1][0]:<3d} {last:.4f} "
              f"({peak / last:.1f}x loss)   epochs above the floor: "
              f"{n_above}/{len(rows)}")


if __name__ == "__main__":
    main()
