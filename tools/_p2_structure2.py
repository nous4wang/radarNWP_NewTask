"""The structural comparison: the peak is not the whole story.

Two arms can have different peak heights and still be different in kind.
A model that repeats what it already saw scores well at the first lead
time and falls off; a model that extrapolates holds on.  The shape is
what separates them, and it is measured here on the same checkpoints
the peak was measured on.

Three quantities, all read from the evaluated reports:

1. the long-lead score -- the mean over L5 to L9, the part of the
   forecast that repetition cannot serve;
2. the retention ratio -- the long-lead score as a fraction of the
   score at L0, which is the "did it keep anything" number;
3. the calibration at the heavy threshold -- BIAS at 2.0 mm, which
   separates a model that places heavy rain from one that simply
   spreads the same total rainfall around.

The persistence reference is on the same scale, so it is the control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FLOOR = 0.0924
ARMS = ("qpe_0", "qpe_05", "base_concat")
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


def peak_row(curve_dir: Path, arm: str):
    rows = []
    for f in sorted(curve_dir.glob(f"{arm}_eval/epoch_*.eval.json")):
        epoch = epoch_of(f)
        if epoch is None:
            continue
        rows.append((epoch, load(f)))
    if not rows:
        return None
    return max(rows, key=lambda r: r[1]["thresholds"]["2.0"]["csi"])


def main() -> None:
    root = Path(sys.argv[1])
    curve_dir = root / "curve"
    reference = root.parent / "reference" / "persistence.eval.json"
    ref = load(reference)
    ref_leads = ref["lead_scores"]

    def long_mean(leads: dict) -> float:
        # The persistence reference stores a full report per lead time;
        # the model's reports store the score directly.
        vals = []
        for k in LONG_LEADS:
            if k not in leads:
                continue
            v = leads[k]["2.0"]
            vals.append(v["csi"] if isinstance(v, dict) else v)
        return sum(vals) / len(vals) if vals else float("nan")

    ref_long = long_mean(ref_leads)
    ref_l0 = ref_leads["0"]["2.0"]["csi"]

    print("=" * 100)
    print("1. the peak and the long-lead score, at the peak epoch")
    print("=" * 100)
    print(f"  {'arm':14s}{'ep':>4}{'peak CSI@2.0':>15}{'L5-9 mean':>12}"
          f"{'vs pers':>10}{'retention':>12}{'pers retention':>17}")
    rows = {}
    for arm in ARMS:
        got = peak_row(curve_dir, arm)
        if got is None:
            continue
        epoch, rep = got
        leads = rep.get("lead_csi") or {}
        if not leads:
            continue
        lm = long_mean(leads)
        l0 = leads["0"]["2.0"]
        ret = lm / l0 if l0 else float("nan")
        ref_ret = ref_long / ref_l0
        rows[arm] = (epoch, rep, lm, ret)
        print(f"  {arm:14s}{epoch:>4}{rep['thresholds']['2.0']['csi']:>15.4f}"
              f"{lm:>12.4f}{lm / ref_long:>9.2f}x{ret:>11.2f}x{ref_ret:>16.2f}x")
    print(f"  {'persistence':14s}{'-':>4}{ref['thresholds']['2.0']['csi']:>15.4f}"
          f"{ref_long:>12.4f}{1.0:>9.2f}x{ref_ret:>11.2f}x")

    print()
    print("=" * 100)
    print("2. calibration at the heavy threshold, at the peak epoch")
    print("=" * 100)
    print(f"  {'arm':14s}{'ep':>4}{'BIAS@0.5':>11}{'BIAS@2.0':>11}"
          f"{'FAR@0.5':>10}{'FAR@2.0':>10}{'hits@2.0':>12}")
    for arm, (epoch, rep, _, _) in rows.items():
        t = rep["thresholds"]
        print(f"  {arm:14s}{epoch:>4}{t['0.5']['bias']:>11.4f}"
              f"{t['2.0']['bias']:>11.4f}{t['0.5']['far']:>10.4f}"
              f"{t['2.0']['far']:>10.4f}{t['2.0']['hits']:>12.0f}")
    t = ref["thresholds"]
    print(f"  {'persistence':14s}{'-':>4}{t['0.5']['bias']:>11.4f}"
          f"{t['2.0']['bias']:>11.4f}{t['0.5']['far']:>10.4f}"
          f"{t['2.0']['far']:>10.4f}{t['2.0']['hits']:>12.0f}")

    print()
    print("=" * 100)
    print("3. is the peak the same kind of thing as the persistence floor?")
    print("=" * 100)
    print("  A peak with BIAS@0.5 near 1.0 at the light threshold and")
    print("  BIAS@2.0 near 1.0 at the heavy one is placing the rain")
    print("  about right in total but not in the right places -- which is")
    print("  what persistence does.  A peak with BIAS@2.0 far from 1.0")
    print("  is either missing heavy rain or inventing it.")
    print()
    for arm, (epoch, rep, _, _) in rows.items():
        t = rep["thresholds"]
        b2 = t["2.0"]["bias"]
        verdict = ("places the heavy rain about right"
                   if 0.8 <= b2 <= 1.25 else
                   "over-predicts heavy rain" if b2 > 1.25 else
                   "under-predicts heavy rain")
        print(f"    {arm:14s} ep{epoch:<3d} BIAS@2.0 = {b2:.4f}  -> {verdict}")
    # The reference's own calibration, read from the reference and not
    # from the last arm's report.
    rt = ref["thresholds"]
    print(f"    {'persistence':14s}        BIAS@2.0 = {rt['2.0']['bias']:.4f}"
          f"  -> the reference itself")
    print(f"    {'persistence':14s}        BIAS@0.5 = {rt['0.5']['bias']:.4f}")


if __name__ == "__main__":
    main()
