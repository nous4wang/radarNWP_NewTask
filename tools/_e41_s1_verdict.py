#!/usr/bin/env python3
"""E41 -- read the structure metrics and apply the S1 criterion (read-only).

The input is the JSON that ``tools/_e41_struct_metrics.py`` wrote: the
median over the (window, lead) frames of every structure quantity, for the
forecast field and for the truth field, per arm and per seed.

S1 (frozen 2026-11-09 09:29) asks for four quantities relative to the
same-batch B0:

    BIAS@2.0            direction positive, absolute change >= +0.10
    tail_max_over_p99   must rise, must not fall
    zero_frac           must move TOWARD the label's 0.912
    kernel_px_gt_p99    must rise, must not fall

and the four must point the SAME way for the mechanism chain to hold.

The label's reference values are the section 2 medians: tail_max_over_p99
= 4.0, zero_frac = 0.912, kernel_px_gt_p99 = 108.8.  The forecast's
"dry fraction" is measured at the 0.1 mm business threshold
(``dry_frac_lt_0p1``), because the probe's own ``zero_frac`` expression
counts the pixels at the field minimum, which is a different thing on a
forecast that has no hard zero floor; the truth field is measured the same
way, so the comparison is like with like.

Nothing here trains, evaluates or writes a score.  It only reads the JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The section 2 label-side medians (E41 section 2.1).
LABEL_TAIL_MAX_OVER_P99 = 4.0
LABEL_DRY_FRAC = 0.912
LABEL_KERNEL_PX = 108.8

#: The S1 threshold on the BIAS@2.0 change.
BIAS_DELTA_MIN = 0.10


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--struct", required=True)
    ap.add_argument("--scores", required=True,
                    help="the score JSON written by _e41_collect_scores.py")
    ap.add_argument("--control", default="B0")
    args = ap.parse_args()

    st = json.loads(Path(args.struct).read_text(encoding="utf-8"))
    tags = st["tags"]

    # The scores, read back from the evaluator's own JSON.
    scores: dict[str, dict] = {}
    score_path = Path(args.scores)
    if score_path.exists():
        blob = json.loads(score_path.read_text(encoding="utf-8"))
        for tag, row in blob.items():
            scores[tag] = row

    arms: dict[str, list[str]] = {}
    for tag in tags:
        arm = tag.split("_s")[0]
        arms.setdefault(arm, []).append(tag)

    print("=" * 100)
    print("S1: the four structure quantities, per arm (median over the three seeds)")
    print("=" * 100)
    hdr = (f"{'arm':6s} {'tag':16s} {'tail_max/p99':>13s} {'dry_frac':>10s} "
           f"{'kernel_px':>11s} {'BIAS@2.0':>10s}")
    print(hdr)
    medians: dict[str, dict[str, float]] = {}
    for arm in sorted(arms):
        acc: dict[str, list[float]] = {}
        for tag in sorted(arms[arm]):
            e = tags[tag]
            if "error" in e:
                print(f"{arm:6s} {tag:16s} ERROR: {e['error']}")
                continue
            p = e["pred"]
            sc = scores.get(tag, {})
            print(f"{arm:6s} {tag:16s} {_fmt(p.get('tail_max_over_p99')):>13s} "
                  f"{_fmt(p.get('dry_frac_lt_0p1')):>10s} "
                  f"{_fmt(p.get('kernel_px_gt_p99')):>11s} "
                  f"{_fmt(sc.get('bias_at_2.0')):>10s}")
            for k in ("tail_max_over_p99", "dry_frac_lt_0p1",
                      "kernel_px_gt_p99", "probe_zero_frac"):
                v = p.get(k)
                if isinstance(v, (int, float)):
                    acc.setdefault(k, []).append(float(v))
        if acc:
            medians[arm] = {k: sum(v) / len(v) for k, v in acc.items()}

    print()
    print("=" * 100)
    print("the truth field measured with the same implementation (the reference)")
    print("=" * 100)
    for arm in sorted(arms):
        tag = sorted(arms[arm])[0]
        e = tags[tag]
        if "error" in e or "truth" not in e:
            continue
        t = e["truth"]
        print(f"{arm:6s} truth: tail_max/p99={_fmt(t.get('tail_max_over_p99'))} "
              f"dry_frac={_fmt(t.get('dry_frac_lt_0p1'))} "
              f"kernel_px={_fmt(t.get('kernel_px_gt_p99'))}")
        break

    print()
    print("=" * 100)
    print(f"S1 verdict, {args.control} -> A arms")
    print("=" * 100)
    print(f"label reference: tail_max/p99={LABEL_TAIL_MAX_OVER_P99}, "
          f"dry_frac={LABEL_DRY_FRAC}, kernel_px={LABEL_KERNEL_PX}")
    ctrl = medians.get(args.control, {})
    for arm in sorted(medians):
        if arm == args.control:
            continue
        m = medians[arm]
        tail_up = m["tail_max_over_p99"] > ctrl.get("tail_max_over_p99", 0.0)
        kern_up = m["kernel_px_gt_p99"] > ctrl.get("kernel_px_gt_p99", 0.0)
        # "toward the label" means the gap to 0.912 shrinks.
        gap_a = abs(m["dry_frac_lt_0p1"] - LABEL_DRY_FRAC)
        gap_c = abs(ctrl.get("dry_frac_lt_0p1", LABEL_DRY_FRAC) - LABEL_DRY_FRAC)
        dry_toward = gap_a < gap_c
        print(f"\n{arm}:")
        print(f"  tail_max_over_p99  {ctrl.get('tail_max_over_p99'):.4f} -> "
              f"{m['tail_max_over_p99']:.4f}   {'UP (ok)' if tail_up else 'NOT UP (fail)'}")
        print(f"  kernel_px_gt_p99   {ctrl.get('kernel_px_gt_p99'):.2f} -> "
              f"{m['kernel_px_gt_p99']:.2f}   {'UP (ok)' if kern_up else 'NOT UP (fail)'}")
        print(f"  dry_frac           {ctrl.get('dry_frac_lt_0p1'):.4f} -> "
              f"{m['dry_frac_lt_0p1']:.4f}   "
              f"{'CLOSER to 0.912 (ok)' if dry_toward else 'NOT closer (fail)'} "
              f"(gap {gap_c:.4f} -> {gap_a:.4f})")
        core = tail_up and kern_up and dry_toward
        print(f"  => the three field-shape quantities "
              f"{'point the SAME way (mechanism chain holds)' if core else 'DO NOT all move (fail)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
