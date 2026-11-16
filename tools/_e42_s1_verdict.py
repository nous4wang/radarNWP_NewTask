"""E42 -- merge the E41 control (B0) and the E42 arm (A2') and read S1.

READ-ONLY.  The control's structure metrics were measured in E41's batch
and the arm's in E42's; both used the same implementation (the section 2
probe's own ``metrics``, imported by tools/_e41_struct_metrics.py) and the
same 538 test windows, so the two are comparable.

It prints the four S1 quantities per arm and applies the criterion as it
was frozen: BIAS@2.0 must move up by at least +0.10, tail_max_over_p99
must rise, dry_frac must move toward the label's 0.912, and
kernel_px_gt_p99 must rise -- all four the same way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The section 2 label-side medians (E41 section 2.1).
LABEL_TAIL_MAX_OVER_P99 = 4.0
LABEL_DRY_FRAC = 0.912
LABEL_KERNEL_PX = 108.8
BIAS_DELTA_MIN = 0.10


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["tags"]


def _median(rows: list[dict], key: str):
    vals = [r[key] for r in rows
            if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control-struct", required=True)
    ap.add_argument("--arm-struct", required=True)
    ap.add_argument("--control-scores", required=True)
    ap.add_argument("--arm-scores", required=True)
    ap.add_argument("--control-prefix", default="B0")
    ap.add_argument("--arm-prefix", default="A2p")
    args = ap.parse_args()

    ctags = _load(args.control_struct)
    atags = _load(args.arm_struct)
    cscores = json.loads(Path(args.control_scores).read_text(encoding="utf-8"))
    ascores = json.loads(Path(args.arm_scores).read_text(encoding="utf-8"))

    def rows_of(tags: dict, prefix: str) -> list[dict]:
        out = []
        for tag in sorted(tags):
            if not tag.startswith(prefix + "_s"):
                continue
            e = tags[tag]
            if "error" in e:
                continue
            r = dict(e["pred"])
            r["tag"] = tag
            out.append(r)
        return out

    crows = rows_of(ctags, args.control_prefix)
    arows = rows_of(atags, args.arm_prefix)
    if not crows or not arows:
        print("missing rows")
        return 1

    def scores_of(d: dict, prefix: str) -> list[float]:
        return [v["bias_at_2.0"] for k, v in sorted(d.items())
                if k.startswith(prefix + "_s")]

    print("=" * 96)
    print("S1: the four structure quantities (median over the three seeds)")
    print("=" * 96)
    hdr = (f"{'arm':6s} {'tag':16s} {'tail_max/p99':>13s} {'dry_frac':>10s} "
           f"{'kernel_px':>11s} {'BIAS@2.0':>10s}")
    print(hdr)
    for label, rows, sd in (("B0", crows, cscores), ("A2'", arows, ascores)):
        for r in rows:
            b = sd.get(r["tag"], {}).get("bias_at_2.0")
            bs = f"{b:10.4f}" if isinstance(b, (int, float)) else f"{'n/a':>10s}"
            print(f"{label:6s} {r['tag']:16s} {r['tail_max_over_p99']:13.4f} "
                  f"{r['dry_frac_lt_0p1']:10.4f} {r['kernel_px_gt_p99']:11.2f} {bs}")

    cm = {k: _median(crows, k) for k in
          ("tail_max_over_p99", "dry_frac_lt_0p1", "kernel_px_gt_p99")}
    am = {k: _median(arows, k) for k in
          ("tail_max_over_p99", "dry_frac_lt_0p1", "kernel_px_gt_p99")}
    cb = sum(scores_of(cscores, args.control_prefix)) / 3.0
    ab = sum(scores_of(ascores, args.arm_prefix)) / 3.0

    truth = None
    for tag in sorted(atags):
        if "truth" in atags[tag]:
            truth = atags[tag]["truth"]
            break
    if truth:
        print()
        print(f"truth field (same implementation, same windows): "
              f"tail_max/p99={truth['tail_max_over_p99']:.4f} "
              f"dry_frac={truth['dry_frac_lt_0p1']:.4f} "
              f"kernel_px={truth['kernel_px_gt_p99']:.2f}")

    print()
    print("=" * 96)
    print(f"S1 verdict: {args.control_prefix} -> {args.arm_prefix}")
    print("=" * 96)
    print(f"label reference: tail_max/p99={LABEL_TAIL_MAX_OVER_P99}, "
          f"dry_frac={LABEL_DRY_FRAC}, kernel_px={LABEL_KERNEL_PX}")

    tail_up = am["tail_max_over_p99"] > cm["tail_max_over_p99"]
    kern_up = am["kernel_px_gt_p99"] > cm["kernel_px_gt_p99"]
    gap_a = abs(am["dry_frac_lt_0p1"] - LABEL_DRY_FRAC)
    gap_c = abs(cm["dry_frac_lt_0p1"] - LABEL_DRY_FRAC)
    dry_toward = gap_a < gap_c
    bias_ok = (ab - cb) >= BIAS_DELTA_MIN

    print(f"\n{args.arm_prefix}:")
    print(f"  BIAS@2.0             {cb:.4f} -> {ab:.4f}   "
          f"delta={ab - cb:+.4f}  "
          f"{'OK (>= +0.10)' if bias_ok else 'FAIL (< +0.10)'}")
    print(f"  tail_max_over_p99    {cm['tail_max_over_p99']:.4f} -> "
          f"{am['tail_max_over_p99']:.4f}   "
          f"{'UP (ok)' if tail_up else 'NOT UP (fail)'}")
    print(f"  dry_frac             {cm['dry_frac_lt_0p1']:.4f} -> "
          f"{am['dry_frac_lt_0p1']:.4f}   "
          f"{'CLOSER to 0.912 (ok)' if dry_toward else 'NOT closer (fail)'} "
          f"(gap {gap_c:.4f} -> {gap_a:.4f})")
    print(f"  kernel_px_gt_p99     {cm['kernel_px_gt_p99']:.2f} -> "
          f"{am['kernel_px_gt_p99']:.2f}   "
          f"{'UP (ok)' if kern_up else 'NOT UP (fail)'}")
    core = tail_up and kern_up and dry_toward and bias_ok
    print(f"  => the four quantities "
          f"{'point the SAME way (mechanism chain holds)' if core else 'DO NOT all move (S1 does not hold)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
