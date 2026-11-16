#!/usr/bin/env python3
"""Compare the score-aware matrix with the standard-loss matrix.

The standard-loss numbers are the ones the paper already reports, in
``runs/tgrs``.  They are the comparable set: same split, same sample
count, same normalization stamp.  ``runs/hqma`` holds a different
experiment and only covers four models, so it is not the baseline here.

    python scripts/compare_score_matrix.py [--out-dir runs/score_matrix]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TIERS = ("0.1", "0.5", "2.0", "4.0")

# The eighteen models of the matrix, in the order the driver runs them.
MODELS = [
    "simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_early_concat", "simvp_radar_only", "convlstm",
    "predrnn", "earthformer", "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
]

# The analytical baseline is not trained, so it has no score-aware twin.
# It is carried along as the bar the models have to clear.
REFERENCE = {"pysteps_adv": "runs/tgrs/pysteps_adv.eval.json"}


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def csi(report, tier):
    if report is None:
        return float("nan")
    return report["thresholds"].get(tier, {}).get("csi", float("nan"))


def far(report, tier):
    if report is None:
        return float("nan")
    return report["thresholds"].get(tier, {}).get("far", float("nan"))


def hits(report, tier):
    if report is None:
        return float("nan")
    return report["thresholds"].get(tier, {}).get("hits", float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/score_matrix")
    parser.add_argument("--old-dir", default="runs/tgrs")
    parser.add_argument("--root", default="/home/wf/radarNWP")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    old_dir = root / args.old_dir

    rows = []
    for model in MODELS:
        new = read(out_dir / f"{model}.eval.json")
        old = read(old_dir / f"{model}.eval.json")
        rows.append((model, old, new))

    have_new = sum(1 for _, _, new in rows if new is not None)
    have_old = sum(1 for _, old, _ in rows if old is not None)
    print(f"score-aware reports: {have_new}/{len(MODELS)}")
    print(f"standard-loss reports: {have_old}/{len(MODELS)}")
    print()

    header = (
        f"{'model':22s}{'old@2.0':>9s}{'new@2.0':>9s}{'ratio':>8s}"
        f"{'old@4.0':>9s}{'new@4.0':>9s}{'old@0.1':>9s}{'new@0.1':>9s}"
        f"{'newFAR@0.5':>11s}"
    )
    print("=== heavy rain: does the objective lift every model? ===")
    print(header)
    print("-" * len(header))
    for model, old, new in rows:
        o2, n2 = csi(old, "2.0"), csi(new, "2.0")
        o4, n4 = csi(old, "4.0"), csi(new, "4.0")
        o01, n01 = csi(old, "0.1"), csi(new, "0.1")
        ratio = (n2 / o2) if (o2 and o2 > 0) else float("nan")
        print(
            f"{model:22s}{o2:9.4f}{n2:9.4f}{ratio:8.1f}"
            f"{o4:9.4f}{n4:9.4f}{o01:9.4f}{n01:9.4f}{far(new, '0.5'):11.4f}"
        )

    print()
    print("=== the analytical baseline, and where it now sits ===")
    for name, rel in REFERENCE.items():
        ref = read(root / rel)
        print(
            f"{name:22s} CSI@2.0={csi(ref, '2.0'):.4f} "
            f"CSI@4.0={csi(ref, '4.0'):.4f} CSI@0.1={csi(ref, '0.1'):.4f} "
            f"FAR@0.5={far(ref, '0.5'):.4f}"
        )
    above = [
        model for model, _, new in rows
        if new is not None and csi(new, "2.0") > csi(read(root / REFERENCE["pysteps_adv"]), "2.0")
    ]
    print(f"  models above pysteps_adv at CSI@2.0: {len(above)}/{len(MODELS)}")
    if above:
        print(f"    {', '.join(above)}")

    print()
    print("=== light rain: what the objective cost ===")
    drops = []
    for model, old, new in rows:
        if old is None or new is None:
            continue
        o01, n01 = csi(old, "0.1"), csi(new, "0.1")
        drops.append((n01 - o01, model, o01, n01))
    drops.sort()
    for delta, model, o01, n01 in drops:
        print(f"{model:22s} CSI@0.1 {o01:.4f} -> {n01:.4f}  ({delta:+.4f})")
    if drops:
        below = [m for d, m, _, n01 in drops if n01 < 0.1]
        print(f"  models below CSI@0.1 = 0.1: {len(below)}/{len(drops)}")

    print()
    print("=== hits at 2 mm, the raw count behind the ratio ===")
    for model, old, new in rows:
        print(f"{model:22s} old={hits(old, '2.0'):12.0f} new={hits(new, '2.0'):12.0f}")

    print()
    print("=== ranking by CSI@2.0, standard loss then score-aware ===")
    old_rank = sorted(
        [(m, csi(o, "2.0")) for m, o, _ in rows if o is not None],
        key=lambda t: -t[1],
    )
    new_rank = sorted(
        [(m, csi(n, "2.0")) for m, _, n in rows if n is not None],
        key=lambda t: -t[1],
    )
    print(f"{'rank':>5s}{'standard loss':>34s}{'score-aware':>34s}")
    for i in range(max(len(old_rank), len(new_rank))):
        left = f"{old_rank[i][0]} {old_rank[i][1]:.4f}" if i < len(old_rank) else ""
        right = f"{new_rank[i][0]} {new_rank[i][1]:.4f}" if i < len(new_rank) else ""
        print(f"{i + 1:5d}{left:>34s}{right:>34s}")

    old_order = [m for m, _ in old_rank]
    new_order = [m for m, _ in new_rank]
    common = [m for m in new_order if m in old_order]
    old_common = [m for m in old_order if m in common]
    if len(common) > 1:
        rho = _spearman(old_common, common)
        print(f"\nSpearman rho between the two orderings (common models): {rho:.3f}")
    return 0


def _spearman(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ra = {name: i + 1 for i, name in enumerate(a)}
    rb = {name: i + 1 for i, name in enumerate(b)}
    d2 = sum((ra[name] - rb[name]) ** 2 for name in a)
    return 1 - (6 * d2) / (n * (n * n - 1))


if __name__ == "__main__":
    raise SystemExit(main())
