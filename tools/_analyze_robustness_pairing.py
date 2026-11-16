"""Read the missing-frame pairing: one checkpoint, the interpolation switched.

The split holds the twenty-three cases whose record has a hole in it.
That is a small sample, so the total is not the number to read: the
per-lead differences and their intervals are.

The two runs differ by one thing. The same checkpoint is scored twice,
once with the gap repaired by interpolating across it and once with the
tolerance switched off, so the case is dropped instead. Nothing else
moves: the model, the weights, the split and the verifier are the same.

The judge is the interval, not the sign. A difference whose interval
crosses zero is not a demonstration that the repair helps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 10000
RNG_SEED = 20261103
TIERS = ("0.1", "0.5", "2.0", "4.0")


def csi_of(h: float, m: float, f: float) -> float:
    d = h + m + f
    return h / d if d > 0 else float("nan")


def load(path: Path):
    data = json.loads(path.read_text())
    rows = data.get("windows") or []
    if not rows:
        return None, None
    out = np.zeros((len(rows), 3))
    for i, w in enumerate(rows):
        c = w
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out, data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--with", dest="with_", default="robustness_with_interp.eval.json")
    ap.add_argument("--without", dest="without",
                    default="robustness_no_interp.eval.json")
    args = ap.parse_args()

    d = Path(args.directory)
    pa, pb = d / args.with_, d / args.without
    for p in (pa, pb):
        if not p.is_file():
            print(f"  {p.name} is missing")
            return 1

    print("=" * 92)
    print("the missing-frame pairing: the same checkpoint, the repair switched")
    print("=" * 92)
    print()

    for tier in TIERS:
        ca, da = load(pa)
        cb, db = load(pb)
        if ca is None or cb is None:
            print(f"  CSI@{tier}: no window tables")
            continue
        n = min(len(ca), len(cb))
        ca, cb = ca[:n], cb[:n]
        point_a = csi_of(*ca.sum(axis=0))
        point_b = csi_of(*cb.sum(axis=0))
        rng = np.random.default_rng(RNG_SEED)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            diffs[i] = csi_of(*ca[idx].sum(axis=0)) - csi_of(*cb[idx].sum(axis=0))
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        crosses = lo <= 0.0 <= hi
        print(f"  CSI@{tier}")
        print(f"    with the repair    : {point_a:.4f}")
        print(f"    without the repair : {point_b:.4f}")
        print(f"    difference         : {point_a - point_b:+.4f}")
        print(f"    95 % interval      : [{lo:+.4f}, {hi:+.4f}]")
        print("    verdict            : " + (
            "includes zero -- the repair is not shown to help"
            if crosses else
            "excludes zero -- the repair changes the score beyond the "
            "sampling uncertainty"
        ))
        print()

    # The lead-time breakdown is what a paper's table needs: the claim is
    # usually that the repair helps more the further out the forecast is.
    la = json.loads(pa.read_text()).get("lead_csi", {})
    lb = json.loads(pb.read_text()).get("lead_csi", {})
    if la and lb:
        print("  CSI@2.0 per lead frame")
        print(f"    {'lead':<6} {'with':>10} {'without':>10} {'difference':>12}")
        for lead in sorted(la, key=int):
            if lead not in lb:
                continue
            a = la[lead].get("2.0")
            b = lb[lead].get("2.0")
            if a is None or b is None:
                continue
            print(f"    {lead:<6} {a:>10.4f} {b:>10.4f} {a - b:>+12.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
