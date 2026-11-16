"""Two-level bootstrap over the multi-seed evaluations, run where the data is.

The per-window tables are about 317 kB each and the SSH channel truncates at
about 256 kB, so the bootstrap has to run on the remote.

Two levels are resampled: the seed, because the spread between runs is the
question, and the evaluation window, because the score is a ratio of
accumulated counts over the windows and the windows are the unit of
independent sampling. The windows are paired between the two arms -- the
same window index in both -- so the difference is computed on the same
weather.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EVALS = Path("/home/wf/radarNWP/runs/route_seeds/evals")
FLOOR = 0.0924
N_BOOT = 10000
RNG_SEED = 20261103
TIERS = ("0.1", "0.5", "2.0", "4.0")


def csi(h: float, m: float, f: float) -> float:
    d = h + m + f
    return h / d if d > 0 else float("nan")


def window_counts(path: Path, tier: str) -> np.ndarray:
    data = json.loads(path.read_text())
    rows = data["windows"]
    out = np.zeros((len(rows), 3))
    for i, w in enumerate(rows):
        c = w[tier]
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out


def seeds_for(arm: str) -> list[int]:
    found = set()
    for p in EVALS.glob(f"{arm}_s*_ep*.eval.json"):
        stem = p.name[: -len(".eval.json")]
        seed = stem.rsplit("_", 2)[1]
        found.add(int(seed[1:]))
    return sorted(found)


def main() -> int:
    print("=" * 92)
    print("the two-level bootstrap over seeds and windows (10000 resamples)")
    print("=" * 92)

    for epoch in (0, 1, 2):
        tag = f"ep{epoch:02d}"
        arm_a, arm_b = "qpe_0", "base_concat"
        seeds = sorted(set(seeds_for(arm_a)) & set(seeds_for(arm_b)))
        if not seeds:
            continue
        for tier in TIERS:
            per_seed = []
            for s in seeds:
                pa = EVALS / f"{arm_a}_s{s}_{tag}.windows.json"
                pb = EVALS / f"{arm_b}_s{s}_{tag}.windows.json"
                if not (pa.exists() and pb.exists()):
                    per_seed = []
                    break
                ca, cb = window_counts(pa, tier), window_counts(pb, tier)
                n = min(len(ca), len(cb))
                per_seed.append((ca[:n], cb[:n]))
            if not per_seed:
                print(f"\n  {tag} CSI@{tier}: window tables missing, skipped")
                continue

            n_win = min(len(a) for a, _ in per_seed)
            rng = np.random.default_rng(RNG_SEED)
            diffs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                # resample the seed
                si = rng.integers(0, len(per_seed), len(per_seed))
                # resample the windows, the same indices for both arms
                wi = rng.integers(0, n_win, n_win)
                da = db = 0.0
                for k in si:
                    ca, cb = per_seed[k]
                    da += csi(*ca[wi].sum(axis=0))
                    db += csi(*cb[wi].sum(axis=0))
                diffs[i] = da - db

            point_a = np.mean([csi(*a.sum(axis=0)) for a, _ in per_seed])
            point_b = np.mean([csi(*b.sum(axis=0)) for _, b in per_seed])
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            crosses = lo <= 0.0 <= hi
            print(f"\n  {tag}  CSI@{tier}   (qpe_0 minus base_concat, {len(seeds)} seeds)")
            print(f"    mean over seeds : {point_a:.4f} - {point_b:.4f} = {point_a - point_b:+.4f}")
            print(f"    95 % interval    : [{lo:+.4f}, {hi:+.4f}]")
            print(
                "    verdict         : "
                + (
                    "includes zero -- the arms cannot be separated"
                    if crosses
                    else "excludes zero -- the difference exceeds the sampling uncertainty"
                )
            )

    print()
    print("=" * 92)
    print("the spread between seeds, per arm and epoch (the ratio of the best to the worst)")
    print("=" * 92)
    for arm in ("qpe_0", "base_concat"):
        for epoch in (0, 1, 2):
            tag = f"ep{epoch:02d}"
            vals = []
            for s in seeds_for(arm):
                p = EVALS / f"{arm}_s{s}_{tag}.eval.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                vals.append(float(d["thresholds"]["2.0"]["csi"]))
            if len(vals) < 2:
                continue
            lo, hi = min(vals), max(vals)
            print(
                f"  {arm:<14} {tag}  n={len(vals)}  spread {hi / lo:6.2f}x"
                f"   range [{lo:.4f}, {hi:.4f}]"
                f"   over the floor {sum(1 for v in vals if v > FLOOR)}/{len(vals)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
