"""Run the bootstrap over the evaluation windows, on the remote.

The window tables are about 320 kB each, and the SSH channel truncates
around 256 kB, so they cannot be brought down to be analysed locally: a
fetched file arrives with its tail missing and will not parse. The
bootstrap therefore runs where the data is.

Two things are reported for each comparison: the difference between the
two configurations at the observed score, and the interval that comes from
resampling the evaluation windows. The resampling unit is the window,
because that is the independent unit of the evaluation -- the pixels
inside one window are not independent of each other.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED_DIR = Path("/home/wf/radarNWP/runs/route_p2/seeds")
N_BOOT = 10000
RNG_SEED = 12345
TIERS = ["0.1", "0.5", "2.0", "4.0"]
FLOOR = 0.0924


def csi_of(h: float, m: float, f: float) -> float:
    d = h + m + f
    return h / d if d > 0 else float("nan")


def counts(path: Path, thr: str) -> np.ndarray:
    rows = json.loads(path.read_text())["windows"]
    out = np.zeros((len(rows), 3))
    for i, w in enumerate(rows):
        c = w[thr]
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out


def main() -> None:
    print("=" * 92)
    print("the bootstrap over the evaluation windows (10000 resamples)")
    print("=" * 92)
    for ep in ("00", "19"):
        for thr in TIERS:
            a_files = sorted(SEED_DIR.glob(f"qpe_0_s*_ep{ep}.windows.json"))
            b_files = sorted(SEED_DIR.glob(f"base_concat_s*_ep{ep}.windows.json"))
            if not a_files or not b_files:
                continue
            ca = counts(a_files[0], thr)
            cb = counts(b_files[0], thr)
            n = min(len(ca), len(cb))
            ca, cb = ca[:n], cb[:n]
            point = (
                csi_of(ca[:, 0].sum(), ca[:, 1].sum(), ca[:, 2].sum())
                - csi_of(cb[:, 0].sum(), cb[:, 1].sum(), cb[:, 2].sum())
            )
            rng = np.random.default_rng(RNG_SEED)
            diffs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                idx = rng.integers(0, n, n)
                sa = ca[idx].sum(axis=0)
                sb = cb[idx].sum(axis=0)
                diffs[i] = csi_of(sa[0], sa[1], sa[2]) - csi_of(sb[0], sb[1], sb[2])
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            crosses = lo <= 0.0 <= hi
            print(f"\n  ep{ep}  CSI@{thr}   (qpe_0 minus base_concat, seed 0)")
            print(f"    windows      : {n}")
            print(f"    difference   : {point:+.4f}")
            print(f"    95 % interval: [{lo:+.4f}, {hi:+.4f}]")
            print(
                "    verdict      : "
                + (
                    "includes zero -- the sampling uncertainty alone does "
                    "not separate them"
                    if crosses
                    else "excludes zero -- the difference exceeds the "
                    "sampling uncertainty"
                )
            )


if __name__ == "__main__":
    main()
