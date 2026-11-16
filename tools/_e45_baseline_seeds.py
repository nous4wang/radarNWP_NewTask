#!/usr/bin/env python3
"""E45-prep -- the baseline picture, seed by seed.

The survey found that the highest CSI@2.0 ever measured on this dataset
is 0.186511, on the arm called ``base_concat``.  That number is above
the Oracle bound of 0.1653, which cannot be right as a ceiling, and
the master document already explains why: it is one lucky seed out of
ten, and the same arm's ten-seed spread at the peak epoch is 9.59x,
from 0.0194 to 0.1865.

So the question this script answers is not "what is the best number" but
"what does the baseline actually look like when the seeds are lined up",
because a candidate method has to beat THAT, not beat a single lucky
draw.

For every arm family in the ten-seed batches the script reports, per
epoch, the per-seed CSI@2.0, the median, the pass rate against the
persistence floor of 0.0924, and the spread.  The seed-level pairing
matters: the master document's pass rates (qpe_0 5/10 against
base_concat 4/10, Fisher one-sided p = 0.5) are the honest comparison,
and any new arm has to be judged against the same kind of pairing.

Read-only: it reads the evaluation JSON files that already exist.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

FLOOR = 0.0924
ORACLE = 0.1653

# arm name -> the seed tag it produces
FAMILIES = {
    "qpe_0": "qpe_0",
    "base_concat": "base_concat",
}


def _parse(tag: str):
    """``<arm>_s<seed>_<rest>`` -> ``(arm, seed, rest)`` when it fits."""
    m = re.match(r"^(?P<arm>.+?)_s(?P<seed>\d+)(?P<rest>.*)$", tag)
    if not m:
        return None
    return m.group("arm"), int(m.group("seed")), m.group("rest")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    rows = d["rows"]

    # --- the ten-seed families ------------------------------------------
    families: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        p = _parse(r["tag"])
        if not p:
            continue
        arm, seed, rest = p
        if arm not in FAMILIES:
            continue
        ep = None
        m = re.search(r"ep(\d+)", rest)
        if m:
            ep = int(m.group(1))
        else:
            ep = -1                       # a bare tag: the final checkpoint
        families[arm].setdefault(ep, {})[seed] = r

    out: Dict[str, Any] = {
        "floor": FLOOR,
        "oracle": ORACLE,
        "families": {},
    }

    for arm, eps in sorted(families.items()):
        out["families"][arm] = {}
        for ep, seeds in sorted(eps.items()):
            csis = np.array([seeds[s]["csi"] for s in sorted(seeds)])
            biases = np.array([seeds[s]["bias"] for s in sorted(seeds)])
            n_pass = int((csis > FLOOR).sum())
            entry = {
                "epoch": ep,
                "n_seeds": len(csis),
                "seeds": sorted(seeds),
                "csi_per_seed": {str(s): seeds[s]["csi"] for s in sorted(seeds)},
                "bias_per_seed": {str(s): seeds[s]["bias"] for s in sorted(seeds)},
                "csi_median": float(np.median(csis)),
                "csi_min": float(csis.min()),
                "csi_max": float(csis.max()),
                "csi_spread_ratio": (float(csis.max() / csis.min())
                                     if csis.min() > 0 else float("inf")),
                "pass_rate_over_floor": f"{n_pass}/{len(csis)}",
                "n_pass": n_pass,
                "bias_median": float(np.median(biases)),
            }
            out["families"][arm][str(ep)] = entry

    print(f"floor {FLOOR}   oracle {ORACLE}\n")
    for arm, eps in sorted(out["families"].items()):
        print(f"===== {arm} =====")
        for ep, e in sorted(eps.items(), key=lambda kv: int(kv[0])):
            if e["n_seeds"] < 3:
                continue
            print(f"  ep{int(ep):02d}  n={e['n_seeds']}  "
                  f"median {e['csi_median']:.6f}  "
                  f"min {e['csi_min']:.6f}  max {e['csi_max']:.6f}  "
                  f"spread {e['csi_spread_ratio']:.2f}x  "
                  f"pass {e['pass_rate_over_floor']}  "
                  f"BIASmed {e['bias_median']:.4f}")
            cs = " ".join(f"{v:.4f}" for v in
                          sorted(e["csi_per_seed"].values(), reverse=True))
            print(f"        seeds: {cs}")
        print()

    # --- the paired comparison at the peak epoch ------------------------
    print("=== the paired comparison, same seeds, peak epoch ===")
    peaks = {}
    for arm, eps in out["families"].items():
        cand = [(int(ep), e) for ep, e in eps.items() if e["n_seeds"] >= 5]
        if cand:
            best = max(cand, key=lambda kv: kv[1]["csi_median"])
            peaks[arm] = (arm, best[0], best[1])
    arms = sorted(peaks)
    if len(arms) == 2:
        a0, ep0, e0 = peaks[arms[0]]
        a1, ep1, e1 = peaks[arms[1]]
        common = sorted(set(e0["csi_per_seed"]) & set(e1["csi_per_seed"]))
        if common:
            diffs = [e1["csi_per_seed"][s] - e0["csi_per_seed"][s] for s in common]
            wins = sum(1 for x in diffs if x > 0)
            print(f"  {a0} (ep{ep0}) vs {a1} (ep{ep1}) over {len(common)} shared seeds")
            for s in common:
                print(f"    seed {s}: {a0} {e0['csi_per_seed'][s]:.6f}   "
                      f"{a1} {e1['csi_per_seed'][s]:.6f}   "
                      f"diff {e1['csi_per_seed'][s] - e0['csi_per_seed'][s]:+.6f}")
            print(f"  mean diff {np.mean(diffs):+.6f}   "
                  f"{wins} positive / {len(diffs) - wins} negative")
            out["paired"] = {
                "arm_a": a0, "epoch_a": ep0, "arm_b": a1, "epoch_b": ep1,
                "seeds": common, "diffs": diffs,
                "n_positive": wins, "n_negative": len(diffs) - wins,
                "mean_diff": float(np.mean(diffs)),
            }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
