#!/usr/bin/env python3
"""Pre-flight check for ``--dump-fields`` files used by E25 / E26 / E28.

The offline probes score the dumped fields, but the FSS anchor, the paired
bootstrap and the per-seed scores all compare against float32 reports and
window tables produced during inference. float16 cannot represent 0.1
exactly, so a truth value of exactly 0.1 mm/6 min becomes 0.0999756 and
flips from "rain" to "no rain" at the 0.1 threshold; continuous forecasts
are likewise rounded across 0.5 / 2.0 / 4.0.

Two checks, both zero-training-compute:

1. observed-rain pixels (hits+misses) of the dumped truth vs the reference
   persistence window table (an independent float32 run);
2. all four contingency counts of dumped pred/truth vs the same run's own
   ``--per-window-out`` table, when that table exists.

Exit code 1 and an ABORT line mean the dumped fields must be regenerated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/wf/radarNWP")
TIERS = ("0.1", "0.5", "2.0", "4.0")
N_CHECK = 32
COUNTS = ("hits", "misses", "false_alarms", "correct_negatives")


def counts_from(pred: np.ndarray, truth: np.ndarray, thr: float) -> dict:
    p, t = pred >= thr, truth >= thr
    return {
        "hits": float(np.count_nonzero(p & t)),
        "misses": float(np.count_nonzero(~p & t)),
        "false_alarms": float(np.count_nonzero(p & ~t)),
        "correct_negatives": float(np.count_nonzero(~p & ~t)),
    }


def main() -> int:
    truth_path = ROOT / "runs/fields/base_concat_s0.truth.npy"
    pred_path = ROOT / "runs/fields/base_concat_s0.pred.npy"
    ref_path = ROOT / "runs/pmm_k1.windows.json"
    own_path = ROOT / "runs/fields/base_concat_s0.windows.json"
    for p in (truth_path, pred_path, ref_path):
        if not p.exists():
            print(f"MISSING {p}")
            return 1

    truth = np.load(truth_path, mmap_mode="r")
    pred = np.load(pred_path, mmap_mode="r")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))["windows"]
    own = (json.loads(own_path.read_text(encoding="utf-8"))["windows"]
           if own_path.exists() else None)
    print(f"truth {truth.shape} {truth.dtype}; pred {pred.shape} {pred.dtype}; "
          f"reference windows {len(ref)}; own window table "
          f"{'present' if own else 'absent'}")

    n = min(N_CHECK, truth.shape[0], len(ref))
    bad_truth = {k: 0 for k in TIERS}
    max_abs = {k: 0.0 for k in TIERS}
    bad_own = 0
    for i in range(n):
        t = np.asarray(truth[i], dtype=np.float32)
        p = np.asarray(pred[i], dtype=np.float32)
        for key in TIERS:
            thr = float(key)
            hm_ref = ref[i][key]["hits"] + ref[i][key]["misses"]
            diff = int(np.count_nonzero(t >= thr)) - hm_ref
            if diff:
                bad_truth[key] += 1
                max_abs[key] = max(max_abs[key], abs(diff))
            if own is not None:
                got = counts_from(p, t, thr)
                if any(got[c] != own[i][key][c] for c in COUNTS):
                    bad_own += 1
        if i == 0:
            for key in TIERS:
                thr = float(key)
                near = t[(t > thr * 0.995) & (t < thr * 1.005)]
                vals = np.unique(near)
                print(f"  tier {key}: unique dumped values within 0.5% of "
                      f"threshold = {vals[:6].tolist()}")

    for key in TIERS:
        print(f"  truth tier {key}: {bad_truth[key]}/{n} windows disagree on "
              f"observed-rain pixels, max |diff| = {max_abs[key]:.0f}")
    if own is not None:
        print(f"  own-table full contingency mismatch: {bad_own}/{n * len(TIERS)} "
              f"window-tier pairs")

    ok = (str(truth.dtype) == "float32" and str(pred.dtype) == "float32"
          and not any(bad_truth.values()) and bad_own == 0)
    print("VERDICT:",
          "THRESHOLD_SAFE" if ok else
          "ABORT: dumped fields break the per-window invariant; "
          "regenerate with float32 --dump-fields before scoring")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
