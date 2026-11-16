#!/usr/bin/env python
"""Read-only partial reader for the E45 epoch axis.

Runs ON THE REMOTE, reads only already-written *.eval.json files under
runs/e45_epochaxis/evals/, and prints a compact JSON blob.

Rules honoured here (from HANDOVER_E45.md):
  * post-hoc diagnostic only -- never a new criterion, never a verdict change;
  * report median + pass rate, never a single-point score as a conclusion;
  * every "epoch makes it better/worse" statement must carry the direction
    and the size, not just a sign;
  * the ep00 column is a SAME-PROTOCOL BASELINE (byte-identical to the
    published E45 files for B0_s0) -- differences across epoch therefore
    belong to the epoch, not to the re-run pipeline.

It NEVER writes into runs/e45/ (the published tree) and never mutates
anything under runs/e45_epochaxis/.
"""

import json
import os
import re
import statistics
import sys

ROOT = "/home/wf/radarNWP/runs/e45_epochaxis/evals"
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
EPOCHS = ["ep00", "ep01", "ep02"]
FLOOR = 0.0924  # persistence floor, seed-0 convention (HANDOVER sec.3)

FNAME = re.compile(r"^(?P<arm>[A-Za-z0-9]+)_s(?P<seed>\d+)_(?P<epoch>ep\d\d)\.eval\.json$")


def read_cell(path):
    """Return (csi_at_2.0, bias_at_2.0) from one *.eval.json.

    Layout confirmed on the remote: {"n_samples": 538,
    "thresholds": {"0.1": {...}, "0.5": {...}, "2.0": {"csi":..., "bias":...},
    "4.0": {...}}, ...}.  The 2.0 mm block is the pre-registered S3 guard
    threshold, so that is the one read here.
    """
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        return None, None
    thr = obj.get("thresholds")
    if isinstance(thr, dict):
        for key in ("2.0", "2", 2.0):
            blk = thr.get(key)
            if isinstance(blk, dict) and "csi" in blk:
                bias = blk.get("bias")
                return float(blk["csi"]), (None if bias is None else float(bias))
    # defensive fallback for flat layouts
    for key in ("csi_at_2.0", "CSI@2.0"):
        if key in obj:
            b = obj.get("bias_at_2.0")
            return float(obj[key]), (None if b is None else float(b))
    return None, None


def main():
    if not os.path.isdir(ROOT):
        print(json.dumps({"error": "missing root", "root": ROOT}))
        return 0

    cells = {}
    biases = {}
    for fn in sorted(os.listdir(ROOT)):
        m = FNAME.match(fn)
        if not m:
            continue
        arm, seed, epoch = m.group("arm"), int(m.group("seed")), m.group("epoch")
        csi, bias = read_cell(os.path.join(ROOT, fn))
        if csi is None:
            continue
        cells.setdefault(epoch, {})[f"{arm}_s{seed}"] = csi
        biases.setdefault(epoch, {})[f"{arm}_s{seed}"] = bias

    out = {"root": ROOT, "epochs": {}, "floor": FLOOR}

    # --- per-epoch per-arm summaries (only arms/seeds actually present) ---
    for ep in EPOCHS:
        got = cells.get(ep, {})
        if not got:
            continue
        per_arm = {}
        for arm in ARMS:
            vals = [v for k, v in got.items() if k.startswith(arm + "_s")]
            if not vals:
                continue
            per_arm[arm] = {
                "n": len(vals),
                "median_csi_at_2.0": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "pass_floor": sum(1 for v in vals if v >= FLOOR),
            }
        out["epochs"][ep] = {"n_cells": len(got), "per_arm": per_arm}

    # --- column medians (arm-level, median over arms) per epoch ---
    col = {}
    for ep in EPOCHS:
        block = out["epochs"].get(ep)
        if not block:
            continue
        meds = [a["median_csi_at_2.0"] for a in block["per_arm"].values()]
        col[ep] = {
            "arm_median_median": statistics.median(meds),
            "arm_median_min": min(meds),
            "arm_median_max": max(meds),
            "arms_median_over_floor": sum(1 for m in meds if m >= FLOOR),
            "n_arms": len(meds),
        }
    out["column_medians"] = col

    # --- same-seed per-arm deltas vs the ep00 baseline column ---
    deltas = {}
    for ep in ("ep01", "ep02"):
        base, cur = cells.get("ep00", {}), cells.get(ep, {})
        d = {}
        for k in sorted(set(base) & set(cur)):
            d[k] = cur[k] - base[k]
        if d:
            deltas[ep] = {
                "n_paired": len(d),
                "median_delta": statistics.median(d.values()),
                "min_delta": min(d.values()),
                "max_delta": max(d.values()),
                "n_negative": sum(1 for x in d.values() if x < 0),
                "n_positive": sum(1 for x in d.values() if x > 0),
            }
    out["paired_delta_vs_ep00"] = deltas

    # --- the key question: is ep00 really the peak? per-arm sign pattern ---
    pattern = {}
    for arm in ARMS:
        rows = {}
        for ep in EPOCHS:
            vals = [v for k, v in cells.get(ep, {}).items() if k.startswith(arm + "_s")]
            if vals:
                rows[ep] = statistics.median(vals)
        if len(rows) >= 2 and "ep00" in rows:
            best = max(rows, key=lambda e: rows[e])
            pattern[arm] = {
                "arm_medians": rows,
                "best_epoch": best,
                "ep00_is_best": best == "ep00",
                "ep01_minus_ep00": rows.get("ep01", float("nan")) - rows["ep00"],
                "ep02_minus_ep00": rows.get("ep02", float("nan")) - rows["ep00"],
            }
    out["per_arm_epoch_pattern"] = pattern
    out["ep00_is_best_count"] = sum(1 for v in pattern.values() if v["ep00_is_best"])
    out["n_arms_with_full_epoch_axis"] = len(pattern)

    # --- BIAS along the epoch axis (the over-forecast fingerprint) ---
    bias_cols = {}
    for ep in EPOCHS:
        got = biases.get(ep, {})
        if not got:
            continue
        per_arm = {}
        for arm in ARMS:
            vals = [v for k, v in got.items() if k.startswith(arm + "_s") and v is not None]
            if vals:
                per_arm[arm] = {"n": len(vals), "median_bias": statistics.median(vals)}
        bias_cols[ep] = per_arm
    out["bias_per_arm"] = bias_cols
    if "ep00" in bias_cols:
        bias_delta = {}
        for ep in ("ep01", "ep02"):
            base = {k: v for k, v in biases.get("ep00", {}).items() if v is not None}
            cur = {k: v for k, v in biases.get(ep, {}).items() if v is not None}
            common = sorted(set(base) & set(cur))
            if common:
                d = [cur[k] - base[k] for k in common]
                bias_delta[ep] = {
                    "n_paired": len(d),
                    "median_delta_bias": statistics.median(d),
                    "n_increased": sum(1 for x in d if x > 0),
                    "n_decreased": sum(1 for x in d if x < 0),
                }
        out["paired_bias_delta_vs_ep00"] = bias_delta

    json.dump(out, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
