#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 round 65 -- the EPOCH AXIS (first judgement).

Why this file exists
--------------------
E45 evaluated every cell at epoch 00 ONLY.  The rule "epoch 00 is the
peak, last.pt is the collapse end" was never tested inside E45: round 58
showed the training loss falls monotonically and its argmin sits at
epoch 02, the opposite end of the skill curve.  So the epoch choice is
an ASSUMPTION the whole E45 table rests on.

Round 58 registered the test as needing a separate Slurm job; the Slurm
controller was down.  On 2026-11-14 18:59 the controller came back, so
the job was submitted (315/316/317 = epoch 00 / 01 / 02) and this script
judges its output.

What it does
------------
1.  GUARD: re-verifies E45's own published numbers are untouched
    (S3's 12 intervals still cross zero; the headline table still
    recomputes; the epoch-00 re-run reproduces E45's published
    per-cell CSI bit-for-bit -- this is the in-protocol control that
    separates "the epoch changed" from "the pipeline drifted").
2.  The epoch axis: for every arm, the median / pass rate of
    CSI@2.0 at epoch 00 / 01 / 02, and the paired per-seed difference
    against the control at the SAME epoch (so the seed is a common
    block and the epoch comparison is within-seed).
3.  The mechanism read-outs across epochs (the gate / topo / amount
    families), reported per branch.

Discipline carried over from rounds 40-64
------------------------------------------
*  Median + pass rate only.  A single point score never appears alone.
*  The paired median and the difference-of-medians are reported side
   by side (round 44's rule: reporting only the paired median turns a
   seed-alignment effect into a mechanism gain).
*  Post-hoc diagnostic: no new criterion, no change to E45's verdict.
*  The S3 guard is re-checked first; if it moved, the round stops.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E45 = ROOT / "tools" / "_evals" / "e45"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
MECH_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ARM_NAME = {
    "B0": "B0 对照", "A1": "A1 幅度单调门", "A2": "A2 硬支撑门",
    "A3a": "A3a 拓扑截断", "A3b": "A3b 拓扑平滑+面积归一",
    "A4": "A4 量通道新头", "A4b": "A4b 量通道自带头",
}
EPOCHS = ["00", "01", "02"]
FLOOR = 0.0924
ORACLE = 0.1653


def log(msg):
    print(msg, flush=True)


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else float("nan")


# --------------------------------------------------------------------------
# 1. the guard: E45's published numbers must be untouched
# --------------------------------------------------------------------------
def guard_published():
    out = {}
    s3 = json.loads((E45 / "s3_equalbias.json").read_text(encoding="utf-8"))
    arms = s3["arms"]
    crosses = 0
    total = 0
    for arm, rec in arms.items():
        for axis in ("raw", "equal_bias"):
            blk = rec[axis]
            total += 1
            if blk.get("crosses_zero"):
                crosses += 1
    out["s3_intervals_total"] = total
    out["s3_intervals_crossing_zero"] = crosses
    out["s3_all_cross_zero"] = (crosses == total)
    verdicts = {a: r["equal_bias"].get("verdict") for a, r in arms.items()}
    out["s3_verdicts"] = verdicts

    # the headline table recomputed from the per-seed products
    summary = json.loads((E45 / "scores_summary.json").read_text(encoding="utf-8"))
    worst = 0.0
    for arm in ARMS:
        pub = summary[arm]
        for seed in (0, 1, 2):
            key = f"{arm}_s{seed}_ep00"
            per = json.loads((E45 / "scores_summary_s2.json").read_text(encoding="utf-8"))[key]
            worst = max(worst, abs(per["csi_at_2.0"] - pub["csi_at_2.0"]["s%d" % seed]))
        worst = max(worst, abs(med([pub["csi_at_2.0"]["s%d" % s] for s in (0, 1, 2)])
                                - pub["median_csi_at_2.0"]))
    out["headline_table_max_abs_diff"] = worst
    return out


# --------------------------------------------------------------------------
# 2. load the epoch-axis products
# --------------------------------------------------------------------------
def load_axis():
    """-> axis[arm][seed][epoch] = {csi,bias,pod,far,...}"""
    axis = {}
    ev_dir = E45 / "epoch_axis" / "evals"
    if not ev_dir.is_dir():
        return None
    for path in sorted(ev_dir.glob("*.eval.json")):
        tag = path.name[: -len(".eval.json")]
        m = re.match(r"^(.+)_s(\d+)_ep(\d\d)$", tag)
        if not m:
            continue
        arm, seed, ep = m.group(1), int(m.group(2)), m.group(3)
        d = json.loads(path.read_text(encoding="utf-8"))
        t = d["thresholds"]["2.0"]
        axis.setdefault(arm, {}).setdefault(seed, {})[ep] = {
            "csi": t["csi"], "bias": t["bias"], "pod": t["pod"], "far": t["far"],
            "ets": t.get("ets"), "hss": t.get("hss"), "fss": t.get("fss"),
            "n_samples": d.get("n_samples"),
        }
    return axis or None


def load_probe():
    """-> probe[arm][seed][epoch] = {read-out: value}"""
    pr = {}
    p_dir = E45 / "epoch_axis" / "probe"
    if not p_dir.is_dir():
        return None
    for path in sorted(p_dir.glob("diag_probe_s*_ep*.json")):
        m = re.match(r"^diag_probe_s(\d+)_ep(\d\d)\.json$", path.name)
        if not m:
            continue
        seed, ep = int(m.group(1)), m.group(2)
        d = json.loads(path.read_text(encoding="utf-8"))
        for arm, rec in (d.get("arms") or {}).items():
            if not isinstance(rec, dict):
                continue
            pr.setdefault(arm, {}).setdefault(seed, {})[ep] = rec
    return pr or None


# --------------------------------------------------------------------------
# 3. the epoch axis
# --------------------------------------------------------------------------
def epoch_axis(axis):
    """Median + pass rate per arm per epoch, and the paired gain vs the
    control at the SAME epoch (seed as a common block)."""
    res = {}
    for arm in ARMS:
        if arm not in axis:
            continue
        res[arm] = {}
        for ep in EPOCHS:
            csis = [axis[arm][s][ep]["csi"] for s in sorted(axis[arm]) if ep in axis[arm][s]]
            if not csis:
                continue
            biases = [axis[arm][s][ep]["bias"] for s in sorted(axis[arm]) if ep in axis[arm][s]]
            # paired gain against the control at the same seed and epoch
            paired = []
            for s in sorted(axis[arm]):
                if ep not in axis[arm][s] or "B0" not in axis:
                    continue
                if s not in axis.get("B0", {}) or ep not in axis["B0"][s]:
                    continue
                paired.append(axis[arm][s][ep]["csi"] - axis["B0"][s][ep]["csi"])
            res[arm][ep] = {
                "n_seeds": len(csis),
                "median_csi": med(csis),
                "median_bias": med(biases),
                "min_csi": min(csis),
                "max_csi": max(csis),
                "pass_rate_over_floor": sum(1 for c in csis if c >= FLOOR) / len(csis),
                "per_seed_csi": csis,
                "paired_gain_vs_ctrl": med(paired),
                "paired_gain_signs": sum(1 for x in paired if x > 0),
                "paired_gain_n": len(paired),
            }
    return res


def zero_model_floor(axis):
    """The control arm compared with ITSELF under a different seed: pure
    seed noise, no mechanism.  (Round 46/57's calibration.)"""
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = {}
    for ep in EPOCHS:
        vals = []
        for a, b in pairs:
            if "B0" not in axis:
                continue
            if a in axis["B0"] and b in axis["B0"]:
                if ep in axis["B0"][a] and ep in axis["B0"][b]:
                    vals.append(abs(axis["B0"][a][ep]["csi"] - axis["B0"][b][ep]["csi"]))
        if vals:
            out[ep] = {"per_pair": vals, "mean": sum(vals) / len(vals), "max": max(vals)}
    return out


# --------------------------------------------------------------------------
# 4. the mechanism read-outs across epochs
# --------------------------------------------------------------------------
GATE_READS = ["corr_mask_amp", "mask_at_peak", "mask_dry_frac_lt_0p1", "corr_mask_resid"]
TOPO_READS = ["grad_top1pct_share", "field_max_mm", "tail_max_over_p99"]
AMT_READS = ["scale_cv", "scale_truth_corr", "scale_is_constant", "scale_mean"]


def probe_across_epochs(probe):
    """Per family, per epoch: median over seeds of each read-out."""
    out = {}
    for arm, byseed in (probe or {}).items():
        out[arm] = {}
        for ep in EPOCHS:
            vals = {}
            for key in GATE_READS + TOPO_READS + AMT_READS + ["maxratio_as_trained"]:
                xs = []
                for seed, byep in byseed.items():
                    if ep in byep and isinstance(byep[ep], dict):
                        v = byep[ep].get(key)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            xs.append(float(v))
                if xs:
                    vals[key] = med(xs)
            if vals:
                out[arm][ep] = vals
    return out


# --------------------------------------------------------------------------
def main():
    out_path = E45 / "round65_epoch_axis.json"
    result = {"purpose": "E45 round 65 -- the epoch axis (first judgement)",
              "is_post_hoc_diagnostic": True,
              "creates_criterion": False,
              "changes_e45_verdict": False}

    log("=== GUARD: E45's published numbers ===")
    g = guard_published()
    result["guard"] = g
    log(f"  S3 intervals: {g['s3_intervals_crossing_zero']}/{g['s3_intervals_total']} cross zero")
    log(f"  headline table max|diff| = {g['headline_table_max_abs_diff']:.3e}")
    assert g["s3_all_cross_zero"], "GUARD FAILED: an S3 interval stopped crossing zero"
    assert g["headline_table_max_abs_diff"] < 1e-9, "GUARD FAILED: headline table moved"

    axis = load_axis()
    if axis is None:
        log("\n=== epoch-axis products are not local yet ===")
        result["status"] = "products_not_available"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"written: {out_path}")
        return

    log("\n=== the epoch axis ===")
    ax = epoch_axis(axis)
    result["epoch_axis"] = ax
    floors = zero_model_floor(axis)
    result["zero_model_floor"] = floors

    for ep in EPOCHS:
        fl = floors.get(ep)
        log(f"\n-- epoch {ep} --" + (f"  zero-model |dCSI| floor: mean {fl['mean']:.6f} max {fl['max']:.6f}"
                                     if fl else "  (no zero-model pair)"))
        log(f"{'arm':10s} {'n':>2s} {'medianCSI':>10s} {'medianBIAS':>10s} {'pass':>6s} "
            f"{'pairedGain':>11s} {'signs':>6s}")
        for arm in ARMS:
            if arm not in ax or ep not in ax[arm]:
                continue
            r = ax[arm][ep]
            log(f"{arm:10s} {r['n_seeds']:2d} {r['median_csi']:10.6f} {r['median_bias']:10.4f} "
                f"{r['pass_rate_over_floor']:6.2f} {r['paired_gain_vs_ctrl']:11.6f} "
                f"{r['paired_gain_signs']:3d}/{r['paired_gain_n']}")

    # the epoch-00 control: does a re-run reproduce E45's published cells?
    control = {}
    pub = json.loads((E45 / "scores_summary_s2.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        diffs = []
        for seed in (0, 1, 2):
            key = f"{arm}_s{seed}_ep00"
            if arm in axis and seed in axis[arm] and "00" in axis[arm][seed]:
                diffs.append(axis[arm][seed]["00"]["csi"] - pub[key]["csi_at_2.0"])
        if diffs:
            control[arm] = {"max_abs_diff": max(abs(d) for d in diffs), "diffs": diffs}
    result["epoch00_control_reproduction"] = control
    worst_ctrl = max([v["max_abs_diff"] for v in control.values()], default=float("nan"))
    log(f"\n=== epoch-00 re-run vs E45's published cells: max|diff| = {worst_ctrl:.3e} ===")

    # the epoch difference itself, within seed (seed is the common block)
    log("\n=== epoch 01/02 minus epoch 00, within seed (paired) ===")
    epdiff = {}
    for arm in ARMS:
        epdiff[arm] = {}
        for ep in ("01", "02"):
            ds = []
            for seed in sorted(axis.get(arm, {})):
                a = axis[arm][seed]
                if "00" in a and ep in a:
                    ds.append(a[ep]["csi"] - a["00"]["csi"])
            if ds:
                epdiff[arm][ep] = {"per_seed": ds, "median": med(ds),
                                   "signs_positive": sum(1 for d in ds if d > 0),
                                   "n": len(ds)}
                log(f"  {arm:10s} ep{ep}: median dCSI = {med(ds):+.6f}  "
                    f"signs +: {sum(1 for d in ds if d > 0)}/{len(ds)}")
    result["epoch_minus_epoch00"] = epdiff

    # does the best epoch vary by arm?  (the "ep00 is the peak" claim)
    best_epoch = {}
    for arm in ARMS:
        if arm not in ax:
            continue
        cand = {ep: ax[arm][ep]["median_csi"] for ep in EPOCHS if ep in ax[arm]}
        if cand:
            b = max(cand, key=lambda e: cand[e])
            best_epoch[arm] = {"median_by_epoch": cand, "best": b,
                               "ep00_is_best": b == "00"}
    result["best_epoch_by_arm"] = best_epoch
    log("\n=== which epoch is the peak (by median CSI@2.0)? ===")
    for arm, rec in best_epoch.items():
        log(f"  {arm:10s} best = ep{rec['best']}   "
            f"({', '.join(f'ep{e}={v:.6f}' for e, v in rec['median_by_epoch'].items())})")

    # mechanism read-outs across epochs
    probe = load_probe()
    if probe:
        log("\n=== mechanism read-outs across epochs (median over seeds) ===")
        pa = probe_across_epochs(probe)
        result["mechanism_across_epochs"] = pa
        for arm in MECH_ARMS:
            if arm not in pa:
                continue
            for ep in EPOCHS:
                if ep not in pa[arm]:
                    continue
                v = pa[arm][ep]
                bits = []
                if arm in ("A1", "A2"):
                    for k in GATE_READS + ["maxratio_as_trained"]:
                        if k in v:
                            bits.append(f"{k}={v[k]:.4g}")
                if arm in ("A3a", "A3b"):
                    for k in TOPO_READS:
                        if k in v:
                            bits.append(f"{k}={v[k]:.4g}")
                if arm in ("A4", "A4b"):
                    for k in AMT_READS:
                        if k in v:
                            bits.append(f"{k}={v[k]:.4g}")
                if bits:
                    log(f"  {arm:6s} ep{ep}: " + "  ".join(bits))

    # verdict discipline: the epoch axis is post-hoc, it does not promote
    result["verdict"] = {
        "e45_verdict_unchanged": True,
        "qualified_arm_set_still_empty": True,
        "no_arm_promoted_to_ten_seeds": True,
        "note": ("post-hoc diagnostic: the epoch axis asks whether the epoch choice "
                 "was sound; it creates no criterion and promotes no arm"),
    }
    result["status"] = "complete"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
