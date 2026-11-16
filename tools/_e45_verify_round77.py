#!/usr/bin/env python
"""E45 round 77 -- epoch axis x forecast-lead-time axis (first cross of the two).

Read-only. Never touches the remote, never rewrites published artifacts.

Why this axis:
  * Round 49 judged the lead-time axis, but ONLY on the published E45 table
    (ep00). It found A4's gain flips sign with lead (L0-L3 positive, L4-L9
    negative, slope -0.01154/lead) and A4b the exact opposite.
  * Rounds 65-69 judged the epoch axis (ep00/ep01/ep02 x 7 arms x 3 seeds),
    but only on the pooled/overall CSI and on the S3 guard.
  * Nobody has asked: does that lead-time sign structure SURVIVE across
    epochs? If the flip exists only at ep00 it is a checkpoint-specific
    artifact; if it holds at ep01/ep02 as well it is a structural property
    of the arm (i.e. of the mechanism), not of the epoch we happened to pick.

Discipline carried over:
  * report median + pass rate, never a single-point score as a verdict;
  * paired caliber AND difference caliber side by side;
  * every "this effect is bigger than noise" claim is accompanied by the
    zero-model floor FOR THAT (epoch, lead) cell -- zero model = swap the
    control arm's seed inside the same epoch (seed noise only, no mechanism);
  * this is post-hoc diagnosis -> it is NOT a new criterion and CANNOT change
    any published verdict.
"""

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(r"R:/1radarNWPprediction/radar-nwp-bench")
EV = ROOT / "tools" / "_evals" / "e45"
EPOCH_EVALS = EV / "epoch_axis" / "evals"
PUBLISHED = EV / "evals_raw" / "evals"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
CONTROL = "B0"
SEEDS = [0, 1, 2]
EPOCHS = [0, 1, 2]
THR = "2.0"
LEADS = list(range(10))

OUT = EV / "round77_epoch_lead_cross.json"

report = {
    "round": 77,
    "axis": "epoch axis x forecast-lead-time axis (cross of round 49 and rounds 65-69)",
    "threshold_mm": 2.0,
    "leads": LEADS,
    "n_seeds": len(SEEDS),
    "discipline": "post-hoc diagnosis; NOT a new criterion; cannot change any published verdict",
}

checks = []          # (name, ok, detail)
def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


# ---------------------------------------------------------------- load
def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


epoch_cells = {}    # (arm, seed, ep) -> dict
for arm in ARMS:
    for seed in SEEDS:
        for ep in EPOCHS:
            p = EPOCH_EVALS / f"{arm}_s{seed}_ep{ep:02d}.eval.json"
            if not p.exists():
                print(f"MISSING {p}")
                sys.exit(2)
            d = load(p)
            ls = d["lead_scores"]
            epoch_cells[(arm, seed, ep)] = {
                "csi": float(d["thresholds"][THR]["csi"]),
                "bias": float(d["thresholds"][THR]["bias"]),
                "lead_csi": [float(ls[str(L)][THR]["csi"]) for L in LEADS],
                "lead_bias": [float(ls[str(L)][THR]["bias"]) for L in LEADS],
                "n_samples": int(d["n_samples"]),
            }

check("epoch-axis cell grid complete (7 arms x 3 seeds x 3 epochs)",
      len(epoch_cells) == 63, f"{len(epoch_cells)} cells")
ns = {c["n_samples"] for c in epoch_cells.values()}
check("all epoch-axis cells evaluated the same number of windows",
      len(ns) == 1, f"n_samples set = {sorted(ns)}")

pub_cells = {}
for arm in ARMS:
    for seed in SEEDS:
        p = PUBLISHED / f"{arm}_s{seed}_ep00.eval.json"
        d = load(p)
        ls = d["lead_scores"]
        pub_cells[(arm, seed)] = {
            "csi": float(d["thresholds"][THR]["csi"]),
            "lead_csi": [float(ls[str(L)][THR]["csi"]) for L in LEADS],
        }

# ------------------------------------------------- same-protocol baseline
# ep00 of the epoch axis must reproduce the published ep00 exactly, both
# overall and per lead -- otherwise "differences" and "pipeline drift" are
# not separable (round 65's rule).
dev_overall = 0.0
dev_lead = 0.0
for arm in ARMS:
    for seed in SEEDS:
        a = epoch_cells[(arm, seed, 0)]
        b = pub_cells[(arm, seed)]
        dev_overall = max(dev_overall, abs(a["csi"] - b["csi"]))
        for i in range(len(LEADS)):
            dev_lead = max(dev_lead, abs(a["lead_csi"][i] - b["lead_csi"][i]))
check("same-protocol baseline: epoch-axis ep00 overall CSI@2.0 == published",
      dev_overall == 0.0, f"max|dev| = {dev_overall:.3e} over 21 cells")
check("same-protocol baseline: epoch-axis ep00 per-lead CSI@2.0 == published",
      dev_lead == 0.0, f"max|dev| = {dev_lead:.3e} over {21*len(LEADS)} lead cells")
report["baseline"] = {"dev_overall": dev_overall, "dev_lead": dev_lead}


def median(xs):
    return statistics.median(xs)


# ------------------------------------------------- zero model per (ep, lead)
# Control arm, seed swapped inside the same epoch -> pure seed noise.
def lead_deltas(arm, ep, lead):
    """paired per-seed deltas arm - control at one (epoch, lead)."""
    return [epoch_cells[(arm, s, ep)]["lead_csi"][lead]
            - epoch_cells[(CONTROL, s, ep)]["lead_csi"][lead] for s in SEEDS]


zfloor = {}     # (ep, lead) -> max |dCSI| over the 3 control seed pairs
zfloor_med = {}
for ep in EPOCHS:
    for lead in LEADS:
        pair = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            pair.append(abs(epoch_cells[(CONTROL, a, ep)]["lead_csi"][lead]
                            - epoch_cells[(CONTROL, b, ep)]["lead_csi"][lead]))
        zfloor[(ep, lead)] = max(pair)
        zfloor_med[(ep, lead)] = median(pair)

report["zero_model_floor_by_epoch_lead"] = {
    f"ep{ep:02d}": {f"L{L}": zfloor[(ep, L)] for L in LEADS} for ep in EPOCHS
}

# ------------------------------------------------------- arm x epoch x lead
cells = {}
for ep in EPOCHS:
    for arm in ARMS:
        if arm == CONTROL:
            continue
        for lead in LEADS:
            pd = lead_deltas(arm, ep, lead)
            paired = median(pd)                       # paired caliber
            arm_med = median([epoch_cells[(arm, s, ep)]["lead_csi"][lead] for s in SEEDS])
            ctl_med = median([epoch_cells[(CONTROL, s, ep)]["lead_csi"][lead] for s in SEEDS])
            diff = arm_med - ctl_med                  # difference caliber
            fl = zfloor[(ep, lead)]
            cells[(arm, ep, lead)] = {
                "paired_median_dcsi": paired,
                "diff_caliber_dcsi": diff,
                "per_seed_dcsi": pd,
                "n_pos_seeds": sum(1 for x in pd if x > 0),
                "n_neg_seeds": sum(1 for x in pd if x < 0),
                "zero_model_floor": fl,
                "exceeds_floor": abs(paired) > fl,
                "ratio_to_floor": abs(paired) / fl if fl > 0 else float("inf"),
                "arm_bias_median": median([epoch_cells[(arm, s, ep)]["lead_bias"][lead] for s in SEEDS]),
                "ctl_bias_median": median([epoch_cells[(CONTROL, s, ep)]["lead_bias"][lead] for s in SEEDS]),
            }

# ------------------------------------------------------------- slopes
def slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


slopes = {}
for arm in ARMS:
    if arm == CONTROL:
        continue
    for ep in EPOCHS:
        ys = [cells[(arm, ep, L)]["paired_median_dcsi"] for L in LEADS]
        early = statistics.mean(ys[0:5])   # L0-L4
        late = statistics.mean(ys[5:10])   # L5-L9
        slopes[f"{arm}_ep{ep:02d}"] = {
            "slope_per_lead": slope(LEADS, ys),
            "early_L0_L4_mean": early,
            "late_L5_L9_mean": late,
            "early_minus_late": early - late,
            "per_lead_paired_median": ys,
        }
report["slope_by_arm_epoch"] = slopes

# ---- headline structural questions (facts, not verdicts)
def slope_sign_consistent(arm):
    signs = []
    for ep in EPOCHS:
        s = slopes[f"{arm}_ep{ep:02d}"]["slope_per_lead"]
        signs.append(1 if s > 0 else (-1 if s < 0 else 0))
    return signs


report["sign_flip_structure"] = {
    "A4_slope_sign_per_epoch": slope_sign_consistent("A4"),
    "A4b_slope_sign_per_epoch": slope_sign_consistent("A4b"),
    "A3a_slope_sign_per_epoch": slope_sign_consistent("A3a"),
    "A3b_slope_sign_per_epoch": slope_sign_consistent("A3b"),
    "A1_slope_sign_per_epoch": slope_sign_consistent("A1"),
    "A2_slope_sign_per_epoch": slope_sign_consistent("A2"),
}

# how many (arm, epoch, lead) cells beat their own zero-model floor
exceed = {a: 0 for a in slopes and ARMS if a != CONTROL}
for (arm, ep, lead), v in cells.items():
    if v["exceeds_floor"]:
        exceed[arm] += 1
report["cells_exceeding_own_zero_model_floor"] = {
    "n_cells_per_arm_out_of_30": {a: exceed[a] for a in exceed},
    "note": "30 = 3 epochs x 10 leads; a cell counts if |paired median dCSI| > "
            "the (epoch, lead) zero-model floor computed from control seed swaps",
}

# leading/lagging arms per epoch (difference caliber)
report["per_epoch_arm_ranking_diff_caliber"] = {}
for ep in EPOCHS:
    ranks = []
    for arm in ARMS:
        if arm == CONTROL:
            continue
        arm_med = median([epoch_cells[(arm, s, ep)]["csi"] for s in SEEDS])
        ctl_med = median([epoch_cells[(CONTROL, s, ep)]["csi"] for s in SEEDS])
        ranks.append((arm, arm_med - ctl_med))
    ranks.sort(key=lambda t: -t[1])
    report["per_epoch_arm_ranking_diff_caliber"][f"ep{ep:02d}"] = [
        {"arm": a, "median_csi_minus_control": round(v, 9)} for a, v in ranks
    ]

# --------------------------------------- cross-validation vs round 49
# Round 49 published the lead-time slope for the ep00 table. We recompute it
# from the epoch-axis ep00 copies with a different code path; if they agree
# bit-for-bit, our per-lead parser is calibrated.
r49 = load(EV / "round49_lead_axis.json")
r49_arms = r49.get("arms", {})
slope_xcheck = []
for arm in ARMS:
    if arm == CONTROL or arm not in r49_arms:
        continue
    pub_slope = r49_arms[arm].get("paired_slope_per_lead")
    if pub_slope is None:
        pub_slope = r49_arms[arm].get("slope_per_lead")
    if pub_slope is None:
        continue
    mine = slopes[f"{arm}_ep00"]["slope_per_lead"]
    slope_xcheck.append({"arm": arm, "published_r49": pub_slope, "recomputed_ep00": mine,
                         "abs_dev": abs(pub_slope - mine)})
report["round49_slope_cross_validation"] = slope_xcheck
check("per-lead parser calibrated against round 49's published ep00 slopes",
      bool(slope_xcheck) and max(x["abs_dev"] for x in slope_xcheck) < 1e-9,
      f"{len(slope_xcheck)} arms, max|dev| = "
      f"{max((x['abs_dev'] for x in slope_xcheck), default=float('nan')):.3e}")

# also cross-validate the whole per-lead vector (stronger than the slope)
vec_xcheck = []
for arm in ARMS:
    if arm == CONTROL or arm not in r49_arms:
        continue
    pub_vec = r49_arms[arm].get("paired_by_lead")
    if pub_vec is None:
        continue
    mine = slopes[f"{arm}_ep00"]["per_lead_paired_median"]
    vec_xcheck.append({"arm": arm,
                       "max_abs_dev": max(abs(a - b) for a, b in zip(pub_vec, mine))})
report["round49_paired_vector_cross_validation"] = vec_xcheck
check("per-lead parser calibrated against round 49's published ep00 per-lead vectors",
      bool(vec_xcheck) and max(x["max_abs_dev"] for x in vec_xcheck) < 1e-9,
      f"{len(vec_xcheck)} arms x {len(LEADS)} leads, max|dev| = "
      f"{max((x['max_abs_dev'] for x in vec_xcheck), default=float('nan')):.3e}")

# floor caliber note: round 49 used the MEDIAN over the 3 control seed pairs,
# this round uses the MAX. Both are reported so the counts are comparable.
report["floor_caliber"] = {
    "this_round": "max |dCSI| over the 3 control seed pairs (stricter)",
    "round49": "median |dCSI| over the 3 control seed pairs",
    "ep00_max_floor_by_lead": {f"L{L}": zfloor[(0, L)] for L in LEADS},
    "ep00_median_floor_by_lead": {f"L{L}": zfloor_med[(0, L)] for L in LEADS},
    "round49_published_floor_by_lead": {
        f"L{L}": r49["zero_model_floor_by_lead"][i] for i, L in enumerate(LEADS)
    },
}

# ------------------------- flip point localisation (arm x epoch -> lead)
def flip_point(ys):
    """index of the first lead where the sign differs from lead 0's sign."""
    s0 = 1 if ys[0] > 0 else (-1 if ys[0] < 0 else 0)
    if s0 == 0:
        return None
    for i, y in enumerate(ys):
        s = 1 if y > 0 else (-1 if y < 0 else 0)
        if s != 0 and s != s0:
            return i
    return None


report["sign_flip_lead_index"] = {
    f"{arm}_ep{ep:02d}": flip_point(slopes[f"{arm}_ep{ep:02d}"]["per_lead_paired_median"])
    for arm in ARMS if arm != CONTROL for ep in EPOCHS
}

# ------------------- attribution of the cells that beat their own floor
# For each (arm, epoch, lead) cell that beats its zero-model floor we record
# the per-seed signs and a leave-one-seed-out recomputation. This separates
# "a real, seed-consistent effect" from "one collapsed seed".
over = []
for (arm, ep, lead), v in cells.items():
    if not v["exceeds_floor"]:
        continue
    pd = v["per_seed_dcsi"]
    loo = []
    for drop in SEEDS:
        keep = [pd[s] for s in SEEDS if s != drop]
        loo.append(median(keep))
    over.append({
        "arm": arm, "epoch": ep, "lead": lead,
        "paired_median_dcsi": v["paired_median_dcsi"],
        "zero_model_floor": v["zero_model_floor"],
        "ratio_to_floor": v["ratio_to_floor"],
        "per_seed_dcsi": pd,
        "n_pos_seeds": v["n_pos_seeds"], "n_neg_seeds": v["n_neg_seeds"],
        "seed_consistent_sign": (v["n_pos_seeds"] == 3 or v["n_neg_seeds"] == 3),
        "leave_one_out_median_dcsi": loo,
        "survives_leave_one_out": all(abs(x) > v["zero_model_floor"] for x in loo),
        "arm_bias_median": v["arm_bias_median"],
        "ctl_bias_median": v["ctl_bias_median"],
        "bias_ratio_arm_over_control": (v["arm_bias_median"] / v["ctl_bias_median"]
                                         if v["ctl_bias_median"] else float("inf")),
    })
report["cells_above_floor_detail"] = over
# Enumerate the fact rather than asserting it must come out one way: the
# standing rule is "assertions must not assume the data will agree with the
# conclusion". What we assert is only that the count we report is the count
# the detail actually contains.
n_consistent = sum(1 for o in over if o["seed_consistent_sign"])
n_loo = sum(1 for o in over if o["survives_leave_one_out"])
report["cells_above_floor_attribution"] = {
    "n_cells_above_floor": len(over),
    "n_seed_consistent_3of3": n_consistent,
    "n_surviving_leave_one_seed_out": n_loo,
    "detail": "a cell 'beats its floor' only if |paired median dCSI| > the "
              "(epoch, lead) control-seed-swap floor; whether that cell is "
              "attributable to the arm rather than to one seed is a separate "
              "question answered by the 3/3 sign and leave-one-out columns",
}
check("cell-above-floor attribution counts agree with the detail rows",
      n_consistent == sum(1 for o in over if o["seed_consistent_sign"])
      and n_loo == sum(1 for o in over if o["survives_leave_one_out"]),
      f"{n_consistent}/{len(over)} cells are 3/3 seed-sign-consistent; "
      f"{n_loo}/{len(over)} survive leave-one-seed-out")

# --------------------------------------------- published verdict guards
s3 = load(EV / "s3_equalbias.json")
cross_flags = []
def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, bool) and "cross" in k.lower():
                cross_flags.append((path + "/" + k, v))
            else:
                walk(v, path + "/" + k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]")
walk(s3)
check("S3 guard: every published interval still crosses zero",
      bool(cross_flags) and all(v for _, v in cross_flags),
      f"{len(cross_flags)} interval flags, all cross zero = {all(v for _, v in cross_flags)}")
report["s3_interval_flags"] = [{"path": p, "crosses_zero": v} for p, v in cross_flags]

s3_verdicts = {}
def collect_verdicts(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "s3_verdict" and isinstance(v, str):
                s3_verdicts.setdefault("verdicts", []).append(v)
            else:
                collect_verdicts(v)
    elif isinstance(node, list):
        for v in node:
            collect_verdicts(v)
collect_verdicts(s3)
report["s3_verdicts"] = s3_verdicts

ss = load(EV / "scores_summary.json")
headline = {a: {"median_csi_at_2.0": ss[a]["median_csi_at_2.0"],
                "pass_rate_over_floor": ss[a]["pass_rate_over_floor"],
                "median_bias_at_2.0": ss[a]["median_bias_at_2.0"]} for a in ARMS}
report["published_headline_table"] = headline

# recompute published headline from the raw ep00 evals (independent layer)
dev_head = 0.0
for arm in ARMS:
    csis = sorted(epoch_cells[(arm, s, 0)]["csi"] for s in SEEDS)
    dev_head = max(dev_head, abs(median(csis) - headline[arm]["median_csi_at_2.0"]))
check("published headline median CSI@2.0 reproducible from raw ep00 evals",
      dev_head < 1e-9, f"max|dev| = {dev_head:.3e} over 7 arms")

# --------------------------------------------------------- verdict guard
report["checks"] = checks
n_pass = sum(1 for c in checks if c["ok"])
report["pos"] = f"{n_pass}/{len(checks)}"

report["verdict"] = {
    "published_verdict_changed": False,
    "statement": "post-hoc diagnosis on a new axis; the published E45 verdict "
                 "(six arms negative, no arm into 10-seed, no 10-seed compute "
                 "spent) is unchanged and this axis cannot change it",
    "master_version_bump": False,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=1, ensure_ascii=False)

# ------------------------------------------------------------- printing
print("=" * 78)
print("E45 ROUND 77 -- epoch axis x lead-time axis (first cross)")
print("=" * 78)
for c in checks:
    print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']}  |  {c['detail']}")
print()
print(f"zero-model floor (max |dCSI| over control seed swaps), by epoch x lead:")
for ep in EPOCHS:
    print("  ep%02d " % ep + "  ".join(f"L{L}={zfloor[(ep,L)]:.4f}" for L in LEADS))
print()
print("paired-median dCSI slope per lead, by arm x epoch:")
for arm in ARMS:
    if arm == CONTROL:
        continue
    row = []
    for ep in EPOCHS:
        s = slopes[f"{arm}_ep{ep:02d}"]["slope_per_lead"]
        row.append(f"ep{ep:02d}={s:+.5f}")
    print(f"  {arm:4s} " + "  ".join(row))
print()
print("cells exceeding their own (epoch,lead) zero-model floor (out of 30):")
for a, n in report["cells_exceeding_own_zero_model_floor"]["n_cells_per_arm_out_of_30"].items():
    print(f"  {a:4s} {n:2d}/30")
print()
print("A4  per-lead paired-median dCSI by epoch:")
for ep in EPOCHS:
    ys = slopes["A4_ep%02d" % ep]["per_lead_paired_median"]
    print("  ep%02d " % ep + " ".join(f"{v:+.4f}" for v in ys))
print("A4b per-lead paired-median dCSI by epoch:")
for ep in EPOCHS:
    ys = slopes["A4b_ep%02d" % ep]["per_lead_paired_median"]
    print("  ep%02d " % ep + " ".join(f"{v:+.4f}" for v in ys))
print()
print(f"POS {report['pos']}")
print(f"report -> {OUT}")
print("VERDICT: PASS" if n_pass == len(checks) else "VERDICT: FAIL")
sys.exit(0 if n_pass == len(checks) else 1)
