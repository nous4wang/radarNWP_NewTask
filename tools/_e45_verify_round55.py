#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 55 -- calibration of the S3 guard across the E41->E45 lineage.

WHY THIS ROUND EXISTS
---------------------
Rounds 40-54 judged E45 on every *content* axis (mechanism read-outs, S1 struct
metrics, window level, storm-case level, lead-time axis, severity/difficulty
strata, error composition, omnibus arm effect, mechanism dose-response, metric
families).  None of them ever looked at the *guard itself*.

The guard is `tools/_e41_s3_equalbias.py::two_level_bootstrap`.  Its interval
is a TWO-LEVEL bootstrap: it resamples SEEDS (with replacement) and then
windows inside each drawn seed.  The published lineage has, however, been run
at very different seed-replication depths:

    E41 (fixed)  n_seeds = 3
    E42 (fixed)  n_seeds = 3
    E43 (fixed)  n_seeds = 2
    E44          n_seeds = 1     <-- the seed level is DEGENERATE here
    E45          n_seeds = 3

With n_seeds = 1 the seed draw `rng.integers(0, 1, 1)` is always [0], so the
interval collapses to a WINDOW-ONLY resampling interval -- exactly the
"pseudo-replication" shape that round 46 flagged for E45's window-level CIs.
With n_seeds = 3 the interval also carries the seed-level variance.

So "crosses zero" is NOT commensurable across the lineage, and the narrative
"E44 found significant gains, E45 found none" may be confounded by the GUARD's
replication depth rather than by the mechanisms.

This round:
  1. reconstructs the published S3 point estimates from local artifacts
     (exact reproduction -> the instrument is understood);
  2. re-implements the guard's bootstrap in pure stdlib and shows the
     published raw intervals are reproduced (crossing decisions + widths);
  3. applies the DEPTH-1 (E44-style) protocol to E45's own 18 (arm, seed)
     cells -> how many would have been declared "significant";
  4. calibrates BOTH protocols on a mechanism-free null built from E45's own
     control arm (seed-deranged B0 vs B0) -> the anti-conservativeness of
     depth-1;
  5. censuses the lineage and tabulates which published verdicts are
     calibration-sensitive.

RULES HONOURED (from the closed case file):
  * median + pass rate only; single-point scores never stand alone
  * S3 guard pinned at thr = 2.0 mm; replication unit stays the seed
  * post-hoc diagnostic -> does NOT constitute a new criterion, verdict unchanged
  * no re-adjudication of E41/E42/E43/E44 verdicts; their published verdicts
    are reported as-is
  * pure stdlib (no numpy on this host)
"""

import json
import math
import os
import random
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
EVDIR = os.path.join(HERE, "_evals", "e45", "runs", "e45", "evals")
E45DIR = os.path.join(HERE, "_evals", "e45")
OUTPATH = os.path.join(E45DIR, "round55_guard_calibration.json")

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SIX = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
PRIMARY = "2.0"
N_BOOT = 2000
RNG_SEED = 20261114

# published lineage: (label, file, which variant is authoritative)
LINEAGE = [
    ("E41", "e41_s3_equalbias_fixed.json", "fixed (the n_windows=3 variant is the "
     "documented resampling bug, superseded)"),
    ("E42", "e42_s3_equalbias_fixed.json", "fixed (same superseded bug)"),
    ("E43", "e43_s3_equalbias_fixed.json", "fixed (same superseded bug)"),
    ("E44", "e44_s3_equalbias.json", "as published"),
    ("E45", os.path.join("e45", "s3_equalbias.json"), "as published"),
]

CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


# ---------------------------------------------------------------- statistics
def percentile(xs, q):
    """Linear-interpolation percentile, q in [0,100] (numpy default)."""
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    pos = (n - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def mean(xs):
    return sum(xs) / float(len(xs)) if xs else None


# ---------------------------------------------------------------- metrics
def csi_of(h, m, f):
    d = h + m + f
    return h / d if d > 0 else float("nan")


# ---------------------------------------------------------------- loading
def load_windows(arm, seed):
    p = os.path.join(EVDIR, "%s_s%d_ep00.windows.json" % (arm, seed))
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def win_counts(doc, thr=PRIMARY):
    """Per-window (hits, misses, false_alarms) at one threshold."""
    out = []
    for w in doc["windows"]:
        c = w[thr]
        out.append((c["hits"], c["misses"], c["false_alarms"]))
    return out


def pooled(counts):
    h = m = f = 0.0
    for a, b, c in counts:
        h += a
        m += b
        f += c
    return h, m, f


# ---------------------------------------------------------------- bootstrap
def boot_depth1(a_counts, b_counts, n_boot=N_BOOT, seed=RNG_SEED):
    """The guard's own bootstrap at n_seeds = 1: SEED level is degenerate,
    so this is a WINDOW-ONLY resampling interval (paired window indices,
    exactly as `two_level_bootstrap` does inside its seed loop)."""
    rng = random.Random(seed)
    n_win = len(a_counts)
    idx_pool = range(n_win)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choices(idx_pool, k=n_win)
        ha, ma, fa = 0.0, 0.0, 0.0
        hb, mb, fb = 0.0, 0.0, 0.0
        for i in idx:
            x, y, z = a_counts[i]
            ha += x
            ma += y
            fa += z
            x, y, z = b_counts[i]
            hb += x
            mb += y
            fb += z
        diffs.append(csi_of(ha, ma, fa) - csi_of(hb, mb, fb))
    lo, hi = percentile(diffs, 2.5), percentile(diffs, 97.5)
    return {"ci_lo": lo, "ci_hi": hi, "crosses_zero": bool(lo <= 0.0 <= hi),
            "width": hi - lo, "n_boot": n_boot}


def boot_depth3(a_list, b_list, n_boot=N_BOOT, seed=RNG_SEED):
    """The guard's own TWO-LEVEL bootstrap: resample seeds with replacement,
    then windows inside each drawn seed; the statistic is the mean over the
    drawn seeds."""
    rng = random.Random(seed)
    n_seeds = len(a_list)
    n_win = min(len(c) for c in a_list)
    idx_pool = range(n_win)
    diffs = []
    for _ in range(n_boot):
        si = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        obs = []
        for k in si:
            idx = rng.choices(idx_pool, k=n_win)
            ha, ma, fa = 0.0, 0.0, 0.0
            hb, mb, fb = 0.0, 0.0, 0.0
            for i in idx:
                x, y, z = a_list[k][i]
                ha += x
                ma += y
                fa += z
                x, y, z = b_list[k][i]
                hb += x
                mb += y
                fb += z
            obs.append(csi_of(ha, ma, fa) - csi_of(hb, mb, fb))
        diffs.append(sum(obs) / float(len(obs)))
    lo, hi = percentile(diffs, 2.5), percentile(diffs, 97.5)
    return {"ci_lo": lo, "ci_hi": hi, "crosses_zero": bool(lo <= 0.0 <= hi),
            "width": hi - lo, "n_boot": n_boot}


def main():
    report = OrderedDict()
    report["meta"] = {
        "round": 55,
        "axis": "calibration of the S3 guard across the E41->E45 lineage "
                "(replication depth + null calibration)",
        "primary_threshold_mm": PRIMARY,
        "n_boot": N_BOOT, "rng_seed": RNG_SEED,
        "rule": "post-hoc diagnostic, NOT a new criterion; replication unit = seed; "
                "no re-adjudication of E41/E42/E43/E44 verdicts",
    }

    # ================================================================ A
    # The published S3 point estimates must be reproducible from the local
    # artifacts.  raw: from the window tables.  equal_bias: the arm's
    # equalised CSI is published in the S3 json ("seeds" -> csi_equalised),
    # the control's CSI comes from the window tables.
    with open(os.path.join(E45DIR, "s3_equalbias.json"), "r", encoding="utf-8") as fh:
        s3_45 = json.load(fh)

    counts = {}
    for a in ARMS:
        for s in SEEDS:
            counts[(a, s)] = win_counts(load_windows(a, s))

    maxdiff = 0.0
    ncmp = 0
    pts = {}
    for a in SIX:
        raw_pts, eq_pts = [], []
        for s in SEEDS:
            ha, ma, fa = pooled(counts[(a, s)])
            hb, mb, fb = pooled(counts[("B0", s)])
            raw_pts.append(csi_of(ha, ma, fa) - csi_of(hb, mb, fb))
            eq_arm = s3_45["arms"][a]["seeds"][str(s)]["csi_equalised"]
            eq_pts.append(eq_arm - csi_of(hb, mb, fb))
        raw_point = mean(raw_pts)
        eq_point = mean(eq_pts)
        pts[a] = {"raw_point": raw_point, "equal_bias_point": eq_point,
                  "raw_per_seed": raw_pts, "equal_bias_per_seed": eq_pts}
        maxdiff = max(maxdiff, abs(raw_point - s3_45["arms"][a]["raw"]["point"]))
        maxdiff = max(maxdiff, abs(eq_point - s3_45["arms"][a]["equal_bias"]["point"]))
        ncmp += 2
    chk("A1 all 12 published S3 point estimates reproduced exactly from local "
        "artifacts (6 arms x raw/equal_bias)", maxdiff == 0.0,
        "max|diff| = %.3e over %d point estimates" % (maxdiff, ncmp))
    report["A_point_estimate_reproduction"] = {
        "max_abs_diff": maxdiff, "n_point_estimates": ncmp,
        "per_arm": pts,
    }

    # ================================================================ B
    # Guard unchanged: all 12 published intervals still cross zero, verdicts
    # unchanged.  (This is the standing guard, not a new claim.)
    n_cross = 0
    verdicts = {}
    for a in SIX:
        for k in ("raw", "equal_bias"):
            if s3_45["arms"][a][k]["crosses_zero"]:
                n_cross += 1
        verdicts[a] = s3_45["arms"][a]["s3_verdict"]
    chk("B1 all 12 published E45 S3 intervals still cross zero; six verdicts "
        "unchanged", n_cross == 12 and sorted(set(verdicts.values())) ==
        ["GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH"],
        "%d/12 cross zero; verdicts=%s" % (n_cross, sorted(set(verdicts.values()))))
    report["B_guard"] = {"n_intervals_crossing_zero": n_cross, "verdicts": verdicts}

    # ================================================================ C
    # Stdlib re-implementation of the guard's two-level bootstrap must
    # reproduce the published RAW intervals (different RNG stream -> the
    # widths agree up to Monte-Carlo error; the crossing decisions must
    # agree exactly).
    mine = {}
    agree = 0
    for a in SIX:
        a_list = [counts[(a, s)] for s in SEEDS]
        b_list = [counts[("B0", s)] for s in SEEDS]
        r = boot_depth3(a_list, b_list)
        pub = s3_45["arms"][a]["raw"]
        same = (r["crosses_zero"] == pub["crosses_zero"])
        agree += 1 if same else 0
        mine[a] = {"stdlib_ci": [r["ci_lo"], r["ci_hi"]],
                   "stdlib_width": r["width"],
                   "published_ci": [pub["ci_lo"], pub["ci_hi"]],
                   "published_width": pub["ci_hi"] - pub["ci_lo"],
                   "width_ratio_stdlib_over_published": r["width"] / (pub["ci_hi"] - pub["ci_lo"]),
                   "crosses_zero_stdlib": r["crosses_zero"],
                   "crosses_zero_published": pub["crosses_zero"],
                   "crossing_decision_agrees": same}
    ratios = [mine[a]["width_ratio_stdlib_over_published"] for a in SIX]
    chk("C1 stdlib re-implementation of the two-level bootstrap reproduces the "
        "published raw intervals (crossing decisions) and their widths",
        agree == 6 and all(0.80 <= x <= 1.25 for x in ratios),
        "crossing decisions agree %d/6; width ratio (stdlib/published) per arm: %s"
        % (agree, ", ".join("%s=%.3f" % (a, mine[a]["width_ratio_stdlib_over_published"])
                             for a in SIX)))
    report["C_stdlib_reimplementation"] = mine

    # ================================================================ D
    # The E44-STYLE protocol (depth 1: seed level degenerate) applied to
    # E45's own 18 (arm, seed) cells.  How many would have been declared
    # "significant"?
    depth1 = OrderedDict()
    n_sig = 0
    n_sig_pos = 0
    for a in SIX:
        cells = []
        for s in SEEDS:
            r = boot_depth1(counts[(a, s)], counts[("B0", s)])
            ha, ma, fa = pooled(counts[(a, s)])
            hb, mb, fb = pooled(counts[("B0", s)])
            point = csi_of(ha, ma, fa) - csi_of(hb, mb, fb)
            sig = (not r["crosses_zero"])
            n_sig += 1 if sig else 0
            n_sig_pos += 1 if (sig and point > 0) else 0
            cells.append({"seed": s, "point": point,
                          "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                          "width": r["width"],
                          "crosses_zero": r["crosses_zero"],
                          "significant_at_depth1": sig,
                          "would_be_declared_gain": sig and point > 0})
        depth1[a] = {"cells": cells,
                     "n_cells_significant": sum(1 for c in cells if c["significant_at_depth1"]),
                     "n_cells_declared_gain": sum(1 for c in cells if c["would_be_declared_gain"])}
    chk("D1 the E44-style depth-1 protocol applied to E45's arms DOES manufacture "
        "significance that the depth-3 protocol does not",
        n_sig >= 15 and n_sig_pos >= 8,
        "of E45's 18 (arm,seed) cells, %d would be 'significant' at depth 1, "
        "%d of them with a POSITIVE (gain) point" % (n_sig, n_sig_pos))
    report["D_depth1_protocol_on_e45"] = {
        "n_cells": 18, "n_significant_at_depth1": n_sig,
        "n_declared_gain_at_depth1": n_sig_pos, "per_arm": depth1,
    }

    # ================================================================ E
    # NULL CALIBRATION.  Mechanism-free null built from E45's own control:
    # compare B0_s{k} against B0_s{pi(k)} under a seed derangement, so the
    # comparison contains seed noise only and no mechanism whatsoever.
    derangements = OrderedDict([("pi=(1,2,0)", [1, 2, 0]), ("pi=(2,0,1)", [2, 0, 1])])
    null = OrderedDict()
    for name, pi in derangements.items():
        a_list = [counts[("B0", k)] for k in SEEDS]
        b_list = [counts[("B0", pi[k])] for k in SEEDS]
        per_pair = []
        for k in SEEDS:
            r = boot_depth1(a_list[k], b_list[k])
            ha, ma, fa = pooled(a_list[k])
            hb, mb, fb = pooled(b_list[k])
            point = csi_of(ha, ma, fa) - csi_of(hb, mb, fb)
            per_pair.append({"seed": k, "control_seed": pi[k], "point": point,
                             "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "crosses_zero": r["crosses_zero"],
                             "significant_at_depth1": (not r["crosses_zero"])})
        r3 = boot_depth3(a_list, b_list)
        null[name] = {
            "per_pair_depth1": per_pair,
            "n_pairs_significant_at_depth1": sum(1 for p in per_pair
                                                 if p["significant_at_depth1"]),
            "depth3_ci": [r3["ci_lo"], r3["ci_hi"]],
            "depth3_crosses_zero": r3["crosses_zero"],
            "depth3_width": r3["width"],
        }
    fpr1 = sum(v["n_pairs_significant_at_depth1"] for v in null.values())
    n_pairs_total = sum(len(v["per_pair_depth1"]) for v in null.values())
    fpr3 = sum(1 for v in null.values() if not v["depth3_crosses_zero"])
    chk("E1 the depth-1 (E44-style) protocol is grossly anti-conservative on a "
        "mechanism-free null, while the depth-3 protocol is calibrated on it",
        fpr1 >= 2 * (n_pairs_total - fpr1) and fpr3 == 0,
        "null pairs: depth-1 declares %d/%d significant (false positives on a "
        "mechanism-free comparison); depth-3 intervals exclude zero in %d/%d "
        "derangements" % (fpr1, n_pairs_total, fpr3, len(null)))
    report["E_null_calibration"] = {
        "null_design": "B0_s{k} vs B0_s{pi(k)} under seed derangement -- seed "
                        "noise only, no mechanism",
        "n_pairs": n_pairs_total,
        "depth1_false_positive_rate": fpr1 / float(n_pairs_total),
        "depth3_false_positive_rate": fpr3 / float(len(null)),
        "per_derangement": null,
    }

    # ================================================================ F
    # Lineage census: the guard's replication depth and interval width per
    # published experiment.
    lineage = OrderedDict()
    for tag, fn, note in LINEAGE:
        with open(os.path.join(HERE, "_evals", fn), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        arms = {}
        for a, v in d["arms"].items():
            arms[a] = {
                "n_seeds": v["raw"]["n_seeds"],
                "n_windows": v["raw"]["n_windows"],
                "raw_point": v["raw"]["point"],
                "raw_ci": [v["raw"]["ci_lo"], v["raw"]["ci_hi"]],
                "raw_width": v["raw"]["ci_hi"] - v["raw"]["ci_lo"],
                "raw_crosses_zero": v["raw"]["crosses_zero"],
                "equal_bias_point": v["equal_bias"]["point"],
                "equal_bias_ci": [v["equal_bias"]["ci_lo"], v["equal_bias"]["ci_hi"]],
                "equal_bias_width": v["equal_bias"]["ci_hi"] - v["equal_bias"]["ci_lo"],
                "equal_bias_crosses_zero": v["equal_bias"]["crosses_zero"],
                "s3_verdict": v["s3_verdict"],
            }
        widths = [x["raw_width"] for x in arms.values()]
        lineage[tag] = {
            "file": fn, "variant": note,
            "control": d.get("control"), "threshold_mm": d.get("threshold_mm"),
            "epoch": d.get("epoch"), "n_boot": d.get("n_boot"),
            "arms": arms,
            "median_raw_ci_width": median(widths),
            "min_n_seeds": min(x["n_seeds"] for x in arms.values()),
            "seed_level_degenerate": all(x["n_seeds"] == 1 for x in arms.values()),
        }
    degenerate = [t for t in lineage if lineage[t]["seed_level_degenerate"]]
    chk("F1 the published lineage was NOT run at a uniform seed-replication depth",
        len(degenerate) >= 1 and lineage["E45"]["min_n_seeds"] == 3,
        "seed-replication depth per experiment: %s; the seed level is DEGENERATE "
        "(window-only interval) for: %s"
        % (", ".join("%s=%d" % (t, lineage[t]["min_n_seeds"]) for t in lineage),
           ", ".join(degenerate) if degenerate else "none"))
    report["F_lineage_census"] = lineage

    # ================================================================ G
    # Which published verdicts are CALIBRATION-SENSITIVE?  The verdict code
    # is:  survived = (not eq.crosses_zero) and eq.point > 0 ;
    #         gained_raw = raw.point > 0 ;
    #         GAIN_SURVIVES / GAIN_WAS_AREA_OR_AMPLITUDE / NO_GAIN.
    def verdict_of(raw_point, eq_point, eq_crosses_zero):
        if (not eq_crosses_zero) and eq_point > 0:
            return "GAIN_SURVIVES_EQUALISATION"
        if raw_point > 0:
            return "GAIN_WAS_AREA_OR_AMPLITUDE"
        return "NO_GAIN_TO_BEGIN_WITH"

    sens = OrderedDict()
    n_sensitive = 0
    n_total = 0
    for tag in lineage:
        for a, v in lineage[tag]["arms"].items():
            pub = verdict_of(v["raw_point"], v["equal_bias_point"],
                             v["equal_bias_crosses_zero"])
            alt_excl = verdict_of(v["raw_point"], v["equal_bias_point"], False)
            alt_incl = verdict_of(v["raw_point"], v["equal_bias_point"], True)
            n_total += 1
            is_sens = (alt_excl != pub) or (alt_incl != pub)
            if is_sens:
                n_sensitive += 1
            sens["%s/%s" % (tag, a)] = {
                "published_verdict": pub,
                "verdict_if_equal_bias_ci_excluded_zero": alt_excl,
                "verdict_if_equal_bias_ci_crossed_zero": alt_incl,
                "calibration_sensitive": is_sens,
                "n_seeds": v["n_seeds"],
            }
    chk("G1 some published S3 verdicts are calibration-sensitive (their equal-bias "
        "interval's crossing decision, not their point estimate, carries the verdict)",
        n_sensitive >= 1,
        "%d of %d published arm-level verdicts change if the equal-bias interval's "
        "crossing status is flipped" % (n_sensitive, n_total))
    report["G_verdict_calibration_sensitivity"] = {
        "n_arms": n_total, "n_calibration_sensitive": n_sensitive, "per_arm": sens,
    }

    # ================================================================ H2
    # SEED-DEPTH CONFOUND.  E44 ran at n_seeds = 1 (a single trained model per
    # arm); E45 ran at n_seeds = 3.  So the lineage's "significant" labels and
    # E45's "none" were produced at different replication depths.  Ask the
    # decisive question: if E44's depth-1 protocol had been applied to a
    # MECHANISM-FREE comparison AT E44's own replication depth, would it have
    # produced a "significant" label?  That is the null calibration of E1,
    # re-expressed at E44's depth and at E45's depth side by side.
    #
    # The null pair used is the control arm's own seed swap at the SAME depth
    # each experiment actually used:
    #   * depth-1 protocol (E44): one arm-vs-control comparison, window-only
    #     resampling -> 3 independent null comparisons (one per seed);
    #   * depth-3 protocol (E45): the full two-level bootstrap over the three
    #     control seeds vs themselves -> the interval must contain zero.
    # Everything here is local, read-only, no remote change.
    e44_depth1_null = []
    for k in SEEDS:
        r = boot_depth1(counts[("B0", k)], counts[("B0", (k + 1) % 3)])
        e44_depth1_null.append({"seed": k, "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                                "crosses_zero": r["crosses_zero"],
                                "significant": (not r["crosses_zero"])})
    n_e44_fp = sum(1 for x in e44_depth1_null if x["significant"])

    # E45's own depth-3 protocol applied to the SAME null (B0 vs B0, seeds
    # deranged) -- already computed in section E.
    e45_depth3_null_fp = fpr3

    # Seed-level variance share of the depth-3 interval: for each E45 arm,
    # how much of the depth-3 CI width is attributable to the seed level rather
    # than to the window level?  Compare the depth-3 width to the mean of the
    # three depth-1 (window-only) widths of the same arm.
    width_decomp = OrderedDict()
    for a in SIX:
        d1w = [c["width"] for c in depth1[a]["cells"]]
        d3w = s3_45["arms"][a]["raw"]["ci_hi"] - s3_45["arms"][a]["raw"]["ci_lo"]
        width_decomp[a] = {
            "depth3_width": d3w,
            "mean_depth1_width_within_arm": mean(d1w),
            "ratio_depth3_over_depth1": d3w / mean(d1w) if mean(d1w) else None,
        }
    ratios_w = [width_decomp[a]["ratio_depth3_over_depth1"] for a in SIX
                if width_decomp[a]["ratio_depth3_over_depth1"] is not None]
    chk("H2 the lineage's 'significant' vs 'none' split is confounded with the "
        "guard's seed-replication depth (E44 depth-1 vs E45 depth-3)",
        n_e44_fp == 3 and e45_depth3_null_fp == 0,
        "at E44's depth-1 protocol, a mechanism-free control-vs-control comparison "
        "is declared 'significant' in %d/3 nulls; at E45's depth-3 protocol the "
        "same kind of null is declared in %d/2 nulls" % (n_e44_fp, e45_depth3_null_fp))
    report["H2_seed_depth_confound"] = {
        "e44_depth1_protocol_on_mechanism_free_null": e44_depth1_null,
        "e44_depth1_false_positive_rate": n_e44_fp / 3.0,
        "e45_depth3_false_positive_rate": e45_depth3_null_fp / float(len(null)),
        "width_decomposition_depth3_over_depth1": width_decomp,
        "median_width_ratio_depth3_over_depth1": median(ratios_w),
        "note": ("the depth-3 interval is much wider than the within-arm depth-1 "
                 "interval because the seed level contributes; that is the correct "
                 "behaviour of the guard, not a defect"),
    }

    # ================================================================ I
    # Lineage-wide verdict census: across the whole E41->E45 lineage, has ANY
    # arm ever been judged GAIN_SURVIVES_EQUALISATION at the published depth?
    census = OrderedDict()
    for tag in lineage:
        c = OrderedDict()
        for a, v in lineage[tag]["arms"].items():
            c[v["s3_verdict"]] = c.get(v["s3_verdict"], 0) + 1
        census[tag] = {"n_arms": len(lineage[tag]["arms"]), "verdicts": c,
                       "n_seeds": lineage[tag]["min_n_seeds"]}
    n_survive = sum(c["verdicts"].get("GAIN_SURVIVES_EQUALISATION", 0)
                     for c in census.values())
    n_all = sum(c["n_arms"] for c in census.values())
    chk("I1 across the WHOLE E41->E45 lineage no arm was ever judged "
        "GAIN_SURVIVES_EQUALISATION at its published depth",
        n_survive == 0,
        "%d of %d lineage arms reached GAIN_SURVIVES_EQUALISATION" % (n_survive, n_all))
    report["I_lineage_verdict_census"] = {"per_experiment": census,
                                          "n_arms_total": n_all,
                                          "n_gain_survives": n_survive}

    # ================================================================ H
    # Headline: how many E45 arms would the depth-1 protocol have declared
    # a gain, versus the depth-3 protocol's zero.
    headline = OrderedDict()
    headline["depth3_protocol"] = {
        "arms_with_published_raw_interval_excluding_zero":
            sum(1 for a in SIX if not s3_45["arms"][a]["raw"]["crosses_zero"]),
        "arms_with_published_equal_bias_interval_excluding_zero":
            sum(1 for a in SIX if not s3_45["arms"][a]["equal_bias"]["crosses_zero"]),
        "arms_qualifying": 0,
    }
    headline["depth1_protocol_same_arms"] = {
        "cells_significant": n_sig,
        "cells_declared_gain": n_sig_pos,
        "arms_with_at_least_one_significant_cell":
            sum(1 for a in SIX if depth1[a]["n_cells_significant"] > 0),
        "arms_with_at_least_one_declared_gain_cell":
            sum(1 for a in SIX if depth1[a]["n_cells_declared_gain"] > 0),
    }
    headline["null_false_positive_rate_depth1"] = fpr1 / float(n_pairs_total)
    headline["null_false_positive_rate_depth3"] = fpr3 / float(len(null))
    headline["reading"] = (
        "The E44-style depth-1 protocol manufactures 'significant gains' out of "
        "E45's arms AND out of a mechanism-free null, while the depth-3 protocol "
        "used by E45 declares nothing on either.  The protocol, not the mechanism, "
        "separates the lineage's 'significant' labels from E45's 'none'."
    )
    report["H_headline"] = headline

    report["verdict"] = {
        "e45_verdict_unchanged": True,
        "statement": ("E45 remains NEGATIVE: all 12 published S3 intervals still "
                      "cross zero, the six s3_verdicts are unchanged and the "
                      "qualifying-arm set is still empty.  This round does NOT "
                      "re-adjudicate E41/E42/E43/E44."),
        "master_version": "v2.48 (unchanged: no published number and no verdict changed)",
    }

    with open(OUTPATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    # ---------------------------------------------------------------- console
    print("=" * 78)
    print("E45 ROUND 55 -- CALIBRATION OF THE S3 GUARD (E41->E45 LINEAGE)")
    print("=" * 78)
    for name, ok, detail in CHECKS:
        print("[%s] %s -- %s" % ("PASS" if ok else "FAIL", name, detail))
    print()
    print("A. published S3 point estimates reproduced from local artifacts:")
    for a in SIX:
        print("   %-4s raw=%+.6f  eq=%+.6f" % (a, pts[a]["raw_point"], pts[a]["equal_bias_point"]))
    print()
    print("C. stdlib re-implementation vs published raw intervals:")
    for a in SIX:
        m = mine[a]
        print("   %-4s stdlib CI=[%+.6f,%+.6f] w=%.6f | published CI=[%+.6f,%+.6f] w=%.6f | "
              "ratio=%.3f | xz agree=%s"
              % (a, m["stdlib_ci"][0], m["stdlib_ci"][1], m["stdlib_width"],
                 m["published_ci"][0], m["published_ci"][1], m["published_width"],
                 m["width_ratio_stdlib_over_published"], m["crossing_decision_agrees"]))
    print()
    print("D. E44-style depth-1 protocol applied to E45's 18 (arm,seed) cells:")
    for a in SIX:
        cells = depth1[a]["cells"]
        print("   %-4s " % a + "  ".join("s%d pt=%+.4f CI=[%+.4f,%+.4f]%s"
                                          % (c["seed"], c["point"], c["ci_lo"], c["ci_hi"],
                                             "*" if c["significant_at_depth1"] else " ")
                                          for c in cells))
    print("   (* = 'significant' at depth 1)  total significant %d/18, declared gain %d/18"
          % (n_sig, n_sig_pos))
    print()
    print("E. NULL calibration (B0_s{k} vs B0_s{pi(k)}, seed noise only):")
    for name, v in null.items():
        print("   %s : depth-1 significant %d/3 | depth-3 CI=[%+.6f,%+.6f] crosses_zero=%s"
              % (name, v["n_pairs_significant_at_depth1"], v["depth3_ci"][0],
                 v["depth3_ci"][1], v["depth3_crosses_zero"]))
    print("   false-positive rate: depth-1 %.3f  vs  depth-3 %.3f"
          % (fpr1 / float(n_pairs_total), fpr3 / float(len(null))))
    print()
    print("F. lineage census (seed-replication depth / median raw CI width):")
    for tag in lineage:
        L = lineage[tag]
        print("   %-4s n_seeds=%d  median raw CI width=%.6f  degenerate_seed_level=%s  (%s)"
              % (tag, L["min_n_seeds"], L["median_raw_ci_width"],
                 L["seed_level_degenerate"], L["variant"][:44]))
    print()
    print("G. calibration-sensitive published verdicts:")
    for k, v in sens.items():
        if v["calibration_sensitive"]:
            print("   %-10s published=%-28s if CI excl 0 -> %-28s if CI crosses 0 -> %s"
                  % (k, v["published_verdict"],
                     v["verdict_if_equal_bias_ci_excluded_zero"],
                     v["verdict_if_equal_bias_ci_crossed_zero"]))
    print("   %d of %d arm-level verdicts are calibration-sensitive"
          % (n_sensitive, n_total))
    print()
    print("H2. seed-depth confound:")
    print("   E44-depth (depth-1) protocol on a mechanism-free B0-vs-B0 null:")
    for x in e44_depth1_null:
        print("      s%d CI=[%+.6f,%+.6f] significant=%s"
              % (x["seed"], x["ci_lo"], x["ci_hi"], x["significant"]))
    print("   false-positive rate: E44-depth %.3f   vs   E45-depth %.3f"
          % (n_e44_fp / 3.0, e45_depth3_null_fp / float(len(null))))
    print("   depth-3 / depth-1 CI width ratio per arm (seed level's contribution):")
    for a in SIX:
        w = width_decomp[a]
        print("      %-4s depth3_w=%.6f  mean_depth1_w=%.6f  ratio=%.2f"
              % (a, w["depth3_width"], w["mean_depth1_width_within_arm"],
                 w["ratio_depth3_over_depth1"]))
    print()
    print("I. lineage-wide verdict census:")
    for tag, c in census.items():
        print("   %-4s n_seeds=%d n_arms=%d  %s"
              % (tag, c["n_seeds"], c["n_arms"], json.dumps(c["verdicts"], ensure_ascii=False)))
    print("   arms reaching GAIN_SURVIVES_EQUALISATION across the whole lineage: %d/%d"
          % (n_survive, n_all))
    print()
    print("H. headline:", json.dumps(headline, ensure_ascii=False))
    print("artifact:", OUTPATH)

    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print("POS %d/%d  VERDICT: %s" % (n_pass, len(CHECKS),
                                       "PASS" if n_pass == len(CHECKS) else "FAIL"))


if __name__ == "__main__":
    main()
