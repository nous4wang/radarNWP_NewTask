#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 54 -- metric-family axis (CSI / ETS / HSS / FSS).

WHY THIS ROUND EXISTS
---------------------
Every one of the 53 previous rounds judged E45 on **CSI / BIAS / POD / FAR only**.
E45's own failure mode is precisely that CSI+BIAS cannot distinguish:
    "real skill"  vs  "over-forecast (area/amplitude) that CSI rewards"
-- which is exactly what the S3 guard concluded (GAIN_WAS_AREA_OR_AMPLITUDE).

The stage-2 artifacts, however, carry **four** contingency/field skill scores:
    CSI  = h/(h+m+f)                      (no random-guessing correction)
    ETS  = (h-E)/(h+m+f-E),  E=rand        (climatological null, symmetric)
    HSS  = 2(h*CN - m*f) / (...)          (hit-weighted, FA-penalising)
    FSS  = fractions skill score           (displacement-tolerant neighbourhood score)
HSS/ETS/FSS are **hit-weighted**: adding false alarms without hits actively
destroys them. So they are an independent instrument for the area question.
This round judges E45's six arms on all four families, for the first time.

RULES HONOURED (from the closed case file):
  * median + pass rate only, single-point scores never stand alone
  * paired convention and difference convention reported side by side
  * zero-model floors computed PER METRIC FAMILY (round-52 lesson: a CSI floor
    applied to POD produced a false positive; families are not interchangeable)
  * the S3 guard stays pinned at thr=2.0 mm, replication unit stays the seed
  * post-hoc diagnostic -> does NOT constitute a new criterion, verdict unchanged
  * pure stdlib (no numpy on this host)
"""

import json
import os
import math
import random
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVDIR = os.path.join(HERE, "_evals", "e45", "runs", "e45", "evals")
OUTPATH = os.path.join(HERE, "_evals", "e45", "round54_metric_family.json")

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SIX = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
THRS = ["0.1", "0.5", "2.0", "4.0"]
PRIMARY = "2.0"
FAMILIES = ["csi", "ets", "hss", "fss"]
FLOOR = 0.0924                      # persistence floor (seed-0 convention)
ORACLE = 0.1653
N_BOOT = 10000
RNG_SEED = 20261114

CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


# ---------------------------------------------------------------- statistics
def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def mean(xs):
    return sum(xs) / float(len(xs)) if xs else None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sxy = sxx = syy = 0.0
    for a, b in zip(xs, ys):
        da, db = a - mx, b - my
        sxy += da * db
        sxx += da * da
        syy += db * db
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def percentile(xs, q):
    """Linear-interpolation percentile, q in [0,100]."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ---------------------------------------------------------------- metrics
def m_csi(h, m, f, cn):
    d = h + m + f
    return h / d if d > 0 else None


def m_pod(h, m, f, cn):
    d = h + m
    return h / d if d > 0 else None


def m_far(h, m, f, cn):
    d = h + f
    return f / d if d > 0 else None


def m_bias(h, m, f, cn):
    d = h + m
    return (h + f) / d if d > 0 else None


def m_ets(h, m, f, cn):
    n = h + m + f + cn
    d1 = h + m + f
    if n <= 0 or d1 <= 0:
        return None
    rand = (h + f) * (h + m) / n
    den = d1 - rand
    return (h - rand) / den if den != 0 else None


def m_hss(h, m, f, cn):
    den = (h + m) * (h + f) + (m + cn) * (f + cn)
    if den <= 0:
        return None
    return 2.0 * (h * cn - m * f) / den


FUNCS = {"csi": m_csi, "pod": m_pod, "far": m_far, "bias": m_bias,
         "ets": m_ets, "hss": m_hss}


# ---------------------------------------------------------------- loading
def load_eval(arm, seed):
    p = os.path.join(EVDIR, "%s_s%d_ep00.eval.json" % (arm, seed))
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_windows(arm, seed):
    p = os.path.join(EVDIR, "%s_s%d_ep00.windows.json" % (arm, seed))
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    report = OrderedDict()
    report["meta"] = {
        "round": 54, "axis": "metric-family (CSI/ETS/HSS/FSS)",
        "primary_threshold_mm": PRIMARY, "n_boot": N_BOOT, "rng_seed": RNG_SEED,
        "floor_persistence": FLOOR, "oracle": ORACLE,
        "rule": "post-hoc diagnostic, NOT a new criterion; replication unit = seed",
    }

    evals = {}
    wins = {}
    for a in ARMS:
        for s in SEEDS:
            evals[(a, s)] = load_eval(a, s)
            wins[(a, s)] = load_windows(a, s)

    # ---- A. window key identity across the 21 cells -------------------
    ref = wins[("B0", 0)]["window_keys"]
    bad = 0
    for a in ARMS:
        for s in SEEDS:
            if wins[(a, s)]["window_keys"] != ref:
                bad += 1
    chk("A1 window_keys identical across 21 cells", bad == 0,
        "mismatched cells=%d, n_windows=%d" % (bad, len(ref)))

    # ---- B. recompute published cell metrics from the window tables ----
    recompute = {}
    maxdiff = {"csi": 0.0, "bias": 0.0, "pod": 0.0, "far": 0.0,
               "ets": 0.0, "hss": 0.0}
    ncmp = 0
    for a in ARMS:
        for s in SEEDS:
            agg = {}
            for t in THRS:
                h = m_ = f = cn = 0.0
                for w in wins[(a, s)]["windows"]:
                    c = w[t]
                    h += c["hits"]; m_ += c["misses"]
                    f += c["false_alarms"]; cn += c["correct_negatives"]
                agg[t] = (h, m_, f, cn)
            recompute[(a, s)] = agg
            for t in THRS:
                h, m_, f, cn = agg[t]
                pub = evals[(a, s)]["thresholds"][t]
                for fam in ("csi", "bias", "pod", "far", "ets", "hss"):
                    v = FUNCS[fam](h, m_, f, cn)
                    if v is None:
                        continue
                    d = abs(v - pub[fam])
                    maxdiff[fam] = max(maxdiff[fam], d)
                    ncmp += 1
    chk("B1 published CSI/BIAS/POD/FAR reproduced from window tables (exact)",
        max(maxdiff["csi"], maxdiff["bias"], maxdiff["pod"], maxdiff["far"]) == 0.0,
        "max|diff| csi=%.3e bias=%.3e pod=%.3e far=%.3e over %d comparisons"
        % (maxdiff["csi"], maxdiff["bias"], maxdiff["pod"], maxdiff["far"], ncmp))
    report["B_recompute_vs_published"] = {
        "max_abs_diff": maxdiff, "n_comparisons": ncmp,
        "note": ("ets/hss are INFO: standard textbook formulas are used here; "
                 "whether the repo uses the same variant is reported, not asserted"),
    }

    # ---- C. headline table arithmetic (round-47 style) -----------------
    hl = {}
    for s in SEEDS:
        with open(os.path.join(HERE, "_evals", "e45", "scores_summary_s%d.json" % s),
                  "r", encoding="utf-8") as fh:
            hl[s] = json.load(fh)
    with open(os.path.join(HERE, "_evals", "e45", "scores_summary.json"),
              "r", encoding="utf-8") as fh:
        pub_sum = json.load(fh)
    hd = 0.0
    hn = 0
    for a in ARMS:
        cs = [hl[s]["%s_s%d_ep00" % (a, s)]["csi_at_2.0"] for s in SEEDS]
        bs = [hl[s]["%s_s%d_ep00" % (a, s)]["bias_at_2.0"] for s in SEEDS]
        mc, mb = median(cs), median(bs)
        hd = max(hd, abs(mc - pub_sum[a]["median_csi_at_2.0"]))
        hd = max(hd, abs(mb - pub_sum[a]["median_bias_at_2.0"]))
        hn += 2
        for s in SEEDS:
            hd = max(hd, abs(cs[s] - pub_sum[a]["csi_at_2.0"]["s%d" % s]))
            hd = max(hd, abs(bs[s] - pub_sum[a]["bias_at_2.0"]["s%d" % s]))
            hn += 2
    chk("C1 headline table recomputed from per-seed summaries", hd < 1e-9,
        "max|diff|=%.3e over %d items" % (hd, hn))

    # ---- D. S3 guard unchanged ------------------------------------------
    with open(os.path.join(HERE, "_evals", "e45", "s3_equalbias.json"),
              "r", encoding="utf-8") as fh:
        s3 = json.load(fh)
    n_cross = 0
    n_tot = 0
    verdicts = {}
    for a in SIX:
        for k in ("raw", "equal_bias"):
            n_tot += 1
            if s3["arms"][a][k]["crosses_zero"]:
                n_cross += 1
        verdicts[a] = s3["arms"][a]["s3_verdict"]
    chk("D1 all 12 S3 intervals still cross zero", n_cross == n_tot,
        "%d/%d cross zero; verdicts=%s" % (n_cross, n_tot,
                                           sorted(set(verdicts.values()))))

    # ---- E. metric-family cell values ----------------------------------
    fam_vals = {}     # fam -> thr -> arm -> seed -> value
    for fam in FAMILIES:
        for t in THRS:
            for a in ARMS:
                for s in SEEDS:
                    fam_vals.setdefault(fam, {}).setdefault(t, {}).setdefault(a, {})[s] = \
                        evals[(a, s)]["thresholds"][t].get(fam)
    missing = []
    for fam in FAMILIES:
        for t in THRS:
            for a in ARMS:
                for s in SEEDS:
                    if fam_vals[fam][t][a][s] is None:
                        missing.append((fam, t, a, s))
    report["E_metric_availability"] = {
        "missing_cells": len(missing),
        "note": "fss published per cell for all thresholds" if not missing
                else "missing sample: %s" % (missing[:4],),
    }

    # ---- F. per-family zero-model floors (control seed swaps) ----------
    floors = {}
    for fam in FAMILIES:
        for t in THRS:
            d = []
            for i in range(3):
                for j in range(i + 1, 3):
                    d.append(abs(fam_vals[fam][t]["B0"][i] - fam_vals[fam][t]["B0"][j]))
            floors[(fam, t)] = max(d)
    report["F_family_zero_model_floors"] = {
        "%s@%s" % (f, t): floors[(f, t)] for (f, t) in floors
        if t == PRIMARY}

    # ---- G. per-arm, per-family gains at the primary threshold ---------
    def gains_for(fam, t):
        out = OrderedDict()
        for a in SIX:
            g = [fam_vals[fam][t][a][s] - fam_vals[fam][t]["B0"][s] for s in SEEDS]
            arm_med = median([fam_vals[fam][t][a][s] for s in SEEDS])
            ctl_med = median([fam_vals[fam][t]["B0"][s] for s in SEEDS])
            fl = floors[(fam, t)]
            same_sign = sum(1 for x in g if x > 0) if median(g) > 0 else \
                sum(1 for x in g if x < 0)
            out[a] = {
                "per_seed_gain": g,
                "paired_median_gain": median(g),
                "difference_caliber_gain": arm_med - ctl_med,
                "family_floor": fl,
                "n_seeds_exceed_floor": sum(1 for x in g if abs(x) > fl),
                "pass_rate_over_floor": "%d/3" % sum(1 for x in g if abs(x) > fl),
                "n_seeds_same_sign": same_sign,
                "arm_median": arm_med, "control_median": ctl_med,
            }
        return out

    fam_report = OrderedDict()
    for fam in FAMILIES:
        fam_report[fam] = {"primary": gains_for(fam, PRIMARY),
                            "all_thresholds": {t: gains_for(fam, t) for t in THRS}}
    report["G_family_gains"] = fam_report

    # ---- H. winner probability per family (seed bootstrap) -------------
    winners = OrderedDict()
    for fam in FAMILIES:
        rng = random.Random(RNG_SEED)
        cnt = {a: 0 for a in SIX}
        for _ in range(N_BOOT):
            pick = [rng.choice(SEEDS) for _ in range(3)]
            best, bestv = None, None
            for a in SIX:
                vals = [fam_vals[fam][PRIMARY][a][s] for s in pick]
                ctl = [fam_vals[fam][PRIMARY]["B0"][s] for s in pick]
                v = median(vals) - median(ctl)
                if bestv is None or v > bestv:
                    best, bestv = a, v
            cnt[best] += 1
        winners[fam] = {a: cnt[a] / float(N_BOOT) for a in SIX}
    report["H_winner_probability_per_family"] = winners

    # ---- I. cross-family gain correlation over 18 (arm,seed) cells ------
    g_all = {fam: {} for fam in FAMILIES}
    for fam in FAMILIES:
        for a in SIX:
            for s in SEEDS:
                g_all[fam][(a, s)] = fam_vals[fam][PRIMARY][a][s] - \
                    fam_vals[fam][PRIMARY]["B0"][s]
    cross = OrderedDict()
    keys = [(a, s) for a in SIX for s in SEEDS]
    for i, f1 in enumerate(FAMILIES):
        for f2 in FAMILIES[i + 1:]:
            xs = [g_all[f1][k] for k in keys]
            ys = [g_all[f2][k] for k in keys]
            cross["%s_vs_%s" % (f1, f2)] = {
                "pearson_r": pearson(xs, ys),
                "n_cells": len(keys),
                "n_sign_agree": sum(1 for x, y in zip(xs, ys) if (x >= 0) == (y >= 0)),
                "family_floor_1": floors[(f1, PRIMARY)],
                "family_floor_2": floors[(f2, PRIMARY)],
            }
    report["I_cross_family_gain_correlation"] = cross

    # ---- J. sign agreement per arm across families --------------------
    sign_tab = OrderedDict()
    for a in SIX:
        row = OrderedDict()
        for fam in FAMILIES:
            row[fam] = median(g_all[fam][(a, 0)], g_all[fam][(a, 1)],
                              g_all[fam][(a, 2)]) if False else \
                median([g_all[fam][(a, s)] for s in SEEDS])
        pos = sum(1 for fam in FAMILIES if row[fam] > 0)
        neg = sum(1 for fam in FAMILIES if row[fam] < 0)
        sign_tab[a] = {"paired_median_gain": row,
                       "n_families_positive": pos, "n_families_negative": neg,
                       "consistent_sign": pos == 0 or neg == 0}
    report["J_per_arm_sign_across_families"] = sign_tab

    # ---- K. window-level paired gains (secondary; CSI/ETS/HSS only) ----
    wl = OrderedDict()
    for fam in ("csi", "ets", "hss"):
        # per-window metric vectors
        vec = {}
        for a in ARMS:
            for s in SEEDS:
                vals = []
                for w in wins[(a, s)]["windows"]:
                    c = w[PRIMARY]
                    vals.append(FUNCS[fam](c["hits"], c["misses"],
                                           c["false_alarms"], c["correct_negatives"]))
                vec[(a, s)] = vals
        # control-pair floor at window level (round-46 convention)
        cpair = []
        for i in range(3):
            for j in range(i + 1, 3):
                d = [x - y for x, y in zip(vec[("B0", i)], vec[("B0", j)])
                     if x is not None and y is not None]
                cpair.append(median(d))
        wl_floor = max(abs(x) for x in cpair)
        arm_wl = OrderedDict()
        for a in SIX:
            per_seed = []
            for s in SEEDS:
                d = [x - y for x, y in zip(vec[(a, s)], vec[("B0", s)])
                     if x is not None and y is not None]
                per_seed.append(median(d))
            arm_wl[a] = {"per_seed_window_paired_median": per_seed,
                         "median": median(per_seed),
                         "n_seeds_exceed_floor": sum(1 for x in per_seed
                                                    if abs(x) > wl_floor),
                         "pass_rate_over_floor": "%d/3" % sum(
                             1 for x in per_seed if abs(x) > wl_floor)}
        wl[fam] = {"control_pair_window_medians": cpair, "window_floor": wl_floor,
                   "arms": arm_wl}
    report["K_window_level_paired_gains"] = wl

    # ---- L. key question: does the area effect survive the swap? ------
    #    A4/A3b/A4b were the CSI-gain arms with exploding BIAS.
    key = OrderedDict()
    for a in ["A3a", "A3b", "A4", "A4b"]:
        key[a] = OrderedDict()
        for fam in FAMILIES:
            g = fam_report[fam]["primary"][a]
            key[a][fam] = {
                "paired_median_gain": g["paired_median_gain"],
                "difference_caliber_gain": g["difference_caliber_gain"],
                "family_floor": g["family_floor"],
                "pass_rate_over_floor": g["pass_rate_over_floor"],
            }
    report["L_area_effect_swap_test"] = key

    # ---- M. area-equalised contingency-domain counterpart of S3 --------
    # S3 is FIELD-domain: one global scalar rescale of the forecast field so
    # that its bias matches the control's, then CSI is recomputed. Here we build
    # a CONTINGENCY-domain counterpart that uses only the published window
    # tables: rescale the arm's forecast-event AREA to the control's area
    # (keeping the arm's hit rate), and recompute every metric family.
    # If the arm's gain survives that, the gain is NOT an area artifact.
    def pooled(arm, seed, thr=PRIMARY):
        h = m_ = f = cn = 0.0
        for w in wins[(arm, seed)]["windows"]:
            c = w[thr]
            h += c["hits"]; m_ += c["misses"]
            f += c["false_alarms"]; cn += c["correct_negatives"]
        return h, m_, f, cn

    def fams(h, m_, f, cn):
        return {fm: FUNCS[fm](h, m_, f, cn)
                for fm in ("csi", "ets", "hss", "bias", "pod", "far")}

    def equalise_area(h, m_, f, cn, target_area):
        A = h + f
        if A <= 0 or target_area <= 0:
            return None
        sc = target_area / A
        h2, f2 = h * sc, f * sc
        T = h + m_
        m2 = T - h2
        cn2 = cn + (f - f2)
        return fams(h2, m2, f2, cn2)

    # truth invariance: the truth field is shared, so T = h+m must not move
    tdev = 0.0
    for s in SEEDS:
        for wi in range(len(ref)):
            ts = set()
            for a in ARMS:
                c = wins[(a, s)]["windows"][wi][PRIMARY]
                ts.add(round(c["hits"] + c["misses"], 6))
            tdev = max(tdev, max(ts) - min(ts))
    chk("M1 truth event count invariant across the 7 arms (per window)",
        tdev == 0.0,
        "max within-window spread of T=h+m over 3 seeds x 538 windows: %.3e" % tdev)

    pooled_eq = OrderedDict()
    for a in SIX:
        per_seed = OrderedDict()
        for s in SEEDS:
            h, m_, f, cn = pooled(a, s)
            h0, m0, f0, cn0 = pooled("B0", s)
            eq = equalise_area(h, m_, f, cn, h0 + f0)
            ctl = fams(h0, m0, f0, cn0)
            per_seed[s] = {fm: (eq[fm] - ctl[fm]) for fm in eq} if eq else None
        pooled_eq[a] = {
            "per_seed": per_seed,
            "mean_over_seeds": {fm: mean([per_seed[s][fm] for s in SEEDS])
                                for fm in ("csi", "ets", "hss", "bias", "pod", "far")},
        }
    report["M_area_equalised_pooled"] = pooled_eq

    # direct comparability with the published S3 equal-bias point estimates
    s3_cmp = OrderedDict()
    for a in SIX:
        mine = pooled_eq[a]["mean_over_seeds"]["csi"]
        theirs = s3["arms"][a]["equal_bias"]["point"]
        s3_cmp[a] = {"contingency_area_equalised_csi_delta": mine,
                     "s3_field_equal_bias_point": theirs,
                     "same_sign": (mine > 0) == (theirs > 0),
                     "abs_diff": abs(mine - theirs)}
    report["M2_vs_published_s3_equal_bias"] = s3_cmp
    agree = sum(1 for a in SIX if s3_cmp[a]["same_sign"])
    chk("M2 contingency-area-equalised CSI delta agrees in sign with S3 "
        "field-domain equal-bias point estimates", agree == len(SIX),
        "%d/%d arms same sign; median |diff| vs S3 = %.6f"
        % (agree, len(SIX), median([s3_cmp[a]["abs_diff"] for a in SIX])))

    # per-window area equalisation, with a per-family control-pair floor
    wl_eq = OrderedDict()
    for fam in ("csi", "ets", "hss"):
        per_window = OrderedDict()
        for a in SIX:
            per_seed = []
            for s in SEEDS:
                d = []
                for wi in range(len(ref)):
                    ca = wins[(a, s)]["windows"][wi][PRIMARY]
                    cc = wins[("B0", s)]["windows"][wi][PRIMARY]
                    eq = equalise_area(ca["hits"], ca["misses"], ca["false_alarms"],
                                       ca["correct_negatives"],
                                       cc["hits"] + cc["false_alarms"])
                    if eq is None:
                        continue
                    ctl = FUNCS[fam](cc["hits"], cc["misses"], cc["false_alarms"],
                                     cc["correct_negatives"])
                    if ctl is None:
                        continue
                    d.append(eq[fam] - ctl)
                per_seed.append(median(d))
            per_window[a] = per_seed
        cpair = []
        for i in range(3):
            for j in range(i + 1, 3):
                d = []
                for wi in range(len(ref)):
                    ci = wins[("B0", i)]["windows"][wi][PRIMARY]
                    cj = wins[("B0", j)]["windows"][wi][PRIMARY]
                    eq = equalise_area(ci["hits"], ci["misses"], ci["false_alarms"],
                                       ci["correct_negatives"],
                                       cj["hits"] + cj["false_alarms"])
                    if eq is None:
                        continue
                    ctl = FUNCS[fam](cj["hits"], cj["misses"], cj["false_alarms"],
                                     cj["correct_negatives"])
                    if ctl is None:
                        continue
                    d.append(eq[fam] - ctl)
                cpair.append(median(d))
        wl_floor = max(abs(x) for x in cpair)
        wl_eq[fam] = {
            "control_pair_window_medians": cpair, "window_floor": wl_floor,
            "arms": {a: {"per_seed": per_window[a],
                         "median": median(per_window[a]),
                         "n_seeds_exceed_floor": sum(1 for x in per_window[a]
                                                    if abs(x) > wl_floor),
                         "pass_rate_over_floor": "%d/3" % sum(
                             1 for x in per_window[a] if abs(x) > wl_floor)}
                      for a in SIX}}
    report["M3_window_level_area_equalised"] = wl_eq

    # ---- summary / verdict --------------------------------------------
    n_pos = 0
    n_neg = 0
    for a in SIX:
        for fam in FAMILIES:
            v = fam_report[fam]["primary"][a]["paired_median_gain"]
            if v > 0:
                n_pos += 1
            elif v < 0:
                n_neg += 1
    report["M_summary"] = {
        "arms": len(SIX), "families": len(FAMILIES),
        "n_arm_family_cells_positive": n_pos, "n_arm_family_cells_negative": n_neg,
        "s3_verdicts": verdicts,
        "verdict": ("E45 remains NEGATIVE: all 12 S3 intervals still cross zero; "
                    "the metric-family axis is post-hoc diagnostic and does not "
                    "change the verdict; no arm enters the 10-seed tier"),
        "master_version": "v2.48 (unchanged: no new numbers in MASTER, no new criterion)",
    }

    with open(OUTPATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    # ---- console output ------------------------------------------------
    print("=" * 78)
    print("E45 ROUND 54 -- METRIC-FAMILY AXIS (CSI/ETS/HSS/FSS)")
    print("=" * 78)
    for name, ok, detail in CHECKS:
        print("[%s] %s -- %s" % ("PASS" if ok else "FAIL", name, detail))
    print()
    print("family floors @2.0mm (control seed-swap |delta|):")
    for fam in FAMILIES:
        print("   %-4s floor=%.6f" % (fam, floors[(fam, PRIMARY)]))
    print()
    print("per-arm paired-median gain (thr=2.0) with per-family floor:")
    hdr = "arm    " + "".join("%-22s" % fam for fam in FAMILIES)
    print(hdr)
    for a in SIX:
        line = "%-7s" % a
        for fam in FAMILIES:
            g = fam_report[fam]["primary"][a]
            line += "%-22s" % ("%+.6f %s fl=%.4f" % (
                g["paired_median_gain"], g["pass_rate_over_floor"], g["family_floor"]))
        print(line)
    print()
    print("difference-caliber gain (thr=2.0):")
    for a in SIX:
        line = "%-7s" % a
        for fam in FAMILIES:
            g = fam_report[fam]["primary"][a]
            line += "%-22s" % ("%+.6f" % g["difference_caliber_gain"])
        print(line)
    print()
    print("winner probability per family (seed bootstrap, n=%d):" % N_BOOT)
    for fam in FAMILIES:
        print("   %-4s " % fam + "  ".join("%s=%.3f" % (a, winners[fam][a]) for a in SIX))
    print()
    print("cross-family gain correlation (18 cells):")
    for k, v in cross.items():
        print("   %-16s r=%+.4f sign_agree=%d/18" % (
            k, v["pearson_r"] if v["pearson_r"] is not None else float("nan"),
            v["n_sign_agree"]))
    print()
    print("window-level paired median gain (thr=2.0), per-family window floor:")
    for fam, v in wl.items():
        print("   [%s] floor=%.6f" % (fam, v["window_floor"]))
        for a in SIX:
            print("        %-5s med=%+.6f  %s" % (
                a, v["arms"][a]["median"], v["arms"][a]["pass_rate_over_floor"]))
    print()
    print("per-arm sign across the four families:")
    for a in SIX:
        r = sign_tab[a]
        print("   %-5s pos=%d neg=%d  %s" % (
            a, r["n_families_positive"], r["n_families_negative"],
            "  ".join("%s=%+.6f" % (f, r["paired_median_gain"][f]) for f in FAMILIES)))
    print()
    print("summary:", json.dumps(report["M_summary"], ensure_ascii=False))
    print("artifact:", OUTPATH)

    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    verdict = "PASS" if n_pass == len(CHECKS) else "FAIL"
    print("POS %d/%d  VERDICT: %s" % (n_pass, len(CHECKS), verdict))


if __name__ == "__main__":
    main()
