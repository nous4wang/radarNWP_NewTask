#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 60 -- decision-theoretic VALUE axis (cost-loss / CNL).

Why this axis: rounds 40-59 judged CSI/BIAS/POD/FAR (aggregation, window,
case, lead, severity, error-composition), metric families (CSI/ETS/HSS/FSS),
guard calibration, gain-pattern stability, aggregation operators, training
dynamics and the promotion rule.  NOT ONE of them asked the decision-maker's
question: "if I have to pay for a miss C_M and a false alarm C_F, which arm
should I deploy?"  That is a different axis and it can flip the *reading* of
an arm without touching the frozen S3 guardrail.

Model (Murphy 1977; Jolliffe & Stephenson 2008, ch.9):
    p        = base rate of the event at threshold t (truth is SHARED -> p is
               identical for all arms; verified below)
    cost(r)  = r * (1 - POD) * p + FAR * (1 - p)          [per forecast unit]
               with r = C_M / C_F the cost-loss ratio
    cost_clim(r) = p * (1 - p) * (r + 1)                  [base-rate forecast]
    CS(r)    = 1 - cost(r) / cost_clim(r)                 [cost-loss skill]
    DC(r)    = cost_arm(r) - cost_ctrl(r)                 [paired decision cost]

DC(r) is LINEAR in r:
    DC(r) = -r * p * u + (1 - p) * v,
    u = POD_arm - POD_ctrl,  v = FAR_arm - FAR_ctrl
so the break-even cost-loss ratio is
    r* = (1 - p) * v / (p * u)
and the sign of u says on which side of r* the arm is cheaper:
    u > 0 (arm hits more AND false-alarms more): arm cheaper for r > r*
    u < 0 (arm hits fewer AND false-alarms fewer): arm cheaper for r < r*
A negative r* means the two arms never trade over: the arm is more expensive
for every r > 0 (or cheaper for every r > 0).

Discipline (carried from rounds 44/46/48/52/54/57/59):
  * report MEDIAN + PASS RATE; a single number never decides anything
  * paired median (median of the 3 per-seed differences) AND difference
    median (difference of the two medians) are always reported side by side
  * the zero model is the control arm with its seeds swapped (pure seed
    noise, no mechanism) -- every "beats the control" statement is checked
    against that floor
  * the case-level bootstrap (8 storm cases) is reported as INTERNAL
    consistency only; the replicate unit for an arm effect stays the seed
    (n = 3), so no case-level CI is ever called cross-seed evidence
  * post-hoc diagnostic -> does NOT create a new criterion and does NOT
    change any verdict; the S3 guardrail (thr = 2.0 mm) is untouched

Outputs: tools/_evals/e45/round60_decision_value.json
Read-only: touches nothing outside this repo's tools/_evals/e45/.
"""
import glob
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO, "tools", "_evals", "e45", "runs", "e45", "evals")
OUT = os.path.join(REPO, "tools", "_evals", "e45", "round60_decision_value.json")

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
TREAT = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
CTRL = "B0"
FLOOR_CSI = 0.0924
ORACLE_CSI = 0.1653
THR_MAIN = "2.0"
THR_EXTRA = ["0.5", "4.0"]
R_GRID = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
N_BOOT = 10000
RNG_SEED = 20261114

POS = []
NEG = []


def check(cond, msg):
    if cond:
        POS.append(msg)
    else:
        NEG.append(msg)
    return bool(cond)


def arm_seed_key(arm, seed):
    return "%s_s%d_ep00" % (arm, seed)


def case_of(window_key):
    parts = window_key.split("_")
    return "_".join(parts[:3])          # 'CP_01_08121100'


def load_cells():
    """cells[(arm, seed)] = {'counts': {thr: (h,m,f,cn)}, 'cases': [...]}"""
    cells = {}
    for arm in ARMS:
        for seed in SEEDS:
            base = os.path.join(EVAL_DIR, arm_seed_key(arm, seed))
            wpath = base + ".windows.json"
            epath = base + ".eval.json"
            if not (os.path.exists(wpath) and os.path.exists(epath)):
                raise SystemExit("missing artifacts for %s seed %d" % (arm, seed))
            wj = json.load(io.open(wpath, "r", encoding="utf-8"))
            ej = json.load(io.open(epath, "r", encoding="utf-8"))
            thresholds = list(wj["thresholds"])
            counts = {}
            per_case = {t: {} for t in thresholds}
            for t in thresholds:
                h = m = f = cn = 0.0
                for wi, wk in enumerate(wj["window_keys"]):
                    c = wj["windows"][wi][t]
                    h += c["hits"]; m += c["misses"]
                    f += c["false_alarms"]; cn += c["correct_negatives"]
                    cs = case_of(wk)
                    acc = per_case[t].setdefault(cs, [0.0, 0.0, 0.0, 0.0])
                    acc[0] += c["hits"]; acc[1] += c["misses"]
                    acc[2] += c["false_alarms"]; acc[3] += c["correct_negatives"]
                counts[t] = (h, m, f, cn)
            cases = sorted(per_case[THR_MAIN].keys())
            cells[(arm, seed)] = {"counts": counts, "per_case": per_case,
                                  "cases": cases, "eval": ej,
                                  "n_windows": wj["n_windows"],
                                  "window_keys": list(wj["window_keys"])}
    return cells


def pod_far(c):
    h, m, f, cn = c
    ev = h + m
    ne = f + cn
    pod = h / ev if ev > 0 else float("nan")
    far = f / ne if ne > 0 else float("nan")          # conditional POFD
    far_std = f / (h + f) if (h + f) > 0 else float("nan")   # published FAR
    csi = h / (h + m + f) if (h + m + f) > 0 else float("nan")
    bias = (h + f) / ev if ev > 0 else float("nan")
    p = ev / (h + m + f + cn) if (h + m + f + cn) > 0 else float("nan")
    return pod, far, csi, bias, p, far_std


def cost_of(pod, far, p, r):
    return r * (1.0 - pod) * p + far * (1.0 - p)


def clim_cost(p, r):
    return p * (1.0 - p) * (r + 1.0)


def main():
    cells = load_cells()

    # ------------------------------------------------------------------ 0
    # rebuild self-check: window tables -> published per-cell metrics
    worst = {"csi": 0.0, "bias": 0.0, "pod": 0.0, "far": 0.0}
    n_cmp = 0
    for (arm, seed), cell in cells.items():
        for t in cell["counts"]:
            pod, far_pofd, csi, bias, p, far_std = pod_far(cell["counts"][t])
            pub = cell["eval"]["thresholds"][t]
            # published FAR is the standard F/(H+F); the cost model uses the
            # conditional POFD = F/(F+CN).  Each is checked against its own
            # definition -- comparing the two would be a unit error.
            for key, val in (("csi", csi), ("bias", bias), ("pod", pod),
                             ("far", far_std)):
                d = abs(val - pub[key])
                worst[key] = max(worst[key], d)
                n_cmp += 1
    check(max(worst.values()) < 1e-12 and n_cmp == 336,
          "rebuild from 21 window tables reproduces published CSI/BIAS/POD/FAR "
          "exactly: %d comparisons, max|diff| = %.3e" % (n_cmp, max(worst.values())))

    # truth is shared -> base rate identical across arms/seeds
    ps = {}
    for (arm, seed), cell in cells.items():
        for t in cell["counts"]:
            ps.setdefault(t, []).append(pod_far(cell["counts"][t])[4])
    spread = {t: (max(v) - min(v)) for t, v in ps.items()}
    check(max(spread.values()) < 1e-12,
          "base rate p is identical across all 21 cells at every threshold "
          "(max spread %.3e)" % max(spread.values()))

    # case structure identical across cells (pairing is legal)
    ref_cases = cells[(CTRL, 0)]["cases"]
    same = all(cells[k]["cases"] == ref_cases for k in cells)
    check(same and len(ref_cases) == 8,
          "21 cells share the same 8 storm cases in the same order "
          "(n_cases = %d)" % len(ref_cases))
    keys_same = all(cells[k]["window_keys"] == cells[(CTRL, 0)]["window_keys"]
                    for k in cells)
    check(keys_same, "21 cells share identical window_keys in identical order")

    # ------------------------------------------------------------------ 1
    # headline re-verification (published table must be reproducible)
    summ = json.load(io.open(os.path.join(REPO, "tools", "_evals", "e45",
                                          "scores_summary.json"),
                              "r", encoding="utf-8"))
    head_maxdiff = 0.0
    for arm in ARMS:
        cs = [pod_far(cells[(arm, s)]["counts"][THR_MAIN])[2] for s in SEEDS]
        med = float(np.median(cs))
        head_maxdiff = max(head_maxdiff, abs(med - summ[arm]["median_csi_at_2.0"]))
        over = sum(1 for c in cs if c > FLOOR_CSI)
        check(over == int(summ[arm]["pass_rate_over_floor"].split("/")[0]),
              "%s floor pass-rate reproduces: %d/3" % (arm, over))
    check(head_maxdiff < 1e-12,
          "published headline median CSI@2.0 re-derived from window tables: "
          "max|diff| = %.3e over 7 arms" % head_maxdiff)

    # ------------------------------------------------------------------ 2
    # S3 guardrail re-check (verdict must not move)
    s3 = json.load(io.open(os.path.join(REPO, "tools", "_evals", "e45",
                                        "s3_equalbias.json"),
                           "r", encoding="utf-8"))

    def collect(obj, key, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    out.append(v)
                collect(v, key, out)
        elif isinstance(obj, list):
            for v in obj:
                collect(v, key, out)
        return out

    cz = collect(s3, "crosses_zero", [])
    vd = collect(s3, "s3_verdict", [])
    check(len(cz) >= 12 and all(bool(x) for x in cz),
          "S3 guardrail re-checked: %d published intervals, crosses_zero all true"
          % len(cz))

    # ------------------------------------------------------------------ 3
    # main decision-value analysis at the frozen threshold
    P = {k: pod_far(v["counts"][THR_MAIN]) for k, v in cells.items()}
    p_main = P[(CTRL, 0)][4]

    def dc_curve(arm, seed, r):
        pod, far = P[(arm, seed)][0], P[(arm, seed)][1]
        return cost_of(pod, far, p_main, r) - cost_of(P[(CTRL, seed)][0],
                                                      P[(CTRL, seed)][1],
                                                      p_main, r)

    def brk(arm, seed):
        pod_a, far_a = P[(arm, seed)][0], P[(arm, seed)][1]
        pod_c, far_c = P[(CTRL, seed)][0], P[(CTRL, seed)][1]
        u = pod_a - pod_c
        v = far_a - far_c
        if abs(u) < 1e-15:
            return float("nan"), u, v
        return (1.0 - p_main) * v / (p_main * u), u, v

    # zero model: control arm with seeds swapped (pure seed noise)
    ctrl_pairs = [(0, 1), (0, 2), (1, 2)]
    zero_dc = {r: [] for r in R_GRID}
    zero_brk = []
    for a, b in ctrl_pairs:
        for r in R_GRID:
            zero_dc[r].append(abs(cost_of(P[(CTRL, a)][0], P[(CTRL, a)][1], p_main, r)
                                  - cost_of(P[(CTRL, b)][0], P[(CTRL, b)][1],
                                            p_main, r)))
        rb, u, v = brk(CTRL, a) if False else (None, None, None)
        pod_a, far_a = P[(CTRL, a)][0], P[(CTRL, a)][1]
        pod_b, far_b = P[(CTRL, b)][0], P[(CTRL, b)][1]
        u = pod_a - pod_b
        v = far_a - far_b
        zero_brk.append((1.0 - p_main) * v / (p_main * u) if abs(u) > 1e-15
                        else float("nan"))
    zero_floor = {r: max(zero_dc[r]) for r in R_GRID}
    zero_brk_abs = [abs(x) for x in zero_brk if np.isfinite(x)]
    zero_brk_med = float(np.median(zero_brk_abs)) if zero_brk_abs else float("nan")

    per_arm = {}
    for arm in TREAT:
        rows = []
        for seed in SEEDS:
            pod_a, far_a = P[(arm, seed)][0], P[(arm, seed)][1]
            pod_c, far_c = P[(CTRL, seed)][0], P[(CTRL, seed)][1]
            u = pod_a - pod_c
            v = far_a - far_c
            rs, _, _ = brk(arm, seed)
            rows.append({
                "seed": seed,
                "pod_arm": pod_a, "pod_ctrl": pod_c, "d_pod": u,
                "far_arm": far_a, "far_ctrl": far_c, "d_far": v,
                "bias_arm": P[(arm, seed)][3], "bias_ctrl": P[(CTRL, seed)][3],
                "dc": {("r=%g" % r): dc_curve(arm, seed, r) for r in R_GRID},
                "break_even_r": rs,
                # the side on which the arm is cheaper depends on sign(u):
                #   u > 0 (more hits, more false alarms) -> cheaper for r > r*
                #   u < 0 (fewer hits, fewer false alarms) -> cheaper for r < r*
                "cheaper_side": ("above" if u > 0 else "below")
                                if np.isfinite(rs) else "no_crossover",
                "dc_at_break_even": (dc_curve(arm, seed, rs)
                                     if np.isfinite(rs) else float("nan")),
                # unambiguous classification read off the whole DC curve
                "cost_profile": (
                    "cheaper_at_all_r" if all(row_dc < 0 for row_dc in
                                              [dc_curve(arm, seed, r) for r in R_GRID])
                    else "more_expensive_at_all_r" if all(row_dc > 0 for row_dc in
                                                          [dc_curve(arm, seed, r) for r in R_GRID])
                    else "crossover"),
            })
        # paired median vs difference median, per r
        curves = {}
        for r in R_GRID:
            key = "r=%g" % r
            diffs = [row["dc"][key] for row in rows]
            paired = float(np.median(diffs))
            cheaper = sum(1 for d in diffs if d < 0)
            floor = zero_floor[r]
            curves[key] = {
                "paired_median_dc": paired,
                "per_seed_dc": diffs,
                "pass_rate_cheaper": "%d/3" % cheaper,
                "zero_model_floor_abs_dc": floor,
                "effect_over_floor": abs(paired) / floor if floor > 0 else float("nan"),
                "beats_zero_floor": bool(abs(paired) > floor),
                "cheaper_and_beyond_floor": bool(paired < 0 and abs(paired) > floor),
            }
        # difference-of-medians over the cost curve (absolute arm levels)
        arm_cost_med = {}
        ctrl_cost_med = {}
        for r in R_GRID:
            arm_cost_med["r=%g" % r] = float(np.median(
                [cost_of(P[(arm, s)][0], P[(arm, s)][1], p_main, r) for s in SEEDS]))
            ctrl_cost_med["r=%g" % r] = float(np.median(
                [cost_of(P[(CTRL, s)][0], P[(CTRL, s)][1], p_main, r) for s in SEEDS]))
        diff_of_median = {k: arm_cost_med[k] - ctrl_cost_med[k] for k in arm_cost_med}

        rs_vals = [row["break_even_r"] for row in rows]
        finite = [x for x in rs_vals if np.isfinite(x)]
        u_med = float(np.median([row["d_pod"] for row in rows]))
        v_med = float(np.median([row["d_far"] for row in rows]))
        u_signs = [row["d_pod"] > 0 for row in rows]
        per_arm[arm] = {
            "per_seed": rows,
            "decision_cost_curves": curves,
            "difference_of_medians_dc": diff_of_median,
            "arm_cost_median": arm_cost_med,
            "ctrl_cost_median": ctrl_cost_med,
            "break_even_r_per_seed": rs_vals,
            "break_even_r_median": float(np.median(finite)) if finite else float("nan"),
            "cost_profile_per_seed": [row["cost_profile"] for row in rows],
            "cost_profile_consistency": "%d/3" % max(
                [sum(1 for row in rows if row["cost_profile"] == prof)
                 for prof in set(row["cost_profile"] for row in rows)]),
            "break_even_r_sign_consistency": "%d/3" % sum(
                1 for x in finite if (x > 0) == (u_med > 0)),
            "d_pod_sign_consistency": "%d/3" % sum(1 for s in u_signs if s),
            "cheaper_side_per_seed": [row["cheaper_side"] for row in rows],
            "median_d_pod": u_med,
            "median_d_far": v_med,
            "median_r_abs_vs_zero_model": (abs(float(np.median(finite))) / zero_brk_med
                                           if (finite and zero_brk_med > 0) else float("nan")),
        }

    # -------------------------------------------------------------- 3b
    # case-level bootstrap (8 storm cases) -- INTERNAL consistency only
    rng = np.random.default_rng(RNG_SEED)
    case_idx = rng.integers(0, len(ref_cases), size=(N_BOOT, len(ref_cases)))
    boot_ci = {}
    for arm in TREAT:
        boot_ci[arm] = {}
        for r in (0.5, 1.0, 2.0, 4.0):
            key = "r=%g" % r
            per_seed_ci = []
            for seed in SEEDS:
                pc = cells[(arm, seed)]["per_case"][THR_MAIN]
                pcc = cells[(CTRL, seed)]["per_case"][THR_MAIN]
                def pooled(sel, table):
                    h = m = f = cn = 0.0
                    for i in sel:
                        a = table[ref_cases[i]]
                        h += a[0]; m += a[1]; f += a[2]; cn += a[3]
                    return h, m, f, cn
                vals = np.empty(N_BOOT)
                for bi in range(N_BOOT):
                    sel = case_idx[bi]
                    ha, ma, fa, cna = pooled(sel, pc)
                    hc, mc, fc, cnc = pooled(sel, pcc)
                    pa = pod_far((ha, ma, fa, cna))
                    pcv = pod_far((hc, mc, fc, cnc))
                    p_here = pa[4]
                    ca = cost_of(pa[0], pa[1], p_here, r)
                    cc = cost_of(pcv[0], pcv[1], p_here, r)
                    vals[bi] = ca - cc
                lo, hi = np.percentile(vals, [2.5, 97.5])
                per_seed_ci.append({"seed": seed, "lo": float(lo), "hi": float(hi),
                                    "excludes_zero": bool(lo > 0 or hi < 0)})
            boot_ci[arm][key] = per_seed_ci

    # ------------------------------------------------------------------ 4
    # control arm's own cost-loss skill (absolute decision value)
    ctrl_skill = {}
    for r in R_GRID:
        cs = []
        for seed in SEEDS:
            pod, far = P[(CTRL, seed)][0], P[(CTRL, seed)][1]
            cs.append(1.0 - cost_of(pod, far, p_main, r) / clim_cost(p_main, r))
        ctrl_skill["r=%g" % r] = {"median": float(np.median(cs)), "per_seed": cs}

    # threshold robustness of the break-even ratio (descriptive only)
    thr_robust = {}
    for t in THR_EXTRA + [THR_MAIN]:
        row = {}
        pt = P and pod_far(cells[(CTRL, 0)]["counts"][t])[4]
        for arm in TREAT:
            rs = []
            for seed in SEEDS:
                c_a = cells[(arm, seed)]["counts"][t]
                c_c = cells[(CTRL, seed)]["counts"][t]
                pa = pod_far(c_a); pcc = pod_far(c_c)
                u = pa[0] - pcc[0]; v = pa[1] - pcc[1]
                rs.append((1.0 - pt) * v / (pt * u) if abs(u) > 1e-15 else float("nan"))
            fin = [x for x in rs if np.isfinite(x)]
            row[arm] = {"break_even_r_median": float(np.median(fin)) if fin else float("nan"),
                        "per_seed": rs}
        thr_robust[t] = row

    # ------------------------------------------------------------------ 5
    out = {
        "round": 60,
        "axis": "decision-theoretic value (cost-loss / CNL / break-even "
                "cost-loss ratio)",
        "guardrail": {"threshold_mm": THR_MAIN, "replicate_unit": "seed",
                      "post_hoc": True, "changes_verdict": False},
        "constants": {"floor_csi": FLOOR_CSI, "oracle_csi": ORACLE_CSI,
                      "base_rate_p": p_main, "r_grid": R_GRID,
                      "cost_loss_ratio_definition": "r = C_M / C_F (cost of a "
                      "miss divided by cost of a false alarm)",
                      "cost_formula": "cost(r) = r*(1-POD)*p + FAR*(1-p)",
                      "n_boot": N_BOOT, "rng_seed": RNG_SEED,
                      "n_storm_cases": len(ref_cases)},
        "self_check": {
            "rebuild_max_abs_diff": worst,
            "rebuild_n_comparisons": n_cmp,
            "base_rate_spread": spread,
            "headline_median_max_abs_diff": head_maxdiff,
            "s3_crosses_zero_count": len(cz),
            "s3_crosses_zero_all_true": all(bool(x) for x in cz),
            "s3_verdicts": vd,
        },
        "zero_model": {
            "pairs": ["%s_s%d vs %s_s%d" % (CTRL, a, CTRL, b) for a, b in ctrl_pairs],
            "abs_dc_floor_by_r": zero_floor,
            "break_even_r_abs_median": zero_brk_med,
            "break_even_r_abs_per_pair": zero_brk,
        },
        "control_cost_loss_skill": ctrl_skill,
        "arms": per_arm,
        "case_bootstrap_internal_consistency": boot_ci,
        "threshold_robustness_break_even_r": thr_robust,
    }

    # ------------------------------------------------------------------ 6
    # headline assertions for this round
    # (a) algebraic self-check: DC(r*) must be 0 by construction
    worst_brk = 0.0
    for arm in TREAT:
        for row in per_arm[arm]["per_seed"]:
            if np.isfinite(row["break_even_r"]):
                worst_brk = max(worst_brk, abs(row["dc_at_break_even"]))
    check(worst_brk < 1e-12,
          "algebraic self-check: DC(r*) = 0 at every defined break-even point "
          "(max|DC(r*)| = %.3e over 18 arm-seed cells)" % worst_brk)

    # (b) the decision-relevant statement: where can any arm be cheaper?
    cheaper_beyond_floor = {}
    for arm in TREAT:
        hits = [k for k, v in per_arm[arm]["decision_cost_curves"].items()
                if v["cheaper_and_beyond_floor"]]
        cheaper_beyond_floor[arm] = hits
    low_r = [r for r in R_GRID if r <= 3.0]
    low_r_breaches = []
    for arm in TREAT:
        for r in low_r:
            v = per_arm[arm]["decision_cost_curves"]["r=%g" % r]
            if v["cheaper_and_beyond_floor"]:
                low_r_breaches.append((arm, r))
    check(not low_r_breaches,
          "NO arm is cheaper than the control beyond the zero-model floor at "
          "any cost-loss ratio r <= 3 (a miss costing at most 3x a false "
          "alarm); breaches: %s" % (low_r_breaches or "none"))

    # (c) the only crossover that does clear the floor, and its robustness
    cross_arms = {a: h for a, h in cheaper_beyond_floor.items() if h}
    check(list(cross_arms.keys()) == ["A4b"] and cross_arms["A4b"] == ["r=10"],
          "the ONLY (arm, r) cell that is cheaper than the control beyond the "
          "zero-model floor is A4b at r = 10: %s" % cross_arms)

    ab = per_arm["A4b"]
    ab10 = ab["decision_cost_curves"]["r=10"]
    check(ab10["pass_rate_cheaper"] == "2/3",
          "that A4b crossover is SEED-UNSTABLE: pass rate %s (not 3/3), and "
          "the paired median |DC| = %.5f is only %.2fx the zero-model floor "
          "%.5f" % (ab10["pass_rate_cheaper"], abs(ab10["paired_median_dc"]),
                    ab10["effect_over_floor"], ab10["zero_model_floor_abs_dc"]))

    check(ab["median_r_abs_vs_zero_model"] < 3.0,
          "A4b's break-even ratio r* = %.3f (median) is only %.2fx the "
          "zero-model's own crossover scale |r*| = %.3f -> the crossover "
          "point itself is not separable from seed noise" %
          (ab["break_even_r_median"], ab["median_r_abs_vs_zero_model"],
           zero_brk_med))

    out["headline"] = {
        "cheaper_beyond_zero_floor_by_arm": cheaper_beyond_floor,
        "low_r_breaches_r_le_3": low_r_breaches,
        "break_even_r_median": {a: per_arm[a]["break_even_r_median"]
                                for a in TREAT},
        "break_even_r_sign_consistency": {a: per_arm[a]["break_even_r_sign_consistency"]
                                          for a in TREAT},
        "d_pod_sign_consistency": {a: per_arm[a]["d_pod_sign_consistency"]
                                   for a in TREAT},
        "zero_model_break_even_r_abs_median": zero_brk_med,
        "verdict": "no arm buys decision value at any realistic cost-loss "
                   "ratio; the single high-r crossover (A4b, r=10) is "
                   "seed-unstable and sits inside the zero-model noise scale",
    }

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, sort_keys=True)

    print("=" * 78)
    print("E45 round 60 -- decision-theoretic value axis (cost-loss / CNL)")
    print("=" * 78)
    print("base rate p = %.6f | cost(r) = r*(1-POD)*p + FAR*(1-p) | "
          "r = C_M/C_F" % p_main)
    print("zero model (B0 seed swaps): |dCost| floor by r = %s"
          % {("r=%g" % r): round(zero_floor[r], 8) for r in R_GRID})
    print("zero-model |r*| median = %.4f" % zero_brk_med)
    print()
    print("%-5s %-10s %-10s %-10s %-10s %-8s %-8s" %
          ("arm", "med dPOD", "med dFAR", "r* (med)", "r* cons",
           "minEff/floor", "cheaper r"))
    for arm in TREAT:
        a = per_arm[arm]
        effs = [v["effect_over_floor"] for v in a["decision_cost_curves"].values()]
        wins = [k for k, v in a["decision_cost_curves"].items()
                if v["paired_median_dc"] < 0 and v["beats_zero_floor"]]
        print("%-5s %+10.5f %+10.5f %+10.4f %-10s %8.3f %-8s" %
              (arm, a["median_d_pod"], a["median_d_far"],
               a["break_even_r_median"], a["break_even_r_sign_consistency"],
               max(effs) if effs else float("nan"),
               (",".join(wins) if wins else "none")))
    print()
    for arm in TREAT:
        cs = per_arm[arm]["decision_cost_curves"]
        line = []
        for r in R_GRID:
            v = cs["r=%g" % r]
            line.append("r=%g: %+0.5f (%s)" % (r, v["paired_median_dc"],
                                                v["pass_rate_cheaper"]))
        print("%-4s %s" % (arm, " | ".join(line)))
    print()
    print("control cost-loss skill CS(r): %s"
          % {k: round(v["median"], 4) for k, v in ctrl_skill.items()})
    print()
    for arm in TREAT:
        print("%-4s break-even r* per seed: %s" %
              (arm, [None if not np.isfinite(x) else round(x, 4)
                     for x in per_arm[arm]["break_even_r_per_seed"]]))
    print()
    print("-" * 78)
    for m in POS:
        print("PASS  %s" % m)
    for m in NEG:
        print("FAIL  %s" % m)
    print("-" * 78)
    print("POS %d/%d   NEG %d/%d" % (len(POS), len(POS) + len(NEG),
                                     len(NEG), len(POS) + len(NEG)))
    print("VERDICT: %s" % ("PASS" if not NEG else "FAIL"))
    print("output: %s" % OUT)
    return 0 if not NEG else 1


if __name__ == "__main__":
    sys.exit(main())
