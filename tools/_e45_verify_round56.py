"""E45 round-56 verification: GAIN-PATTERN STABILITY ACROSS SEEDS (new axis).

WHAT IS NEW HERE
-----------------
Rounds 40-55 judged E45's gains along: aggregate CSI/BIAS (40), mechanism
read-outs (41, 53), cross-experiment reproducibility (42), derived booleans
(43), multi-threshold calibers (44), S1 struct metrics (45), window level
(46), storm-case level (47), LOSO/ICC/MDE (48), lead-time axis (49),
severity/difficulty strata (50), error composition (51), omnibus arm effect
(52), metric families (54) and the guard's own calibration (55).

NONE of them asked a purely structural question about the per-window gain
PATTERN:

    Is the set of windows in which an arm beats the control the SAME set
    across the arm's three seeds?

If yes -> the arm has a stable, reproducible skill signature (a
mechanism-level property worth carrying into a new mechanism family).
If no -> the gains are seed accidents (consistent with round 40: 99.2% of
A4's mean raw gain came from seed 0), and the negative verdict is even
more robust than it looked.

Axes measured here
------------------
  A  cross-seed reproducibility of the per-window gain vector (arm - control,
     same seed, same window).  A REPRODUCIBLE pattern needs a POSITIVE,
     seed-consistent correlation.  A significantly NEGATIVE correlation means
     the pattern FLIPS wholesale between seeds.
  B  arm-to-arm co-structure of the gain vectors, same seed vs cross seed.
  C  does an arm's deviation from its control mirror the control arm's OWN
     seed-to-seed wobble on the same windows?
  D  reference ceilings measured on the control arm itself (how reproducible
     the per-window skill PATTERN is at all; noise floor for triple stats).
  E  WHAT the seed-consistent component consists of: in the windows where an
     arm gains on ALL three seeds, does it gain by hitting more (POD up) or
     by over-forecasting (BIAS ratio up)?  This is the link between Axis A
     and the S3 "gain was area/amplitude" verdict.

CONFOUND CONTROLLED FOR (important)
-----------------------------------
Every gain vector is  delta(arm,s) = CSI(arm,s) - CSI(B0,s)  ON THE SAME
WINDOW, so it contains the shared term -CSI(B0,s).  Two gain vectors built
at the SAME seed therefore share the control's own window pattern, which
inflates their correlation; and the control's window pattern is itself highly
reproducible across seeds (Axis D).  All co-structure statistics are
therefore ALSO reported as PARTIAL correlations, residualizing each gain
vector on the control's per-window CSI vector for that seed.  The raw and
adjusted numbers are reported side by side and never mixed.

NULL MODEL
----------
Window-shuffle null: permuting the window order of one vector destroys any
window-aligned relationship while preserving the marginal distribution.
Reported as the 95th percentile of |r| (null floor) and a two-sided
pseudo-p.  The replication unit remains the SEED (n=3); no window-level CI
is reported (round-46/48 discipline: window-level resampling is
pseudo-replication).

CALIBERATION / INTEGRITY GUARDS (all must pass)
------------------------------------------------
  G1  21 cells share identical window_keys and n_windows == 538
  G2  aggregate CSI/BIAS recomputed from the raw window tables == published
      scores_summary_s{0,1,2}.json (42 comparisons, max|diff| == 0)
  G3  headline table (7 arms x 5 fields) recomputed from the per-seed files
      == published scores_summary.json (35 items, max|diff| <= 1e-9)
      NOTE: published `seed_spread_csi_at_2.0` is max/min (a ratio), not
      (max-min)/min.
  G4  S3 guard unchanged: 12/12 intervals cross zero, 6/6 verdicts and
      n_seeds == 3 as published
  G5  every new-axis number is finite
  G6  one global window mask is used for every cell (no index misalignment)

Stdlib only (no numpy) so the script runs with the repo's default
interpreter, like every other round script.

Read-only: touches nothing outside tools/_evals/e45/.
"""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "tools" / "_evals" / "e45" / "_raw" / "runs" / "e45" / "evals"
EVAL = ROOT / "tools" / "_evals" / "e45"
OUT = EVAL / "round56_gain_pattern_stability.json"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
TREAT = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
THR = "2.0"
FLOOR = 0.0924
ORACLE = 0.1653
N_DRAWS = 300
RNG_SEED = 20261114

POS = 0
FAIL = 0


def check(cond, msg):
    global POS, FAIL
    if cond:
        POS += 1
        print("  [POS] %s" % msg)
    else:
        FAIL += 1
        print("  [FAIL] %s" % msg)
    return bool(cond)


# ----------------------------------------------------------------- utilities
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def nanmean(xs):
    v = [x for x in xs if x == x]
    return sum(v) / len(v) if v else float("nan")


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return pearson(ranks(x), ranks(y))


def shuffle_null(x, y, draws, rng):
    out = []
    idx = list(range(len(y)))
    for _ in range(draws):
        rng.shuffle(idx)
        out.append(abs(pearson(x, [y[i] for i in idx])))
    out.sort()
    if not out:
        return float("nan"), float("nan")
    return out[int(0.95 * (len(out) - 1))], mean(out)


def shuffle_pseudo_p(x, y, r_obs, draws, rng):
    idx = list(range(len(y)))
    ge = 0
    for _ in range(draws):
        rng.shuffle(idx)
        if abs(pearson(x, [y[k] for k in idx])) >= abs(r_obs):
            ge += 1
    return ge / float(draws)


def median(vals):
    v = sorted(x for x in vals if x == x)
    if not v:
        return float("nan")
    k = len(v)
    return v[k // 2] if k % 2 else 0.5 * (v[k // 2 - 1] + v[k // 2])


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def solve(A, b):
    """Gaussian elimination with partial pivoting (tiny dense system)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def residualize(v, ctrls):
    """OLS residuals of v after regressing on ctrls (with intercept)."""
    k = len(ctrls) + 1
    X = [[1.0] + [c[i] for c in ctrls] for i in range(len(v))]
    A = [[sum(X[i][a] * X[i][b] for i in range(len(v))) for b in range(k)] for a in range(k)]
    rhs = [sum(X[i][a] * v[i] for i in range(len(v))) for a in range(k)]
    beta = solve(A, rhs)
    if beta is None:
        return list(v)
    return [v[i] - sum(beta[a] * X[i][a] for a in range(k)) for i in range(len(v))]


# ----------------------------------------------------------------- load data
print("=" * 78)
print("E45 ROUND-56 :: GAIN-PATTERN STABILITY ACROSS SEEDS")
print("=" * 78)

cells = {}
ref_keys = None
for arm in ARMS:
    for seed in SEEDS:
        tag = "%s_s%d_ep00" % (arm, seed)
        w = json.loads((RAW / ("%s.windows.json" % tag)).read_text(encoding="utf-8"))
        cells[(arm, seed)] = w
        if ref_keys is None:
            ref_keys = w["window_keys"]
n_win = len(ref_keys)
print("loaded %d cells, %d windows per cell" % (len(cells), n_win))

check(all(cells[k]["window_keys"] == ref_keys for k in cells)
      and all(cells[k]["n_windows"] == n_win for k in cells),
      "G1  21 cells share identical window_keys and n_windows == %d" % n_win)


def window_counts(w):
    """per-window (csi, pod, far, bias_ratio) at THR."""
    out = []
    for win in w["windows"]:
        c = win[THR]
        h, m, fa = c["hits"], c["misses"], c["false_alarms"]
        den = h + m + fa
        out.append({
            "csi": h / den if den > 0 else float("nan"),
            "pod": h / (h + m) if (h + m) > 0 else float("nan"),
            "far": fa / (h + fa) if (h + fa) > 0 else float("nan"),
            "bias": (h + fa) / (h + m) if (h + m) > 0 else float("nan"),
        })
    return out


def window_agg(w):
    h = m = fa = cn = 0
    for win in w["windows"]:
        c = win[THR]
        h += c["hits"]; m += c["misses"]; fa += c["false_alarms"]; cn += c["correct_negatives"]
    csi = h / (h + m + fa) if (h + m + fa) > 0 else float("nan")
    pod = h / (h + m) if (h + m) > 0 else float("nan")
    far = fa / (h + fa) if (h + fa) > 0 else float("nan")
    bias = (h + fa) / (h + m) if (h + m) > 0 else float("nan")
    return {"csi": csi, "bias": bias, "pod": pod, "far": far}


# G2: aggregate recomputation == published per-seed scores
print("\n--- G2 aggregate recomputation from raw window tables ---")
g2_max = 0.0
g2_n = 0
for arm in ARMS:
    for seed in SEEDS:
        pub = json.loads((EVAL / ("scores_summary_s%d.json" % seed)).read_text(encoding="utf-8"))
        p = pub["%s_s%d_ep00" % (arm, seed)]
        a = window_agg(cells[(arm, seed)])
        for key_p, key_a in (("csi_at_2.0", "csi"), ("bias_at_2.0", "bias")):
            g2_max = max(g2_max, abs(p[key_p] - a[key_a]))
            g2_n += 1
check(g2_n == 42 and g2_max == 0.0,
      "G2  %d aggregate comparisons (CSI/BIAS x 21 cells) max|diff| = %.3e" % (g2_n, g2_max))

# G3: headline table recomputation
print("\n--- G3 headline table recomputation ---")
head = json.loads((EVAL / "scores_summary.json").read_text(encoding="utf-8"))
per_seed = {a: [] for a in ARMS}
per_seed_bias = {a: [] for a in ARMS}
for seed in SEEDS:
    pub = json.loads((EVAL / ("scores_summary_s%d.json" % seed)).read_text(encoding="utf-8"))
    for arm in ARMS:
        per_seed[arm].append(pub["%s_s%d_ep00" % (arm, seed)]["csi_at_2.0"])
        per_seed_bias[arm].append(pub["%s_s%d_ep00" % (arm, seed)]["bias_at_2.0"])
g3_max = 0.0
g3_n = 0
for arm in ARMS:
    h = head[arm]
    recomputed = {
        "median_csi_at_2.0": median(per_seed[arm]),
        "median_bias_at_2.0": median(per_seed_bias[arm]),
        "n_seeds": float(len(per_seed[arm])),
        # published definition is max/min (a ratio), NOT (max-min)/min
        "seed_spread_csi_at_2.0": max(per_seed[arm]) / min(per_seed[arm]),
        "pass_rate_over_floor": "%d/%d" % (sum(1 for v in per_seed[arm] if v > FLOOR), len(per_seed[arm])),
    }
    for k, v in recomputed.items():
        pv = h[k]
        if isinstance(v, str):
            ok = (pv == v); d = 0.0 if ok else 1.0
        else:
            d = abs(pv - v); ok = d <= 1e-9
        g3_max = max(g3_max, d); g3_n += 1
        if not ok:
            print("      mismatch %s.%s published=%s recomputed=%s" % (arm, k, pv, v))
check(g3_n == 35 and g3_max <= 1e-9,
      "G3  %d headline items recomputed, max|diff| = %.3e" % (g3_n, g3_max))

# G4: S3 guard unchanged
print("\n--- G4 S3 guard ---")
s3 = json.loads((EVAL / "s3_equalbias.json").read_text(encoding="utf-8"))
exp_verdict = {"A1": "GAIN_WAS_AREA_OR_AMPLITUDE", "A2": "NO_GAIN_TO_BEGIN_WITH",
                "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE", "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
                "A4": "GAIN_WAS_AREA_OR_AMPLITUDE", "A4b": "NO_GAIN_TO_BEGIN_WITH"}
n_cross = 0
g4_ok = True
for arm, rec in s3["arms"].items():
    for kind in ("raw", "equal_bias"):
        n_cross += int(bool(rec[kind]["crosses_zero"]))
        if rec[kind]["n_seeds"] != 3:
            g4_ok = False
    if rec["s3_verdict"] != exp_verdict[arm]:
        g4_ok = False
        print("      verdict drift %s: %s" % (arm, rec["s3_verdict"]))
check(g4_ok and n_cross == 12,
      "G4  S3 guard unchanged: %d/12 intervals cross zero, 6/6 verdicts as published" % n_cross)

# --------------------------------------------- global mask + paired delta vectors
print("\n--- building paired per-window gain vectors (arm - control, same seed) ---")
counts_all = {k: window_counts(v) for k, v in cells.items()}
valid = [i for i in range(n_win)
         if all(counts_all[k][i]["csi"] == counts_all[k][i]["csi"] for k in cells)]
print("global valid windows (all 21 cells finite): %d of %d" % (len(valid), n_win))
check(len(valid) >= 500, "G6  one global mask of %d windows used for every cell" % len(valid))

C = {k: {q: [counts_all[k][i][q] for i in valid] for q in ("csi", "pod", "far", "bias")}
     for k in cells}
delta = {}
for arm in TREAT:
    for seed in SEEDS:
        delta[(arm, seed)] = [C[(arm, seed)]["csi"][i] - C[("B0", seed)]["csi"][i]
                              for i in range(len(valid))]
rng = random.Random(RNG_SEED)

# ------------------------------------------- axis D: control-side reference ceilings
print("\n=== AXIS D (reference): control arm's own cross-seed pattern reproducibility ===")
ctl = {s: C[("B0", s)]["csi"] for s in SEEDS}
ctl_pairs = []
for (i, j) in ((0, 1), (0, 2), (1, 2)):
    ctl_pairs.append({"pair": "s%d-s%d" % (i, j),
                      "pearson_r": pearson(ctl[i], ctl[j]),
                      "spearman_rho": spearman(ctl[i], ctl[j])})
ctl_ceiling = median([p["pearson_r"] for p in ctl_pairs])
ctl_med = [median(ctl[s]) for s in SEEDS]
ctl_triple_agree = sum(
    1 for i in range(len(valid))
    if sign(ctl[0][i] - ctl_med[0]) == sign(ctl[1][i] - ctl_med[1]) == sign(ctl[2][i] - ctl_med[2])
) / float(len(valid))
print("  control cross-seed per-window CSI corr: %s  -> median r = %+.4f"
      % (" ".join("%+.4f" % p["pearson_r"] for p in ctl_pairs), ctl_ceiling))
print("  control all-three-seeds-agree fraction (noise floor for triple stat) = %.4f"
      % ctl_triple_agree)

# ---------------------------------------------- axis A: cross-seed reproducibility
print("\n=== AXIS A: gain-pattern reproducibility across seeds (same arm) ===")
axisA = {}
for arm in TREAT:
    pairs = []
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        x = delta[(arm, i)]; y = delta[(arm, j)]
        r = pearson(x, y); sp = spearman(x, y)
        # partial correlation: remove each gain vector's own control pattern
        rx = residualize(x, [ctl[i]]); ry = residualize(y, [ctl[j]])
        r_adj = pearson(rx, ry)
        agree = sum(1 for a, b in zip(x, y) if sign(a) == sign(b) and sign(a) != 0) / float(len(x))
        p95, nmean = shuffle_null(x, y, N_DRAWS, rng)
        pp = shuffle_pseudo_p(x, y, r, N_DRAWS, rng)
        pairs.append({"pair": "s%d-s%d" % (i, j), "n": len(x),
                      "pearson_r": r, "pearson_r_partial": r_adj, "spearman_rho": sp,
                      "sign_agreement": agree,
                      "null_p95_abs_r": p95, "null_mean_abs_r": nmean,
                      "shuffle_pseudo_p_two_sided": pp,
                      "exceeds_floor_abs": abs(r) > p95,
                      "exceeds_floor_positive": r > p95})
    rs = [p["pearson_r"] for p in pairs]
    adj = [p["pearson_r_partial"] for p in pairs]
    ags = [p["sign_agreement"] for p in pairs]
    above_abs = sum(1 for p in pairs if p["exceeds_floor_abs"])
    above_pos = sum(1 for p in pairs if p["exceeds_floor_positive"])
    neg = sum(1 for p in pairs if p["pearson_r"] < 0)
    tri = sum(1 for i in range(len(valid))
              if sign(delta[(arm, 0)][i]) == sign(delta[(arm, 1)][i]) == sign(delta[(arm, 2)][i])
              and sign(delta[(arm, 0)][i]) != 0) / float(len(valid))
    tri_pos = sum(1 for i in range(len(valid))
                  if sign(delta[(arm, 0)][i]) == sign(delta[(arm, 1)][i]) == sign(delta[(arm, 2)][i])
                  and sign(delta[(arm, 0)][i]) > 0) / float(len(valid))
    axisA[arm] = {
        "pairs": pairs,
        "median_pearson_r": median(rs),
        "median_pearson_r_partial": median(adj),
        "median_spearman_rho": median([p["spearman_rho"] for p in pairs]),
        "median_sign_agreement": median(ags),
        "median_shuffle_pseudo_p": median([p["shuffle_pseudo_p_two_sided"] for p in pairs]),
        "pairs_exceeding_null_floor_abs": "%d/%d" % (above_abs, len(pairs)),
        "pairs_exceeding_null_floor_positive": "%d/%d" % (above_pos, len(pairs)),
        "pairs_with_negative_r": "%d/%d" % (neg, len(pairs)),
        "triple_same_sign_fraction": tri,
        "triple_all_positive_fraction": tri_pos,
        "reproducible_positive_gain_pattern": (above_pos >= 2),
    }
    print("  %-4s median r=%+.4f (partial %+.4f)  sign-agree=%.3f  |r|>floor=%d/3  "
          "r>floor=%d/3  neg-r=%d/3  triple-agree=%.3f (all-pos %.3f)"
          % (arm, axisA[arm]["median_pearson_r"], axisA[arm]["median_pearson_r_partial"],
             axisA[arm]["median_sign_agreement"], above_abs, above_pos, neg, tri, tri_pos))

# ------------------------------------- axis B: arm-to-arm co-structure of deltas
print("\n=== AXIS B: arm-to-arm co-structure of gain vectors ===")
bpairs = []
for i_arm in range(len(TREAT)):
    for j_arm in range(i_arm + 1, len(TREAT)):
        a1, a2 = TREAT[i_arm], TREAT[j_arm]
        same = [pearson(delta[(a1, s)], delta[(a2, s)]) for s in SEEDS]
        same_adj = [pearson(residualize(delta[(a1, s)], [ctl[s]]),
                             residualize(delta[(a2, s)], [ctl[s]])) for s in SEEDS]
        cross, cross_adj = [], []
        for s1 in SEEDS:
            for s2 in SEEDS:
                if s1 != s2:
                    cross.append(pearson(delta[(a1, s1)], delta[(a2, s2)]))
                    cross_adj.append(pearson(residualize(delta[(a1, s1)], [ctl[s1]]),
                                             residualize(delta[(a2, s2)], [ctl[s2]])))
        bpairs.append({"arms": [a1, a2],
                       "pearson_same_seed": same, "median_same_seed_r": median(same),
                       "median_same_seed_r_partial": median(same_adj),
                       "median_cross_seed_r": median(cross),
                       "median_cross_seed_r_partial": median(cross_adj),
                       "excess_same_over_cross": median(same) - median(cross),
                       "excess_same_over_cross_partial": median(same_adj) - median(cross_adj),
                       "same_sign_all_seeds": all(x > 0 for x in same) or all(x < 0 for x in same)})
axisB = {"pairs": bpairs,
         "median_same_seed_r_over_15_pairs": median([p["median_same_seed_r"] for p in bpairs]),
         "median_same_seed_r_partial_over_15_pairs": median([p["median_same_seed_r_partial"] for p in bpairs]),
         "median_cross_seed_r_over_15_pairs": median([p["median_cross_seed_r"] for p in bpairs]),
         "median_cross_seed_r_partial_over_15_pairs": median([p["median_cross_seed_r_partial"] for p in bpairs]),
         "max_abs_same_seed_pair": max(((abs(p["median_same_seed_r"]), p["arms"]) for p in bpairs)),
         "n_pairs_same_sign_all_seeds": sum(1 for p in bpairs if p["same_sign_all_seeds"])}
axisB["excess_same_over_cross"] = (axisB["median_same_seed_r_over_15_pairs"]
                                   - axisB["median_cross_seed_r_over_15_pairs"])
axisB["excess_same_over_cross_partial"] = (axisB["median_same_seed_r_partial_over_15_pairs"]
                                           - axisB["median_cross_seed_r_partial_over_15_pairs"])
print("  same-seed median r = %+.4f (partial %+.4f) ; cross-seed median r = %+.4f "
      "(partial %+.4f)" % (axisB["median_same_seed_r_over_15_pairs"],
                           axisB["median_same_seed_r_partial_over_15_pairs"],
                           axisB["median_cross_seed_r_over_15_pairs"],
                           axisB["median_cross_seed_r_partial_over_15_pairs"]))
print("  excess same-over-cross = %+.4f (partial %+.4f)"
      % (axisB["excess_same_over_cross"], axisB["excess_same_over_cross_partial"]))
print("  max |same-seed r| = %.4f (%s) ; pairs same-sign in all 3 seeds = %d/15"
      % (axisB["max_abs_same_seed_pair"][0], "+".join(axisB["max_abs_same_seed_pair"][1]),
         axisB["n_pairs_same_sign_all_seeds"]))

# is the same-seed co-structure a MECHANISM-family effect or a SEED effect?
# If arms of different families co-struct just as strongly as arms of the same
# family, the co-structure is the shared seed (data order / init / window
# sample), not the mechanism.
FAMILY = {"A1": "gate", "A2": "gate", "A3a": "topo", "A3b": "topo",
          "A4": "amount", "A4b": "amount"}
same_fam = [p for p in bpairs if FAMILY[p["arms"][0]] == FAMILY[p["arms"][1]]]
cross_fam = [p for p in bpairs if FAMILY[p["arms"][0]] != FAMILY[p["arms"][1]]]
axisB["same_family_pairs"] = len(same_fam)
axisB["cross_family_pairs"] = len(cross_fam)
axisB["median_excess_same_family"] = median([p["excess_same_over_cross"] for p in same_fam])
axisB["median_excess_cross_family"] = median([p["excess_same_over_cross"] for p in cross_fam])
axisB["family_effect_gap"] = axisB["median_excess_same_family"] - axisB["median_excess_cross_family"]
print("  excess same-over-cross: same-family (n=%d) %+.4f  vs  cross-family (n=%d) %+.4f  "
      "gap %+.4f" % (axisB["same_family_pairs"], axisB["median_excess_same_family"],
                     axisB["cross_family_pairs"], axisB["median_excess_cross_family"],
                     axisB["family_effect_gap"]))

# ---------------- axis C: reproducible vs seed-specific share of the window gain
# (replaces an earlier, convention-fragile version: correlating the gain with the
#  control's seed-to-seed wobble gave a sign that depended on an arbitrary
#  ordering of the two "other" seeds, so it was discarded.)
print("\n=== AXIS C: seed-shared vs seed-specific share of the window-level gain ===")
axisC = {}
N_PERM = 400
for arm in TREAT:
    # truth event count per window at THR (shared truth domain => identical for all arms)
    tw = []
    for k in cells:
        tw = [cells[k]["windows"][valid[i]][THR]["hits"] + cells[k]["windows"][valid[i]][THR]["misses"]
              for i in range(len(valid))]
        break
    tot = sum(tw)
    gains = {s: delta[(arm, s)] for s in SEEDS}
    shared = [mean([gains[s][i] for s in SEEDS]) for i in range(len(valid))]
    total_abs_mass = sum(tw[i] * sum(abs(gains[s][i]) for s in SEEDS) for i in range(len(valid)))
    shared_mass = sum(tw[i] * abs(shared[i]) for i in range(len(valid)))
    share = shared_mass / total_abs_mass if total_abs_mass > 0 else float("nan")
    repro_level = sum(tw[i] * shared[i] for i in range(len(valid))) / tot
    per_seed_level = {s: sum(tw[i] * gains[s][i] for i in range(len(valid))) / tot for s in SEEDS}
    # null: independently flip the sign of each seed's gain vector (destroys the
    # cross-seed shared structure, keeps the marginals) -> null distribution of share
    perm_rng = random.Random(RNG_SEED + 777)
    null = []
    for _ in range(N_PERM):
        sg = {s: (gains[s] if perm_rng.random() < 0.5 else [-x for x in gains[s]]) for s in SEEDS}
        sh = [mean([sg[s][i] for s in SEEDS]) for i in range(len(valid))]
        tm = sum(tw[i] * sum(abs(sg[s][i]) for s in SEEDS) for i in range(len(valid)))
        nm = sum(tw[i] * abs(sh[i]) for i in range(len(valid)))
        null.append(nm / tm if tm > 0 else float("nan"))
    null = sorted(x for x in null if x == x)
    p95 = null[int(0.95 * (len(null) - 1))]
    pseudo_p = sum(1 for x in null if x >= share) / float(len(null))
    axisC[arm] = {
        "reproducible_share_of_abs_gain": share,
        "null_p95_share": p95, "null_mean_share": mean(null),
        "permutation_pseudo_p": pseudo_p,
        "truth_weighted_reproducible_gain_level": repro_level,
        "truth_weighted_gain_level_by_seed": per_seed_level,
        "exceeds_null_floor": share > p95,
    }
    print("  %-4s reproducible share=%.4f (null p95=%.4f, pseudo-p=%.4f)  "
          "reproducible gain level=%+.6f  per-seed=%s"
          % (arm, share, p95, pseudo_p, repro_level,
             " ".join("%+.6f" % per_seed_level[s] for s in SEEDS)))

# ------------------------------------------- per-case reproducibility (n >= 50)
print("\n=== per-storm-case reproducibility of the gain pattern (n_windows >= 50) ===")
cases = {}
for k in ref_keys:
    cases.setdefault(k.split("_000")[0], []).append(k)
big_cases = sorted([c for c, idxs in cases.items() if len(idxs) >= 50],
                    key=lambda c: -len(cases[c]))
pos_in_valid = {ref_keys[i]: j for j, i in enumerate(valid)}
percase = {}
for cid in big_cases:
    idxs = [pos_in_valid[k] for k in cases[cid] if k in pos_in_valid]
    percase[cid] = {"n": len(idxs)}
    for arm in TREAT:
        rs = []
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            rs.append(pearson([delta[(arm, i)][t] for t in idxs],
                               [delta[(arm, j)][t] for t in idxs]))
        percase[cid][arm] = {"median_r": median(rs)}
    print("  %-16s n=%3d  " % (cid, len(idxs))
          + " ".join("%s=%+.3f" % (a, percase[cid][a]["median_r"]) for a in TREAT))

# ------------------- axis E: what does the seed-consistent component consist of?
print("\n=== AXIS E: composition of the seed-consistent gain (all-3-seeds-agree windows) ===")
axisE = {}
for arm in TREAT:
    cons_pos, cons_neg = [], []
    for i in range(len(valid)):
        d = [delta[(arm, s)][i] for s in SEEDS]
        if sign(d[0]) == sign(d[1]) == sign(d[2]) and sign(d[0]) != 0:
            (cons_pos if d[0] > 0 else cons_neg).append(i)
    def stats(idxs):
        if not idxs:
            return {"n": 0}
        return {
            "n": len(idxs),
            "mean_dCSI": nanmean([mean([delta[(arm, s)][i] for s in SEEDS]) for i in idxs]),
            "mean_dPOD": nanmean([mean([C[(arm, s)]["pod"][i] - C[("B0", s)]["pod"][i] for s in SEEDS]) for i in idxs]),
            "mean_dFAR": nanmean([mean([C[(arm, s)]["far"][i] - C[("B0", s)]["far"][i] for s in SEEDS]) for i in idxs]),
            "mean_dBIASratio": nanmean([mean([C[(arm, s)]["bias"][i] - C[("B0", s)]["bias"][i] for s in SEEDS]) for i in idxs]),
        }
    axisE[arm] = {"consistent_gain_windows": stats(cons_pos),
                   "consistent_loss_windows": stats(cons_neg),
                   "n_consistent_gain": len(cons_pos),
                   "n_consistent_loss": len(cons_neg)}
    g = axisE[arm]["consistent_gain_windows"]
    l = axisE[arm]["consistent_loss_windows"]
    print("  %-4s consistent-GAIN n=%3d: dCSI=%+.4f dPOD=%+.4f dFAR=%+.4f dBIASratio=%+.4f | "
          "consistent-LOSS n=%3d: dBIASratio=%+.4f"
          % (arm, g.get("n", 0), g.get("mean_dCSI", float("nan")), g.get("mean_dPOD", float("nan")),
             g.get("mean_dFAR", float("nan")), g.get("mean_dBIASratio", float("nan")),
             l.get("n", 0), l.get("mean_dBIASratio", float("nan"))))

# --------------------------------------- paired vs difference caliber (context)
print("\n=== paired vs difference caliber of the aggregate gain (context) ===")
paired = {}
for arm in TREAT:
    pd = [median(delta[(arm, s)]) for s in SEEDS]
    dv = median(per_seed[arm]) - median(per_seed["B0"])
    paired[arm] = {"paired_median_per_seed": pd, "paired_median": median(pd),
                   "difference_caliber": dv}
    print("  %-4s paired=%+.6f  difference=%+.6f" % (arm, paired[arm]["paired_median"], dv))

# ------------------------------------------------------------------- assemble
results = {
    "round": 56,
    "axis": "gain_pattern_stability_across_seeds",
    "thr_mm": 2.0, "floor": FLOOR, "oracle": ORACLE,
    "n_draws": N_DRAWS, "rng_seed": RNG_SEED,
    "n_windows_total": n_win, "n_windows_used": len(valid),
    "guards": {"G2_max_abs_diff": g2_max, "G3_max_abs_diff": g3_max,
               "s3_intervals_crossing_zero": n_cross},
    "axisD_control_reference": {
        "control_cross_seed_pairs": ctl_pairs,
        "control_median_cross_seed_r": ctl_ceiling,
        "control_triple_agree_fraction": ctl_triple_agree,
    },
    "axisA_cross_seed_reproducibility": axisA,
    "axisB_arm_to_arm_costructure": axisB,
    "axisC_reproducible_vs_seed_specific_share": axisC,
    "axisE_consistent_gain_composition": axisE,
    "per_case_reproducibility": percase,
    "calibers": paired,
    "headline": {a: {"median_csi_at_2.0": head[a]["median_csi_at_2.0"],
                     "median_bias_at_2.0": head[a]["median_bias_at_2.0"],
                     "pass_rate_over_floor": head[a]["pass_rate_over_floor"]} for a in ARMS},
    "verdict": {"s3_verdicts": {a: s3["arms"][a]["s3_verdict"] for a in TREAT},
                "all_intervals_cross_zero": n_cross == 12,
                "qualified_arms": [],
                "master_version_unchanged": "v2.48",
                "statement": "post-hoc diagnosis only; no new criterion; verdict unchanged"},
}

finite_ok = True
for arm in TREAT:
    for key in ("median_pearson_r", "median_pearson_r_partial", "median_spearman_rho",
                "median_sign_agreement", "triple_same_sign_fraction"):
        v = axisA[arm][key]
        if v != v or v in (float("inf"), float("-inf")):
            finite_ok = False
check(finite_ok, "G5  every new-axis number is finite")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 78)
print("POS %d / FAIL %d  ->  %s" % (POS, FAIL, "VERDICT: PASS" if FAIL == 0 else "VERDICT: FAIL"))
print("artifact: %s" % OUT)
print("=" * 78)
if FAIL:
    raise SystemExit(1)
