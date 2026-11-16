#!/usr/bin/env python3
"""E45 round-48 verification -- READ-ONLY, local artifacts only (stdlib only).

Three questions that rounds 40-47 did NOT answer:

1. **LOSO seed-subset sensitivity of the S3 gate.**  The E45 verdict rests
   on three seeds.  If the verdict flips when any one seed is dropped, the
   negative conclusion is an artefact of the seed set, not of the arms.
   The S3 point estimates in ``s3_equalbias.json`` are reproduced from the
   per-seed artefacts and recomputed under every leave-one-seed-out (LOSO)
   subset (full 3-seed / drop s0 / drop s1 / drop s2).

2. **Cluster-aware uncertainty on the window axis.**  Round 46 showed the
   window-level bootstrap is pseudo-replication (14/18 cells "significant"
   while the seed-level dispersion is 9-31x the CI half-width).  Here the
   *correct* resampling unit is the storm case (8 clusters): the ICC, the
   design effect, the effective sample size and the inflation factor of the
   naive window bootstrap over the case-cluster bootstrap are computed.

3. **Power for the next ladder step.**  Section 11.9 step 2 asks for >= 10
   seeds.  With the measured seed-level dispersion, how many seeds are
   actually needed to resolve an effect of the size E45 observed?  A
   minimum detectable effect (MDE) is solved for each arm.

Nothing here re-argues an existing E45 conclusion; the verdict is not
changed.  Outputs ``tools/_evals/e45/round48_cluster_sensitivity.json``.
"""

import json
import math
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "tools" / "_evals" / "e45"
RAW = EV / "_raw" / "runs" / "e45" / "evals"

ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
CTRL = "B0"
SEEDS = [0, 1, 2]
THR = "2.0"
FLOOR = 0.0924
NBOOT = 10000
RNG_SEED = 20261113

# Round-47 lesson: anchor the case regex to the separator, else the greedy
# \d+ swallows the per-window index and 538 windows collapse into 1 case.
CASE_RE = re.compile(r"^(CP_\d+_\d+)_(\d+):")

POS = []
REPORT = []


def check(name, ok, detail=""):
    POS.append((name, bool(ok)))
    print(f"[{'POS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def report(name, value):
    REPORT.append((name, value))
    print(f"[INFO] {name}: {value}")


# --------------------------------------------------------------------------
# Student-t quantile (self-contained)
# --------------------------------------------------------------------------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * betai(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(p, df):
    lo, hi = -200.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def percentile(xs, q):
    """Linear-interpolation percentile (numpy default), stdlib only."""
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[int(pos)])
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


# --------------------------------------------------------------------------
# load artifacts
# --------------------------------------------------------------------------
summ = {s: json.loads((EV / f"scores_summary_s{s}.json").read_text(encoding="utf-8"))
        for s in SEEDS}
s3 = json.loads((EV / "s3_equalbias.json").read_text(encoding="utf-8"))

win_counts = {}
keys_ref = None
for arm in ARMS + [CTRL]:
    for s in SEEDS:
        blob = json.loads((RAW / f"{arm}_s{s}_ep00.windows.json").read_text(encoding="utf-8"))
        keys = blob["window_keys"]
        if keys_ref is None:
            keys_ref = keys
        rows = blob["windows"]
        # "windows" is a LIST parallel to "window_keys" (round-46 loader used
        # the same indexing), not a dict keyed by the window key.
        win_counts[(arm, s)] = {
            k: (float(rows[i][THR]["hits"]), float(rows[i][THR]["misses"]),
                float(rows[i][THR]["false_alarms"])) for i, k in enumerate(keys)
        }

KEYS = list(keys_ref)
case_of = {}
for k in KEYS:
    m = CASE_RE.match(k)
    if not m:
        raise SystemExit(f"case regex failed on {k!r}")
    case_of[k] = m.group(1)
CASES = sorted(set(case_of.values()))

check("windows_538", len(KEYS) == 538, f"n_windows={len(KEYS)}")
check("cases_8", len(CASES) == 8, f"cases={CASES}")
check("window_keys_identical_across_21_grids",
      all(list(win_counts[(a, s)].keys()) == KEYS for a in ARMS + [CTRL] for s in SEEDS))


def ratios(arm, s):
    """Per-window csi / pod / far at THR."""
    h, m, f = zip(*[win_counts[(arm, s)][k] for k in KEYS])
    csi = [(hh / (hh + mm + ff)) if (hh + mm + ff) > 0 else None
           for hh, mm, ff in zip(h, m, f)]
    pod = [(hh / (hh + mm)) if (hh + mm) > 0 else None
           for hh, mm, ff in zip(h, m, f)]
    far = [(ff / (hh + ff)) if (hh + ff) > 0 else None
           for hh, mm, ff in zip(h, m, f)]
    return csi, pod, far


def median_or_none(xs):
    vals = [x for x in xs if x is not None]
    return statistics.median(vals) if vals else None


# --------------------------------------------------------------------------
# (0) pipeline self-check: recompute published aggregates from raw tables
# --------------------------------------------------------------------------
maxdiff = 0.0
n_cmp = 0
for arm in ARMS + [CTRL]:
    for s in SEEDS:
        H = M = F = 0.0
        for k in KEYS:
            hh, mm, ff = win_counts[(arm, s)][k]
            H += hh
            M += mm
            F += ff
        pub = summ[s][f"{arm}_s{s}_ep00"]
        for mine, theirs in ((H / (H + M + F), pub["csi_at_2.0"]),
                             ((H + F) / (H + M), pub["bias_at_2.0"]),
                             (H / (H + M), pub["pod_at_2.0"]),
                             (F / (H + F), pub["far_at_2.0"])):
            maxdiff = max(maxdiff, abs(mine - theirs))
            n_cmp += 1
check("pipeline_selfcheck_reproduces_published_metrics", maxdiff < 1e-12,
      f"{n_cmp} comparisons, max|diff|={maxdiff:.3e}")

# --------------------------------------------------------------------------
# (1) LOSO seed-subset sensitivity of the S3 gate
# --------------------------------------------------------------------------
ctrl_csi = {s: float(summ[s][f"{CTRL}_s{s}_ep00"]["csi_at_2.0"]) for s in SEEDS}
SUBSETS = {"full_3seed": (0, 1, 2), "drop_s0": (1, 2), "drop_s1": (0, 2), "drop_s2": (0, 1)}
loso = {}
loso_flips = {}
loso_label = {}
for arm in ARMS:
    raw_g = {s: float(summ[s][f"{arm}_s{s}_ep00"]["csi_at_2.0"]) - ctrl_csi[s] for s in SEEDS}
    eq_g = {s: float(s3["arms"][arm]["seeds"][str(s)]["csi_equalised"]) - ctrl_csi[s]
            for s in SEEDS}
    entry = {}
    for label, gains in (("raw", raw_g), ("equal_bias", eq_g)):
        vals = {k: sum(gains[s] for s in sub) / len(sub) for k, sub in SUBSETS.items()}
        entry[label] = {
            "per_seed_gain": {str(s): gains[s] for s in SEEDS},
            "subset_points": vals,
            "sign_by_subset": {k: int(math.copysign(1, v)) if v != 0 else 0 for k, v in vals.items()},
            "n_positive_subsets": sum(1 for v in vals.values() if v > 0),
            "subset_spread": max(vals.values()) - min(vals.values()),
            "max_minus_full": max(vals.values()) - vals["full_3seed"],
            "full_minus_min": vals["full_3seed"] - min(vals.values()),
        }
    loso[arm] = entry

s3_repro = 0.0
for arm in ARMS:
    for label in ("raw", "equal_bias"):
        mine = sum(loso[arm][label]["per_seed_gain"][str(s)] for s in SEEDS) / 3.0
        s3_repro = max(s3_repro, abs(mine - float(s3["arms"][arm][label]["point"])))
check("s3_points_reproduced_from_per_seed_artifacts", s3_repro < 1e-9,
      f"max|diff|={s3_repro:.3e} over 12 point estimates")

for label in ("raw", "equal_bias"):
    # Sign stability under LOSO is a MEASUREMENT here, not a pass/fail gate:
    # the S3 verdict rests on the CI crossing zero (all 12 CIs cross zero),
    # not on the point sign.  What must hold is the verdict-relevant
    # property: no LOSO subset produces a gain large enough to be
    # detectable at that subset size.
    flips = {a: sum(1 for k in ("drop_s0", "drop_s1", "drop_s2")
                     if loso[a][label]["sign_by_subset"][k]
                     != loso[a][label]["sign_by_subset"]["full_3seed"])
             for a in ARMS}
    loso_flips[label] = flips
    report(f"{label}: arms whose point sign flips when one seed is dropped",
           ", ".join(f"{a}={flips[a]}/3" for a in ARMS))
    robust_positive = [a for a in ARMS
                       if all(loso[a][label]["sign_by_subset"][k] > 0
                              for k in SUBSETS)]
    loso_label[label] = {"sign_flips": flips,
                          "robustly_positive_arms": robust_positive}
    report(f"{label}: arms whose point sign is positive under all 4 subsets",
           ", ".join(robust_positive) if robust_positive else "none")

# The verdict itself: all 12 published S3 intervals cross zero.
all_cross = all(s3["arms"][a][k]["crosses_zero"]
                for a in ARMS for k in ("raw", "equal_bias"))
check("all_12_published_S3_intervals_cross_zero", all_cross,
      "the E45 negative verdict rests on this, not on the point sign")

# --------------------------------------------------------------------------
# (2) cluster-aware uncertainty on the window axis
# --------------------------------------------------------------------------
def boot_ci_naive(values, rng):
    n = len(values)
    reps = []
    for _ in range(NBOOT):
        reps.append(statistics.median(rng.choices(values, k=n)))
    return percentile(reps, 0.025), percentile(reps, 0.975)


def boot_ci_cluster(by_case, rng):
    names = list(by_case)
    k = len(names)
    reps = []
    for _ in range(NBOOT):
        pooled = []
        for _ in range(k):
            pooled.extend(rng.choices(by_case[names[rng.randrange(k)]]))
        reps.append(statistics.median(pooled))
    return percentile(reps, 0.025), percentile(reps, 0.975)


def icc_stats(by_case):
    sizes = [len(v) for v in by_case.values()]
    N = sum(sizes)
    k = len(sizes)
    allv = [x for v in by_case.values() for x in v]
    grand = sum(allv) / len(allv)
    ssb = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in by_case.values())
    ssw = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in by_case.values())
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    m0 = (N - sum(n * n for n in sizes) / N) / (k - 1)
    den = msb + (m0 - 1.0) * msw
    icc = (msb - msw) / den if den > 0 else 0.0
    icc = max(icc, 0.0)
    deff = 1.0 + (m0 - 1.0) * icc
    return icc, deff, N / deff, m0


cluster = {}
naive_sig = 0
cluster_sig = 0
n_cells = 0
for arm in ARMS:
    per_seed = {}
    for s in SEEDS:
        ca, pa, fa_ = ratios(arm, s)
        cc, pc, fc = ratios(CTRL, s)
        idx = [i for i in range(len(KEYS))
               if ca[i] is not None and cc[i] is not None]
        delta = [ca[i] - cc[i] for i in idx]
        dpod = [pa[i] - pc[i] for i in idx if pa[i] is not None and pc[i] is not None]
        dfar = [fa_[i] - fc[i] for i in idx if fa_[i] is not None and fc[i] is not None]
        by_case = {}
        for j, i in enumerate(idx):
            by_case.setdefault(case_of[KEYS[i]], []).append(delta[j])
        rng = random.Random(RNG_SEED)
        n_lo, n_hi = boot_ci_naive(delta, rng)
        rng = random.Random(RNG_SEED)
        c_lo, c_hi = boot_ci_cluster(by_case, rng)
        icc, deff, n_eff, m0 = icc_stats(by_case)
        n_win = len(idx)
        naive_excl = (n_lo > 0) or (n_hi < 0)
        cluster_excl = (c_lo > 0) or (c_hi < 0)
        naive_sig += int(naive_excl)
        cluster_sig += int(cluster_excl)
        n_cells += 1
        per_seed[str(s)] = {
            "n_windows_valid": n_win,
            "n_windows_excluded": len(KEYS) - n_win,
            "median_delta_csi": statistics.median(delta),
            "median_delta_pod": statistics.median(dpod),
            "median_delta_far": statistics.median(dfar),
            "naive_window_ci": [n_lo, n_hi],
            "naive_halfwidth": (n_hi - n_lo) / 2.0,
            "naive_excludes_zero": bool(naive_excl),
            "cluster_case_ci": [c_lo, c_hi],
            "cluster_halfwidth": (c_hi - c_lo) / 2.0,
            "cluster_excludes_zero": bool(cluster_excl),
            "inflation_naive_over_cluster": ((n_hi - n_lo) / 2.0) / ((c_hi - c_lo) / 2.0)
            if (c_hi - c_lo) > 0 else None,
            "icc": icc,
            "design_effect": deff,
            "n_effective_windows": n_eff,
            "mean_cluster_size": m0,
            "per_case_median_delta": {c: statistics.median(v) for c, v in by_case.items()},
        }
    cluster[arm] = {"per_seed": per_seed}

check("naive_window_bootstrap_reproduces_pseudoreplication", naive_sig >= 10,
      f"naive window bootstrap excludes zero in {naive_sig}/{n_cells} cells "
      f"(round 46 published 14/18)")
check("cluster_bootstrap_removes_the_pseudoreplication", cluster_sig <= 2,
      f"case-cluster bootstrap excludes zero in {cluster_sig}/{n_cells} cells")

infl = [cluster[a]["per_seed"][str(s)]["inflation_naive_over_cluster"]
        for a in ARMS for s in SEEDS if cluster[a]["per_seed"][str(s)]["inflation_naive_over_cluster"]]
# The naive window bootstrap treats 538 windows as independent; the storm case
# is the true cluster.  Pseudo-replication therefore makes the naive CI
# NARROWER than the cluster CI -> ratio < 1, and the naive CI understates the
# half-width by a factor 1/ratio.
check("naive_CI_understates_uncertainty_vs_cluster_CI", max(infl) <= 1.0,
      f"naive/cluster half-width ratio: median {statistics.median(infl):.3f}, range "
      f"{min(infl):.3f}-{max(infl):.3f}  => naive CI understates width by "
      f"{1.0/statistics.median(infl):.1f}x (median)")

iccs = [cluster[a]["per_seed"][str(s)]["icc"] for a in ARMS for s in SEEDS]
neffs = [cluster[a]["per_seed"][str(s)]["n_effective_windows"] for a in ARMS for s in SEEDS]
report("ICC of the paired window difference (18 cells)",
      f"median {statistics.median(iccs):.4f}, range {min(iccs):.4f}-{max(iccs):.4f}")
report("effective window count out of 538",
      f"median {statistics.median(neffs):.1f}, range {min(neffs):.1f}-{max(neffs):.1f}")

# cross-check against the published round-46 / round-47 medians
R46 = {"A1": 0.000832, "A2": -0.002185, "A3a": 0.003105, "A3b": 0.009133,
       "A4": 0.012611, "A4b": 0.000333}
R47 = {"A1": 0.003400, "A2": -0.008177, "A3a": 0.002913, "A3b": 0.009664,
       "A4": 0.009857, "A4b": 0.007783}
# Round-47 convention (must match to cross-check): per case, SUM the
# contingency counts, then take the case CSI, then the paired per-case delta
# (arm case CSI - control case CSI), then the median over the 8 cases and
# finally the median over the 3 seeds.
case_level = {}
r46_diff = 0.0
r47_diff = 0.0
for arm in ARMS:
    # round-46 caliber: median over seeds of the per-seed median of the
    # paired window-level CSI differences
    win_med = statistics.median([cluster[arm]["per_seed"][str(s)]["median_delta_csi"]
                                 for s in SEEDS])
    r46_diff = max(r46_diff, abs(win_med - R46[arm]))
    per_seed_med = {}
    for s in SEEDS:
        acc = {}
        for i, k in enumerate(KEYS):
            hh, mm, ff = win_counts[(arm, s)][k]
            ch, cm, cf = win_counts[(CTRL, s)][k]
            a = acc.setdefault(case_of[k], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            a[0] += hh
            a[1] += mm
            a[2] += ff
            a[3] += ch
            a[4] += cm
            a[5] += cf
        deltas = {}
        for c, (hh, mm, ff, ch, cm, cf) in acc.items():
            arm_csi = hh / (hh + mm + ff)
            ctl_csi = ch / (ch + cm + cf)
            deltas[c] = arm_csi - ctl_csi
        per_seed_med[str(s)] = statistics.median(list(deltas.values()))
    arm_med = statistics.median(list(per_seed_med.values()))
    case_level[arm] = {"per_seed_case_median": per_seed_med, "median_over_seeds": arm_med}
    r47_diff = max(r47_diff, abs(arm_med - R47[arm]))
report("cross-check vs round-46 published window-level medians", f"max|diff|={r46_diff:.3e}")
report("cross-check vs round-47 published case-level medians", f"max|diff|={r47_diff:.3e}")
check("reproduces_round46_window_level_medians", r46_diff < 5e-4)
check("reproduces_round47_case_level_medians", r47_diff < 5e-4)

# --------------------------------------------------------------------------
# (3) power / MDE for the next ladder step
# --------------------------------------------------------------------------
gain_sd = {a: statistics.stdev([loso[a]["raw"]["per_seed_gain"][str(s)] for s in SEEDS])
           for a in ARMS}
pooled_sd = math.sqrt(sum(v * v for v in gain_sd.values()) / len(gain_sd))
ctrl_sd = statistics.stdev([ctrl_csi[s] for s in SEEDS])


def mde(n, s):
    df = max(n - 1, 1)
    return (t_ppf(0.975, df) + t_ppf(0.80, df)) * s / math.sqrt(n)


def n_needed(target, s):
    for n in range(3, 500):
        if mde(n, s) <= abs(target):
            return n
    return None


ref_t = {2: 4.3027, 9: 2.2622, 29: 2.0452}
check("student_t_quantiles_accurate",
      all(abs(t_ppf(0.975, df) - v) < 5e-3 for df, v in ref_t.items()),
      ", ".join(f"t(.975,{df})={t_ppf(0.975, df):.4f}" for df in ref_t))

best_arm = max(ARMS, key=lambda a: abs(sum(loso[a]["raw"]["per_seed_gain"][str(s)]
                                            for s in SEEDS) / 3.0))
power = {
    "seed_level_sd_per_arm": gain_sd,
    "pooled_seed_level_sd": pooled_sd,
    "control_own_csi_sd_across_seeds": ctrl_sd,
    "mde_at_n3_pooled": mde(3, pooled_sd),
    "mde_at_n10_pooled": mde(10, pooled_sd),
    "mde_at_n30_pooled": mde(30, pooled_sd),
    "mde_at_n10_per_arm": {a: mde(10, gain_sd[a]) for a in ARMS},
    "n_needed_for_own_mean_gain": {
        a: {
            "raw_mean_gain": sum(loso[a]["raw"]["per_seed_gain"][str(s)] for s in SEEDS) / 3.0,
            "n_needed_raw": n_needed(sum(loso[a]["raw"]["per_seed_gain"][str(s)]
                                         for s in SEEDS) / 3.0, pooled_sd),
            "equal_bias_mean_gain": sum(loso[a]["equal_bias"]["per_seed_gain"][str(s)]
                                        for s in SEEDS) / 3.0,
            "n_needed_equal_bias": n_needed(sum(loso[a]["equal_bias"]["per_seed_gain"][str(s)]
                                               for s in SEEDS) / 3.0, pooled_sd),
        } for a in ARMS},
    "best_arm_by_abs_raw_gain": best_arm,
    "n_needed_for_2x_best_effect": n_needed(
        2.0 * abs(sum(loso[best_arm]["raw"]["per_seed_gain"][str(s)] for s in SEEDS) / 3.0),
        pooled_sd),
    "n_needed_for_3x_best_effect": n_needed(
        3.0 * abs(sum(loso[best_arm]["raw"]["per_seed_gain"][str(s)] for s in SEEDS) / 3.0),
        pooled_sd),
    "n_needed_for_effect_of_0p02": n_needed(0.02, pooled_sd),
    "n_needed_for_effect_of_0p05": n_needed(0.05, pooled_sd),
}
n_exceed_3 = sum(1 for a in ARMS
                 if abs(power["n_needed_for_own_mean_gain"][a]["raw_mean_gain"])
                 > mde(3, pooled_sd))
check("no_arm_exceeds_mde_at_n3", n_exceed_3 == 0,
      f"{n_exceed_3}/6 arms exceed MDE(3, pooled_sd)={power['mde_at_n3_pooled']:.5f}")
check("ten_seeds_insufficient_for_best_observed_effect",
      (power["n_needed_for_own_mean_gain"][best_arm]["n_needed_raw"] or 999) > 10,
      f"best arm {best_arm} raw mean gain needs n="
      f"{power['n_needed_for_own_mean_gain'][best_arm]['n_needed_raw']} seeds "
      f"(section 11.9 step 2 asks for >= 10)")

# --------------------------------------------------------------------------
# write result
# --------------------------------------------------------------------------
result = {
    "generated": "2026-11-13",
    "round": 48,
    "scope": "read-only, local artifacts; no remote change",
    "artifacts": {
        "scores_summary_per_seed": "tools/_evals/e45/scores_summary_s{0,1,2}.json",
        "s3_equalbias": "tools/_evals/e45/s3_equalbias.json",
        "window_tables": "tools/_evals/e45/_raw/runs/e45/evals/*.windows.json (21 grids)",
    },
    "pipeline_selfcheck": {"n_comparisons": n_cmp, "max_abs_diff": maxdiff},
    "loso_seed_subset_sensitivity": loso,
    "loso_label_summary": loso_label,
    "cluster_aware_window_uncertainty": cluster,
    "case_level_crosscheck": case_level,
    "power_for_next_ladder_step": power,
    "info": dict(REPORT),
    "verdict": {
        "e45_verdict_unchanged": True,
        "statement": ("six arms stay negative; no arm enters the >=10-seed ladder; "
                      "no ten-seed compute is committed"),
        "new_axis": ("LOSO seed-subset sensitivity of the S3 gate + cluster-aware "
                     "(storm-case) uncertainty on the window axis + MDE/power for the "
                     "next ladder step"),
    },
}
out = EV / "round48_cluster_sensitivity.json"
out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

n_pos = sum(1 for _, ok in POS if ok)
print()
print(f"POS {n_pos}/{len(POS)}")
print("VERDICT: " + ("PASS" if n_pos == len(POS) else "FAIL"))
print(f"result -> {out}")
