#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 52 轮：臂间增益的方差分解 + 区组随机化检验（复制单元＝seed）。

前 51 轮全部是「逐臂 vs 对照」的比较（配对增益、S3 等偏置、LOSO、MDE…），
从未问过一个更尖锐的问题：**六臂彼此之间是否可区分？**
若把 seed 当区组做一次合并检验（omnibus），答案直接回答
  (1)「E45 是否产生了任何东西」；
  (2)「A4 是否真的比 A2 好」（纪律上不得择优 ⇒ 需要可量化的「择优不成立」依据）。

做法：
  · 增益矩阵 g[arm][seed]（6 臂 × 3 seed），三套结果：
      raw        = CSI@2.0(臂,seed) − CSI@2.0(B0,seed)      （配对口径）
      equal_bias = 等偏置后 CSI(臂,seed) − CSI@2.0(B0,seed) （S3 侧，逐 seed）
      d_pod      = POD(臂,seed) − POD(B0,seed)
  · 两因素无重复方差分解：臂 / seed / 残差(=臂×seed 交互)
  · 区组随机化检验：H0「臂在 seed 内可交换」⇒ 在每个 seed 块内置换臂标签，
    重复 20000 次得 F 的零分布（seed 是区组、也是唯一复制单元 ⇒ 合法）
  · Kendall's W（臂排序在 3 个 seed 块上的一致性）+「每个 seed 的赢家」
    + 对 seed 做 bootstrap 后的「赢家分布」（⇒ 不得择优的量化依据）
  · 零模型：对照臂 B0 换种子互比（三对）
  · 检出力：把「臂间效应量」代入模拟，算出合并检验在各 seed 数下的检出力
    ⇒ 给「要不要投 10/30/60 seed」提供第二个（更便宜的）判据

口径纪律（沿用第 44/46/48/49/50/51 轮）：
  · 报中位数 + 通过率，单点分数不作裁决
  · 事后诊断，不构成新判据；S3 护栏固定 thr=2.0 mm，复制单元固定为 seed
  · 窗/帧/像素级统计量不得当跨种子证据
"""

import json
import os
import math
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVD = os.path.join(ROOT, "tools", "_evals", "e45")
RAW = os.path.join(EVD, "_raw", "runs", "e45", "evals")
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL = ARMS + ["B0"]
SEEDS = [0, 1, 2]
THRS = ["0.1", "0.5", "2.0", "4.0"]
PRIMARY = "2.0"
FLOOR = 0.0924
ORACLE = 0.1653
N_PERM = 20000
N_BOOT = 10000
RNG_SEED = 20261114

POS = 0
NEG = 0


def pos(cond, msg):
    global POS, NEG
    if cond:
        POS += 1
    else:
        NEG += 1
        print("  [NEG] " + msg)


def med(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(xs):
    return sum(xs) / float(len(xs))


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


# ---------------- F 分布（中心 F，纯 stdlib） ----------------
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3.0e-14, 1e-300
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
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
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
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def f_cdf(x, d1, d2):
    if x <= 0:
        return 0.0
    z = d1 * x / (d1 * x + d2)
    return betai(d1 * 0.5, d2 * 0.5, z)


def f_crit(d1, d2, alpha=0.05):
    lo, hi = 0.0, 500.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f_cdf(mid, d1, d2) < 1.0 - alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------- 两因素无重复方差分解（臂 × seed） ----------------
def anova(M):
    """M: 行＝臂、列＝seed。返回 F（臂效应 / 残差）与方差份额。"""
    k, b = len(M), len(M[0])
    flat = [v for row in M for v in row]
    grand = sum(flat) / float(k * b)
    arm_mean = [sum(row) / float(b) for row in M]
    seed_mean = [sum(M[i][s] for i in range(k)) / float(k) for s in range(b)]
    ss_arm = b * sum((m - grand) ** 2 for m in arm_mean)
    ss_seed = k * sum((m - grand) ** 2 for m in seed_mean)
    ss_res = sum((M[i][s] - arm_mean[i] - seed_mean[s] + grand) ** 2
                 for i in range(k) for s in range(b))
    ss_tot = sum((v - grand) ** 2 for v in flat)
    df_arm, df_seed, df_res = k - 1, b - 1, (k - 1) * (b - 1)
    ms_arm = ss_arm / df_arm
    ms_res = ss_res / df_res if df_res > 0 else float("nan")
    F = ms_arm / ms_res if (df_res > 0 and ms_res > 0) else float("inf")
    return {
        "F": F, "df_arm": df_arm, "df_seed": df_seed, "df_res": df_res,
        "ss_arm": ss_arm, "ss_seed": ss_seed, "ss_res": ss_res, "ss_tot": ss_tot,
        "share_arm": ss_arm / ss_tot if ss_tot > 0 else float("nan"),
        "share_seed": ss_seed / ss_tot if ss_tot > 0 else float("nan"),
        "share_res": ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "arm_mean": arm_mean, "seed_mean": seed_mean, "grand": grand,
        "resid_sd": math.sqrt(ss_res / df_res) if df_res > 0 else float("nan"),
    }


def kendall_w(M):
    """臂排序在 seed 块上的一致性（Kendall's W），0＝完全不一致，1＝完全一致。"""
    k, b = len(M), len(M[0])
    ranks = []
    for s in range(b):
        col = [M[i][s] for i in range(k)]
        order = sorted(range(k), key=lambda i: col[i])
        rk = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j + 1 < k and col[order[j + 1]] == col[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        ranks.append(rk)
    R = [sum(ranks[s][i] for s in range(b)) for i in range(k)]
    mean_R = k * (b + 1) / 2.0
    S = sum((v - mean_R) ** 2 for v in R)
    W = 12.0 * S / (k * (k * k - 1) * b)
    return W, R


# ---------------- 载入并重建（与已发布 summary 互校） ----------------
published = {}
for s in SEEDS:
    with open(os.path.join(EVD, "scores_summary_s%d.json" % s), "r", encoding="utf-8") as f:
        for kk, vv in json.load(f).items():
            published[kk] = vv

cells = {}
ncmp = 0
maxdiff = 0.0
for arm in ALL:
    for seed in SEEDS:
        with open(os.path.join(RAW, "%s_s%d_ep00.windows.json" % (arm, seed)),
                  "r", encoding="utf-8") as f:
            w = json.load(f)
        rec = {}
        for thr in THRS:
            H = M = F = CN = 0.0
            for row in w["windows"]:
                c = row[thr]
                H += c["hits"]
                M += c["misses"]
                F += c["false_alarms"]
                CN += c["correct_negatives"]
            pod = H / (H + M) if (H + M) else float("nan")
            far = F / (H + F) if (H + F) else float("nan")
            rec[thr] = {
                "csi": H / (H + M + F) if (H + M + F) else float("nan"),
                "pod": pod, "far": far,
                "bias": (H + F) / (H + M) if (H + M) else float("nan"),
            }
            pub = published.get("%s_s%d_ep00" % (arm, seed), {})
            for pubk, mv in (("csi_at_%s" % thr, "csi"), ("bias_at_%s" % thr, "bias"),
                             ("pod_at_%s" % thr, "pod"), ("far_at_%s" % thr, "far")):
                if pubk in pub:
                    maxdiff = max(maxdiff, abs(pub[pubk] - rec[thr][mv]))
                    ncmp += 1
        cells[(arm, seed)] = rec

pos(ncmp == 168, "重建比较次数应为 168（21 格 × 8 个已发布指标），实际 %d" % ncmp)
pos(maxdiff == 0.0, "由窗表重建的 CSI/BIAS/POD/FAR 与已发布 summary 不一致，max|diff|=%g" % maxdiff)

wk = {}
for arm in ALL:
    for seed in SEEDS:
        with open(os.path.join(RAW, "%s_s%d_ep00.windows.json" % (arm, seed)),
                  "r", encoding="utf-8") as f:
            wk[(arm, seed)] = json.load(f)["window_keys"]
ref = wk[("B0", 0)]
pos(all(v == ref for v in wk.values()), "21 格 window_keys 未完全一致，配对比较不合法")

# ---------------- S3 等偏置逐 seed 增益 ----------------
with open(os.path.join(EVD, "s3_equalbias.json"), "r", encoding="utf-8") as f:
    s3 = json.load(f)

# ---------------- 三套增益矩阵 ----------------
M_raw, M_eq, M_pod = [], [], []
per_seed_gain = {}
for arm in ARMS:
    rrow, erow, prow = [], [], []
    for seed in SEEDS:
        ctl = cells[("B0", seed)][PRIMARY]["csi"]
        raw = cells[(arm, seed)][PRIMARY]["csi"] - ctl
        eq = s3["arms"][arm]["seeds"][str(seed)]["csi_equalised"] - ctl
        pod = cells[(arm, seed)][PRIMARY]["pod"] - cells[("B0", seed)][PRIMARY]["pod"]
        rrow.append(raw)
        erow.append(eq)
        prow.append(pod)
        per_seed_gain[(arm, seed)] = {"raw": raw, "equal_bias": eq, "d_pod": pod}
    M_raw.append(rrow)
    M_eq.append(erow)
    M_pod.append(prow)

# ---------------- 裁决守卫复核 ----------------
cz = 0
nci = 0
verdicts = {}
for arm, blk in s3["arms"].items():
    verdicts[arm] = blk["s3_verdict"]
    for side in ("raw", "equal_bias"):
        nci += 1
        if blk[side]["crosses_zero"]:
            cz += 1
pos(nci == 12 and cz == 12, "S3 的 12 个区间未全部跨零（跨零 %d/12）" % cz)

s3_point_maxdiff = 0.0
for i, arm in enumerate(ARMS):
    for side, Mx in (("raw", M_raw), ("equal_bias", M_eq)):
        calc = sum(Mx[i]) / 3.0
        s3_point_maxdiff = max(s3_point_maxdiff, abs(s3["arms"][arm][side]["point"] - calc))
pos(s3_point_maxdiff < 1e-12, "S3 点估复算不一致，max|diff|=%g" % s3_point_maxdiff)

qualified = []
for i, arm in enumerate(ARMS):
    csi3 = [cells[(arm, s)][PRIMARY]["csi"] for s in SEEDS]
    over = sum(1 for v in csi3 if v > FLOOR)
    s3_ok = not s3["arms"][arm]["equal_bias"]["crosses_zero"]
    if s3_ok and over == 3:
        qualified.append(arm)
pos(len(qualified) == 0, "合格臂集合应为空，实际 %s" % qualified)

ctrl_csi3 = [cells[("B0", s)][PRIMARY]["csi"] for s in SEEDS]
ctrl_over = sum(1 for v in ctrl_csi3 if v > FLOOR)

# ---------------- 零模型：对照臂 B0 换种子互比（按量纲分别给地板） ----------------
null_pairs = []
for i, j in [(0, 1), (0, 2), (1, 2)]:
    null_pairs.append({
        "pair": [str(i), str(j)],
        "d_csi": cells[("B0", j)][PRIMARY]["csi"] - cells[("B0", i)][PRIMARY]["csi"],
        "d_pod": cells[("B0", j)][PRIMARY]["pod"] - cells[("B0", i)][PRIMARY]["pod"],
    })
# 每个结果集各有自己的零模型地板（量纲不同不可混用）
null_floor = {
    "raw paired dCSI@2.0": max(abs(p["d_csi"]) for p in null_pairs),
    "equal-bias dCSI@2.0 (S3 side)": max(abs(p["d_csi"]) for p in null_pairs),
    "paired dPOD@2.0": max(abs(p["d_pod"]) for p in null_pairs),
}
null_floor_csi = null_floor["raw paired dCSI@2.0"]
null_floor_pod = null_floor["paired dPOD@2.0"]

# ---------------- 合并检验（区组随机化） ----------------
def omnibus(M, label):
    k, b = len(M), len(M[0])
    obs = anova(M)
    d1, d2 = obs["df_arm"], obs["df_res"]
    p_analytic = 1.0 - f_cdf(obs["F"], d1, d2)
    crit = f_crit(d1, d2, 0.05)
    rng = random.Random(RNG_SEED)
    null = []
    for _ in range(N_PERM):
        newM = [[0.0] * b for _ in range(k)]
        for s in range(b):
            vals = [M[i][s] for i in range(k)]
            idx = list(range(k))
            rng.shuffle(idx)
            for j2 in range(k):
                newM[j2][s] = vals[idx[j2]]
        null.append(anova(newM)["F"])
    ge = sum(1 for v in null if v >= obs["F"])
    p_perm = (ge + 1.0) / (N_PERM + 1.0)
    W, R = kendall_w(M)
    winners = []
    for s in range(b):
        best = max(range(k), key=lambda i: M[i][s])
        winners.append(ARMS[best])
    rng2 = random.Random(RNG_SEED + 1)
    boot = dict((a, 0) for a in ARMS)
    for _ in range(N_BOOT):
        idx = [rng2.randrange(b) for _ in range(b)]
        meds = [med([M[i][idx[j]] for j in range(b)]) for i in range(k)]
        best = max(range(k), key=lambda i: meds[i])
        boot[ARMS[best]] += 1
    boot = dict((a, boot[a] / float(N_BOOT)) for a in ARMS)
    top = sorted(boot.items(), key=lambda kv: -kv[1])
    return {
        "label": label,
        "F_observed": obs["F"], "df_arm": d1, "df_seed": obs["df_seed"], "df_res": d2,
        "F_crit_05": crit,
        "p_analytic_centralF": p_analytic,
        "p_permutation_blocked": p_perm,
        "n_perm": N_PERM,
        "null_F_median": med(null), "null_F_q95": sorted(null)[int(0.95 * (len(null) - 1))],
        "variance_share_arm": obs["share_arm"],
        "variance_share_seed": obs["share_seed"],
        "variance_share_resid": obs["share_res"],
        "arm_means": dict((ARMS[i], obs["arm_mean"][i]) for i in range(k)),
        "arm_mean_spread": max(obs["arm_mean"]) - min(obs["arm_mean"]),
        "seed_means": obs["seed_mean"],
        "resid_sd": obs["resid_sd"],
        "kendall_W": W,
        "rank_sums": dict((ARMS[i], R[i]) for i in range(k)),
        "winner_per_seed": winners,
        "distinct_winners": len(set(winners)),
        "bootstrap_winner_prob": boot,
        "bootstrap_top1": top[0][0], "bootstrap_top1_prob": top[0][1],
        "per_arm_median": dict((ARMS[i], med([M[i][s] for s in range(b)])) for i in range(k)),
        "per_arm_per_seed": dict((ARMS[i], dict((str(s), M[i][s]) for s in range(b))) for i in range(k)),
        "_obs": obs,
    }


res_raw = omnibus(M_raw, "raw paired dCSI@2.0")
res_eq = omnibus(M_eq, "equal-bias dCSI@2.0 (S3 side)")
res_pod = omnibus(M_pod, "paired dPOD@2.0")

for r in (res_raw, res_eq, res_pod):
    pos(0.0 <= r["kendall_W"] <= 1.0, "%s 的 Kendall W 越界：%g" % (r["label"], r["kendall_W"]))
    pos(0.0 < r["p_permutation_blocked"] <= 1.0,
        "%s 的置换 p 越界：%g" % (r["label"], r["p_permutation_blocked"]))
    pos(r["df_res"] == 10, "%s 的残差自由度应为 10，实际 %d" % (r["label"], r["df_res"]))
    r["null_floor"] = null_floor[r["label"]]
    r["spread_over_floor"] = (r["arm_mean_spread"] / null_floor[r["label"]]
                              if null_floor[r["label"]] > 0 else float("nan"))
    # CSI 轴上的主张：六臂之间的分离度必须小于「把对照臂换个种子」就能得到的分离度
    if r["label"] != "paired dPOD@2.0":
        pos(r["arm_mean_spread"] < null_floor[r["label"]],
            "%s：六臂臂均值跨度 %.5f 不小于零模型地板 %.5f ⇒ 六臂之间的分离弱于换种子"
            % (r["label"], r["arm_mean_spread"], null_floor[r["label"]]))
    else:
        # POD 轴单独记事实（不预设结论）：跨度 / 地板 的比值如实报告
        pos(r["spread_over_floor"] > 0, "dPOD 轴的跨度/地板比值无效：%g" % r["spread_over_floor"])

# ---------------- 逐臂：中位数 + 通过率 ----------------
per_arm = {}
for i, arm in enumerate(ARMS):
    raw3 = M_raw[i]
    eq3 = M_eq[i]
    per_arm[arm] = {
        "median_raw": med(raw3),
        "median_equal_bias": med(eq3),
        "sign_raw": "%d+/3" % sum(1 for v in raw3 if v > 0),
        "sign_equal_bias": "%d+/3" % sum(1 for v in eq3 if v > 0),
        "pass_raw_vs_null_floor": "%d/3" % sum(1 for v in raw3 if abs(v) > null_floor_csi),
        "per_seed_raw": dict((str(s), raw3[s]) for s in SEEDS),
        "per_seed_equal_bias": dict((str(s), eq3[s]) for s in SEEDS),
        "arm_mean_raw": res_raw["arm_means"][arm],
        "bootstrap_winner_prob": res_raw["bootstrap_winner_prob"][arm],
    }

# ---------------- 检出力：合并检验要多少 seed ----------------
power_grid = [3, 5, 10, 15, 20, 30, 45, 60]
scales = [0.5, 1.0, 2.0, 3.0]
sigma = res_raw["resid_sd"]
grand = res_raw["_obs"]["grand"]
alpha_vec = [res_raw["_obs"]["arm_mean"][i] - grand for i in range(len(ARMS))]
power = {}
for sc in scales:
    row = {}
    for b in power_grid:
        d1, d2 = len(ARMS) - 1, (len(ARMS) - 1) * (b - 1)
        crit = f_crit(d1, d2, 0.05)
        rng3 = random.Random(RNG_SEED + 1000 + b * 13 + int(sc * 10))
        hits = 0
        nsim = 600
        for _ in range(nsim):
            Mm = [[alpha_vec[i] * sc + rng3.gauss(0.0, sigma) for _s in range(b)]
                  for i in range(len(ARMS))]
            if anova(Mm)["F"] > crit:
                hits += 1
        row[str(b)] = hits / float(nsim)
    need = None
    for b in power_grid:
        if row[str(b)] >= 0.80:
            need = b
            break
    power["scale_x%.1f" % sc] = {"by_n_seeds": row, "n_seeds_for_80pct_power": need}
    if need is None:
        power["scale_x%.1f" % sc]["n_seeds_for_80pct_power"] = None

# ---------------- 输出 ----------------
out = {
    "round": 52,
    "axis": "between-arm omnibus: variance decomposition + blocked randomisation test (seed = block & replication unit)",
    "generated": "2026-11-14",
    "guardrail": {"threshold_mm": 2.0, "unit": "seed",
                  "note": "post-hoc diagnostic axis; does NOT change any verdict"},
    "reconstruction": {"n_comparisons": ncmp, "max_abs_diff": maxdiff,
                        "window_keys_identical": True,
                        "s3_point_max_abs_diff": s3_point_maxdiff},
    "null_model": {"definition": "control arm B0 with two different seeds compared to each other",
                   "pairs": null_pairs, "floor_abs_d_csi": null_floor_csi,
                   "floor_abs_d_pod": null_floor_pod,
                   "floors_by_result_set": null_floor},
    "omnibus": {"raw": res_raw, "equal_bias": res_eq, "d_pod": res_pod},
    "per_arm": per_arm,
    "power": {"note": "Monte-Carlo, 600 sims per cell (SE at p=0.8 is about 0.016); "
                       "simulated data = observed arm-mean structure x scale + N(0, resid_sd); "
                       "central F critical value computed from the regularised incomplete beta",
              "resid_sd_used": sigma, "grid": power},
    "s3_guard": {"n_intervals": nci, "n_crossing_zero": cz, "verdicts": verdicts},
    "qualified_arms": qualified,
    "control_floor_pass": "%d/3" % ctrl_over,
    "notes": [
        "post-hoc diagnostic axis; does NOT change any verdict",
        "seed is the only replication unit; seed is also the blocking factor in the permutation test",
        "medians and pass rates are reported; single-point scores are not used for any decision",
    ],
}
with open(os.path.join(EVD, "round52_omnibus_arm_effect.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("=" * 98)
print("E45 第 52 轮｜臂间合并检验（seed 为区组与唯一复制单元）")
print("=" * 98)
print("重建自校：%d 次比较 max|diff|=%g；window_keys 21/21 一致；S3 点估复算 max|diff|=%g"
      % (ncmp, maxdiff, s3_point_maxdiff))
print("零模型地板（B0 换 seed，三对取最大 |ΔCSI|）：%.6f  三对=%s"
      % (null_floor_csi, ["%.5f" % p["d_csi"] for p in null_pairs]))
print()
for r in (res_raw, res_eq, res_pod):
    print("── 结果集：%s" % r["label"])
    print("   F=%.4f（df %d,%d）｜临界值 F(0.05)=%.4f｜解析 p=%.4f｜区组置换 p=%.4f"
          % (r["F_observed"], r["df_arm"], r["df_res"], r["F_crit_05"],
             r["p_analytic_centralF"], r["p_permutation_blocked"]))
    print("   零分布 F 中位=%.4f、q95=%.4f｜方差份额 臂=%.3f seed=%.3f 残差=%.3f"
          % (r["null_F_median"], r["null_F_q95"], r["variance_share_arm"],
             r["variance_share_seed"], r["variance_share_resid"]))
    print("   臂均值跨度=%.5f｜该量纲的零模型地板=%.5f（比值 %.2f）｜Kendall W=%.4f"
          "｜逐 seed 赢家=%s（不同赢家 %d 个）"
          % (r["arm_mean_spread"], r["null_floor"], r["spread_over_floor"],
             r["kendall_W"], r["winner_per_seed"], r["distinct_winners"]))
    print("   bootstrap 赢家概率（对 seed 重抽）："
          + json.dumps({a: round(v, 3) for a, v in
                        sorted(r["bootstrap_winner_prob"].items(), key=lambda kv: -kv[1])},
                       ensure_ascii=False))
    print()
print("逐臂（配对口径，报中位数 + 通过率）：")
hdr = "臂     原始增益中位  配平后增益中位  原始符号  配平后符号  过零模型地板"
print(hdr)
print("-" * len(hdr))
for arm in ARMS:
    r = per_arm[arm]
    print("%-6s %+11.5f  %+13.5f  %8s  %9s  %12s"
          % (arm, r["median_raw"], r["median_equal_bias"], r["sign_raw"],
             r["sign_equal_bias"], r["pass_raw_vs_null_floor"]))
print("对照 B0 过地板 %d/3（地板 %.4f，Oracle %.4f）" % (ctrl_over, FLOOR, ORACLE))
print()
print("合并检验的检出力（Monte-Carlo 600 次/格，resid_sd=%.5f）：" % sigma)
for sc in scales:
    row = power["scale_x%.1f" % sc]
    cells_txt = "  ".join("n=%d:%.2f" % (b, row["by_n_seeds"][str(b)]) for b in power_grid)
    print("   臂间效应量 ×%.1f：%s ｜ 达 80%% 检出力需 n=%s"
          % (sc, cells_txt, row["n_seeds_for_80pct_power"]))
print()
print("S3 守卫：12 个区间跨零 %d/12；合格臂集合 = %s（空）" % (cz, qualified))
print("POS %d / NEG %d" % (POS, NEG))
print("VERDICT: " + ("PASS" if NEG == 0 else "FAIL"))
print("NOTE: this axis is post-hoc diagnosis; the S3 guard rail stays pinned at thr=2.0 mm")
print("      and the cross-seed unit stays the seed => it does NOT change any verdict:")
print("      the six arms stay negative, no arm enters the 10-seed branch.")
print("artifact -> " + os.path.join(EVD, "round52_omnibus_arm_effect.json"))
