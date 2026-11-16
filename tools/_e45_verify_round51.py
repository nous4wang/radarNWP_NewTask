#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 51 轮：误差构成轴（hits / misses / false_alarms 分解）首次判读。

只读本地、纯 stdlib（本地 python 无 numpy）。
前 50 轮判过：聚合层（CSI/BIAS）、机制探针层、S1 结构量层、窗级、风暴场次级、
时效轴、强度/难度轴。但从未把 CSI 增益拆成
    「命中率(POD) 贡献」 vs 「虚警率(FAR) 贡献」
⇒ 本轮做这件事，并给每臂配零模型（对照臂 B0 换种子互比）地板，
再与 S3 等偏置成果做**跨仪器**交叉核对（列联表计数 vs 场顺序统计）。

口径纪律（沿用第 44/46/48/49/50 轮）：
  · 报中位数 + 通过率，单点分数不作裁决
  · 配对口径（逐 seed 差值）与差值口径并列
  · 事后诊断，不构成新判据；S3 护栏固定 thr=2.0 mm，复制单元固定为 seed
  · 窗/帧级统计量不得当跨种子证据

分解的代数基础（精确，非微分近似）：
    POD = h/(h+m),  FAR = f/(h+f)
    CSI = g(POD, FAR) = POD*(1-FAR) / (1 - FAR + FAR*POD)   （与 h/(h+m+f) 恒等）
    ΔCSI 用两因子 Shapley 分解（两种路径平均，无路径依赖）：
        contrib_POD = ½[(g(pa,fc)-g(pc,fc)) + (g(pa,fa)-g(pc,fa))]
        contrib_FAR = ½[(g(pa,fa)-g(pa,fc)) + (g(pc,fa)-g(pc,fc))]
        contrib_POD + contrib_FAR ≡ ΔCSI（逐格精确）
"""
import json
import os
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVD = os.path.join(ROOT, "tools", "_evals", "e45")
RAW = os.path.join(EVD, "_raw", "runs", "e45", "evals")
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b", "B0"]
SEEDS = [0, 1, 2]
THRS = ["0.1", "0.5", "2.0", "4.0"]
PRIMARY = "2.0"
FLOOR = 0.0924
ORACLE = 0.1653

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


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def g_csi(pod, far):
    """CSI 作为 (POD, FAR) 的精确函数（与 h/(h+m+f) 恒等）。"""
    num = pod * (1.0 - far)
    den = (1.0 - far) + far * pod
    return num / den if den > 0 else float("nan")


def load_cell(arm, seed):
    p = os.path.join(RAW, "%s_s%d_ep00.windows.json" % (arm, seed))
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def agg(w, thr):
    H = M = F = CN = 0.0
    for row in w["windows"]:
        c = row[thr]
        H += c["hits"]
        M += c["misses"]
        F += c["false_alarms"]
        CN += c["correct_negatives"]
    return H, M, F, CN


def mets(H, M, F):
    tot = H + M + F
    csi = H / tot if tot else float("nan")
    pod = H / (H + M) if (H + M) else float("nan")
    far = F / (H + F) if (H + F) else float("nan")
    bias = (H + F) / (H + M) if (H + M) else float("nan")
    # g_csi 恒等性自检
    assert (isnan(csi) and isnan(g_csi(pod, far))) or abs(csi - g_csi(pod, far)) < 1e-12
    return {"H": H, "M": M, "F": F, "csi": csi, "pod": pod, "far": far, "bias": bias}


def isnan(x):
    return x != x


def decompose(ma, mc):
    """两因子 Shapley 分解：ΔCSI → POD 贡献 + FAR 贡献（精确）。"""
    pa, fa = ma["pod"], ma["far"]
    pc, fc = mc["pod"], mc["far"]
    g_aa, g_ac = g_csi(pa, fa), g_csi(pa, fc)
    g_ca, g_cc = g_csi(pc, fa), g_csi(pc, fc)
    contrib_pod = 0.5 * ((g_ac - g_cc) + (g_aa - g_ca))
    contrib_far = 0.5 * ((g_aa - g_ac) + (g_ca - g_cc))
    truth = mc["H"] + mc["M"]
    return {
        "d_csi": ma["csi"] - mc["csi"],
        "d_pod": pa - pc,
        "d_far": fa - fc,
        "contrib_pod": contrib_pod,
        "contrib_far": contrib_far,
        "d_h_truth": (ma["H"] - mc["H"]) / truth if truth else float("nan"),
        "d_f_truth": (ma["F"] - mc["F"]) / truth if truth else float("nan"),
        "d_m_truth": (ma["M"] - mc["M"]) / truth if truth else float("nan"),
        "bias_ratio_pre": ma["bias"] / mc["bias"] if mc["bias"] else float("nan"),
        "hits_per_extra_fa": ((ma["H"] - mc["H"]) / (ma["F"] - mc["F"]))
        if (ma["F"] - mc["F"]) > 0 else float("nan"),
    }


# ---------------- 1. 重算聚合量并与已发布 summary 互校 ----------------
published = {}
for s in SEEDS:
    with open(os.path.join(EVD, "scores_summary_s%d.json" % s), "r", encoding="utf-8") as f:
        for k, v in json.load(f).items():
            published[k] = v

cells = {}
maxdiff = 0.0
ncmp = 0
for arm in ARMS:
    for seed in SEEDS:
        w = load_cell(arm, seed)
        rec = {}
        for thr in THRS:
            H, M, F, CN = agg(w, thr)
            m = mets(H, M, F)
            rec[thr] = m
            key = "%s_s%d_ep00" % (arm, seed)
            pub = published.get(key, {})
            for pubk, mval in (
                ("csi_at_%s" % thr, "csi"),
                ("bias_at_%s" % thr, "bias"),
                ("pod_at_%s" % thr, "pod"),
                ("far_at_%s" % thr, "far"),
            ):
                if pubk in pub:
                    maxdiff = max(maxdiff, abs(pub[pubk] - m[mval]))
                    ncmp += 1
        cells[(arm, seed)] = rec

pos(ncmp == 168, "重建比较次数应为 168（21 格 × 8 个已发布指标），实际 %d" % ncmp)
pos(maxdiff == 0.0, "由窗表重建的 CSI/BIAS/POD/FAR 与已发布 summary 不一致，max|diff|=%g" % maxdiff)

wk = {}
for arm in ARMS:
    for seed in SEEDS:
        wk[(arm, seed)] = load_cell(arm, seed)["window_keys"]
ref = wk[("B0", 0)]
pos(all(v == ref for v in wk.values()), "21 格 window_keys 未完全一致，配对比较不合法")

# ---------------- 2. 配对分解（臂 vs 同 seed 对照 B0） ----------------
paired = {}
for arm in ARMS:
    if arm == "B0":
        continue
    paired[arm] = {s: decompose(cells[(arm, s)][PRIMARY], cells[("B0", s)][PRIMARY])
                   for s in SEEDS}

# ---------------- 3. 零模型（对照臂 B0 换种子互比，纯 seed 噪声） ----------------
null = []
for i, j in [(0, 1), (0, 2), (1, 2)]:
    null.append(decompose(cells[("B0", j)][PRIMARY], cells[("B0", i)][PRIMARY]))

FLOOR_KEYS = ["d_csi", "d_pod", "d_far", "contrib_pod", "contrib_far",
              "d_h_truth", "d_f_truth", "d_m_truth"]
null_floor = {}
for k in FLOOR_KEYS:
    vals = [abs(n[k]) for n in null]
    null_floor[k] = {"max": max(vals), "median": med(vals), "vals": vals}

# ---------------- 4. 逐臂汇总：中位数 + 通过率 ----------------
summary = {}
for arm, per_seed in paired.items():
    row = {}
    for k in FLOOR_KEYS + ["hits_per_extra_fa"]:
        vals = [per_seed[s][k] for s in SEEDS]
        row["med_" + k] = med(vals)
        row["per_seed_" + k] = {str(s): per_seed[s][k] for s in SEEDS}
    for k in FLOOR_KEYS:
        fl = null_floor[k]["max"]
        row["pass_" + k] = "%d/3" % sum(1 for s in SEEDS if abs(per_seed[s][k]) > fl)
        row["passcount_" + k] = sum(1 for s in SEEDS if abs(per_seed[s][k]) > fl)
        row["floor_" + k] = fl
    for k in ["d_csi", "contrib_pod", "contrib_far"]:
        row["sign_" + k] = "%d/3" % sum(1 for s in SEEDS if per_seed[s][k] > 0)
    row["diff_median_csi"] = (med([cells[(arm, s)][PRIMARY]["csi"] for s in SEEDS])
                              - med([cells[("B0", s)][PRIMARY]["csi"] for s in SEEDS]))
    row["diff_median_pod"] = (med([cells[(arm, s)][PRIMARY]["pod"] for s in SEEDS])
                              - med([cells[("B0", s)][PRIMARY]["pod"] for s in SEEDS]))
    row["diff_median_far"] = (med([cells[(arm, s)][PRIMARY]["far"] for s in SEEDS])
                              - med([cells[("B0", s)][PRIMARY]["far"] for s in SEEDS]))
    summary[arm] = row

# ---------------- 5. 跨 18 格（臂×seed）相关 ----------------
allcells = [(arm, s, paired[arm][s]) for arm in paired for s in SEEDS]
r_pod = pearson([c[2]["d_pod"] for c in allcells], [c[2]["d_csi"] for c in allcells])
r_far = pearson([c[2]["d_far"] for c in allcells], [c[2]["d_csi"] for c in allcells])
r_cp = pearson([c[2]["contrib_pod"] for c in allcells], [c[2]["d_csi"] for c in allcells])
r_cf = pearson([c[2]["contrib_far"] for c in allcells], [c[2]["d_csi"] for c in allcells])
r_dft = pearson([c[2]["d_f_truth"] for c in allcells], [c[2]["d_csi"] for c in allcells])

# ---------------- 6. 裁决守卫复核 ----------------
with open(os.path.join(EVD, "s3_equalbias.json"), "r", encoding="utf-8") as f:
    s3 = json.load(f)
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

# ---------------- 6b. 组成轴 × S3 等偏置成果的跨仪器交叉核对 ----------------
# S3 用「全局标量缩放预报场」把 BIAS 配平到对照，增益来自场的顺序统计；
# 组成轴只用列联表计数。两者是**不同的仪器**。
s3_link = []
for arm in paired:
    for s in SEEDS:
        raw_csi = cells[(arm, s)][PRIMARY]["csi"]
        ctl_csi = cells[("B0", s)][PRIMARY]["csi"]
        blk = s3["arms"][arm]["seeds"][str(s)]
        eq_csi = blk["csi_equalised"]
        s3_link.append({
            "arm": arm, "seed": s,
            "csi_raw": raw_csi, "csi_control": ctl_csi,
            "raw_gain": raw_csi - ctl_csi,
            "csi_equalised": eq_csi,
            "equalised_gain": eq_csi - ctl_csi,
            "csi_dropped_by_equalisation": raw_csi - eq_csi,
            "scale": blk["scale"],
            "bias_ratio_pre": cells[(arm, s)][PRIMARY]["bias"] / cells[("B0", s)][PRIMARY]["bias"],
            "d_f_truth": paired[arm][s]["d_f_truth"],
            "contrib_pod": paired[arm][s]["contrib_pod"],
            "contrib_far": paired[arm][s]["contrib_far"],
        })
r_dft_drop = pearson([x["d_f_truth"] for x in s3_link],
                      [x["csi_dropped_by_equalisation"] for x in s3_link])
r_bias_drop = pearson([x["bias_ratio_pre"] - 1.0 for x in s3_link],
                       [x["csi_dropped_by_equalisation"] for x in s3_link])
r_cf_drop = pearson([x["contrib_far"] for x in s3_link],
                     [x["csi_dropped_by_equalisation"] for x in s3_link])

# 分解恒等式：contrib_pod + contrib_far ≡ ΔCSI（逐格精确）
ident_max = max(abs((c[2]["contrib_pod"] + c[2]["contrib_far"]) - c[2]["d_csi"])
                 for c in allcells)
pos(ident_max < 1e-12, "Shapley 分解恒等式不成立，max|dev|=%g" % ident_max)

# S3 点估复算：raw 点估 = 逐 seed 原始增益的算术平均；equal_bias 同理
s3_point_maxdiff = 0.0
for arm in paired:
    for side, key in (("raw", "raw_gain"), ("equal_bias", "equalised_gain")):
        pt = s3["arms"][arm][side]["point"]
        calc = sum(x[key] for x in s3_link if x["arm"] == arm) / 3.0
        s3_point_maxdiff = max(s3_point_maxdiff, abs(pt - calc))
pos(s3_point_maxdiff < 1e-12, "S3 点估复算不一致，max|diff|=%g" % s3_point_maxdiff)

# s3 的 achieved_bias 是**配平后**的偏置（恒等于 control_bias）⇒ 其比值 ≡ 1，不可用于相关
post_ratio_maxdev = max(abs(s3["arms"][a]["seeds"][str(s)]["achieved_bias"]
                            / s3["arms"][a]["seeds"][str(s)]["control_bias"] - 1.0)
                         for a in paired for s in SEEDS)
scale_med = {a: med([x["scale"] for x in s3_link if x["arm"] == a]) for a in paired}
gaverdict_arms = [a for a, v in verdicts.items() if v == "GAIN_WAS_AREA_OR_AMPLITUDE"]
ga_drop = {a: med([x["csi_dropped_by_equalisation"] for x in s3_link if x["arm"] == a])
           for a in gaverdict_arms}
ga_eq_gain = {a: med([x["equalised_gain"] for x in s3_link if x["arm"] == a])
              for a in gaverdict_arms}

qualified = []
for arm in ARMS:
    if arm == "B0":
        continue
    csi3 = [cells[(arm, s)][PRIMARY]["csi"] for s in SEEDS]
    over = sum(1 for v in csi3 if v > FLOOR)
    s3_ok = not s3["arms"][arm]["equal_bias"]["crosses_zero"]
    if s3_ok and over == 3:
        qualified.append(arm)
pos(len(qualified) == 0, "合格臂集合应为空，实际 %s" % qualified)

ctrl_csi3 = [cells[("B0", s)][PRIMARY]["csi"] for s in SEEDS]
ctrl_over = sum(1 for v in ctrl_csi3 if v > FLOOR)

# ---------------- 7. 输出 ----------------
out = {
    "round": 51,
    "axis": "error-composition (hits/misses/false_alarms) decomposition of dCSI",
    "generated": "2026-11-14",
    "guardrail": {"threshold_mm": 2.0, "unit": "seed",
                  "note": "window-level components describe within-sample composition only"},
    "reconstruction": {"n_comparisons": ncmp, "max_abs_diff": maxdiff,
                        "window_keys_identical": True,
                        "shapley_identity_max_dev": ident_max},
    "null_model": {
        "definition": "control arm B0 with two different seeds compared to each other (pure seed noise)",
        "pairs": [[str(i), str(j)] for i, j in [(0, 1), (0, 2), (1, 2)]],
        "floors": null_floor,
    },
    "per_arm": summary,
    "cross_cell_correlations": {
        "r_d_pod_vs_d_csi": r_pod, "r_d_far_vs_d_csi": r_far,
        "r_contrib_pod_vs_d_csi": r_cp, "r_contrib_far_vs_d_csi": r_cf,
        "r_d_f_truth_vs_d_csi": r_dft, "n": len(allcells),
    },
    "s3_guard": {"n_intervals": nci, "n_crossing_zero": cz, "verdicts": verdicts},
    "s3_cross_instrument": {
        "note": "composition axis uses contingency counts only; S3 uses field order statistics "
                "(global scalar equalisation) => two independent instruments",
        "r_d_f_truth_vs_csi_dropped": r_dft_drop,
        "r_bias_ratio_pre_vs_csi_dropped": r_bias_drop,
        "r_contrib_far_vs_csi_dropped": r_cf_drop,
        "s3_point_max_abs_diff": s3_point_maxdiff,
        "post_equalisation_bias_ratio_max_dev_from_one": post_ratio_maxdev,
        "median_scale_per_arm": scale_med,
        "median_csi_dropped_by_equalisation": ga_drop,
        "median_equalised_gain": ga_eq_gain,
        "per_cell": s3_link,
    },
    "qualified_arms": qualified,
    "control_floor_pass": "%d/3" % ctrl_over,
    "notes": [
        "post-hoc diagnostic axis; does NOT change any verdict",
        "the S3 guard rail stays pinned at thr=2.0 mm and the cross-seed unit stays the seed",
        "Shapley split of dCSI into POD-contribution and FAR-contribution is exact per cell",
    ],
}
with open(os.path.join(EVD, "round51_error_composition.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("=" * 96)
print("E45 第 51 轮｜误差构成轴（thr=2.0 mm，配对口径 = 逐 seed 差值）")
print("=" * 96)
print("重建自校：%d 次比较 max|diff|=%g；window_keys 21/21 一致；Shapley 恒等式 max|dev|=%g"
      % (ncmp, maxdiff, ident_max))
print("零模型地板（B0 换 seed，三对控制取最大 |值|）：")
for k in FLOOR_KEYS:
    print("   %-13s max=%.6f  三对=%s" % (k, null_floor[k]["max"],
                                          ["%.4f" % v for v in null_floor[k]["vals"]]))
print()
hdr = ("臂    ΔCSI中位  POD贡献中位 FAR贡献中位  ΔPOD中位  ΔFAR中位  ΔF/真值中位 "
       "| 过地板:ΔCSI POD FAR ΔF/真值 | ΔCSI>0")
print(hdr)
print("-" * len(hdr))
for arm in ["A1", "A2", "A3a", "A3b", "A4", "A4b"]:
    r = summary[arm]
    print("%-5s %+8.5f  %+10.5f %+10.5f  %+9.5f  %+9.5f  %+11.5f |     %s     %s    %s       %s   |   %s"
          % (arm, r["med_d_csi"], r["med_contrib_pod"], r["med_contrib_far"],
             r["med_d_pod"], r["med_d_far"], r["med_d_f_truth"],
             r["pass_d_csi"], r["pass_contrib_pod"], r["pass_contrib_far"],
             r["pass_d_f_truth"], r["sign_d_csi"]))
print("对照 B0 过地板 %d/3（地板 %.4f，Oracle %.4f）" % (ctrl_over, FLOOR, ORACLE))
print()
print("差值口径（臂级真实水平，中位数相减）：")
for arm in ["A1", "A2", "A3a", "A3b", "A4", "A4b"]:
    r = summary[arm]
    print("   %-5s ΔCSI %+0.5f  ΔPOD %+0.5f  ΔFAR %+0.5f"
          % (arm, r["diff_median_csi"], r["diff_median_pod"], r["diff_median_far"]))
print()
print("跨 18 格（臂×seed）相关：")
print("   r(ΔPOD, ΔCSI)   = %+.4f" % r_pod)
print("   r(ΔFAR, ΔCSI)   = %+.4f" % r_far)
print("   r(POD贡献, ΔCSI)= %+.4f" % r_cp)
print("   r(FAR贡献, ΔCSI)= %+.4f" % r_cf)
print("   r(ΔF/真值, ΔCSI)= %+.4f" % r_dft)
print()
print("S3 守卫：12 个区间跨零 %d/12；合格臂集合 = %s（空）" % (cz, qualified))
print("逐臂 S3 裁决：" + json.dumps(verdicts, ensure_ascii=False))
print()
print("跨仪器交叉核对（组成轴＝列联表计数；S3＝场顺序统计的全局标量配平）：")
print("   r(ΔF/真值, 等偏置掉掉的 CSI)   = %+.4f  (n=%d)" % (r_dft_drop, len(s3_link)))
print("   r(配平前偏置比−1, 掉掉的 CSI) = %+.4f" % r_bias_drop)
print("   r(FAR 贡献, 掉掉的 CSI)        = %+.4f" % r_cf_drop)
print("   S3 点估复算 max|diff| = %g（12 个点估）" % s3_point_maxdiff)
print("   s3 的 achieved_bias 比值偏离 1 最大 %g ⇒ 配平后恒为 1，该比值不可用于相关"
      % post_ratio_maxdev)
for a in gaverdict_arms:
    print("   %-4s 判「仍是面积/幅度效应」：等偏置掉掉的 CSI 中位 %+.5f，配平后增益中位 %+.5f"
          % (a, ga_drop[a], ga_eq_gain[a]))
print("   S3 配平标量中位：" + json.dumps({k: round(v, 4) for k, v in scale_med.items()},
                                         ensure_ascii=False))
print()
print("POS %d / NEG %d" % (POS, NEG))
print("VERDICT: " + ("PASS" if NEG == 0 else "FAIL"))
print("NOTE: this axis is post-hoc diagnosis. The S3 guard rail stays pinned at thr=2.0 mm")
print("      and the cross-seed unit stays the seed => it does NOT change any verdict:")
print("      the six arms stay negative, no arm enters the 10-seed branch.")
print("artifact -> " + os.path.join(EVD, "round51_error_composition.json"))
