# -*- coding: utf-8 -*-
"""E45 第 25 轮独立复核（不 import 任何旧脚本，全部从原始产物重取）。

口径来源：
  - 地板 FLOOR = 0.0924、Oracle ORACLE = 0.1653（seed-0 口径，E45 判据冻结值）
  - 门：A1/A2  corr(mask,amp) >= 0 且 mask_at_peak > 0.5
  - 拓扑：A3a/A3b grad_top1pct_share < 0.5 且 field_max_mm <= 30
  - 量通道：A4/A4b scale_cv >= 0.05 且 scale_truth_corr >= 0.3
断言分三通道：POS（应为真）/ NEG（违规事件不应发生）/ PAT（逐 seed 期望模式）。
"""
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "tools", "_evals", "e45")
FLOOR = 0.0924
ORACLE = 0.1653
ARMS7 = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
S3ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
EXPECT_S3_VERDICT = {
    "A1": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A4": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A2": "NO_GAIN_TO_BEGIN_WITH",
    "A4b": "NO_GAIN_TO_BEGIN_WITH",
}

pos_ok, pos_bad = [], []
neg_ok, neg_bad = [], []
pat_ok, pat_bad = [], []
info = []


def check(name, cond, detail=""):
    if cond:
        pos_ok.append(name)
    else:
        pos_bad.append((name, detail))
    return bool(cond)


def neg(name, violation, detail=""):
    """violation=True 表示违规事件发生（应判 FAIL）。"""
    if violation:
        neg_bad.append((name, detail))
    else:
        neg_ok.append(name)
    return not violation


def pat(name, got, expected):
    if got == expected:
        pat_ok.append(name)
    else:
        pat_bad.append((name, got, expected))
    return got == expected


def load(fn):
    with open(os.path.join(EV, fn), encoding="utf-8") as f:
        return json.load(f)


def is_nan(x):
    return x is None or (isinstance(x, float) and x != x)


# ---------------- 1. 产物存在性 ----------------
need = ["scores_summary.json", "scores_all.json", "struct_metrics_all.json", "s3_equalbias.json",
        "diag_probe_s0.json", "diag_probe_s1.json", "diag_probe_s2.json",
        "scores_summary_s0.json", "scores_summary_s1.json", "scores_summary_s2.json",
        "struct_metrics_s0.json", "struct_metrics_s1.json", "struct_metrics_s2.json"]
for f in need:
    check("artifact_exists:%s" % f, os.path.exists(os.path.join(EV, f)))
if pos_bad:
    print("FATAL: 产物缺失，终止")
    for x in pos_bad:
        print("  MISS", x)
    sys.exit(2)

ss = load("scores_summary.json")
sa = load("scores_all.json")
sm = load("struct_metrics_all.json")
s3 = load("s3_equalbias.json")
dps = [load("diag_probe_s%d.json" % s) for s in (0, 1, 2)]

# ---------------- 2. scores_summary 结构与中位数重算 ----------------
check("ss_top_is_arm_keys", set(ss.keys()) == set(ARMS7), str(sorted(ss.keys())))
per_seed = {}
for arm in ARMS7:
    rec = ss[arm]
    check("ss_seeds:%s" % arm, sorted(rec.get("seeds", [])) == [0, 1, 2], str(rec.get("seeds")))
    check("ss_n_seeds:%s" % arm, rec.get("n_seeds") == 3, str(rec.get("n_seeds")))
    csi = rec.get("csi_at_2.0", {})
    bias = rec.get("bias_at_2.0", {})
    check("ss_csi3:%s" % arm, sorted(csi.keys()) == ["s0", "s1", "s2"], str(sorted(csi.keys())))
    check("ss_bias3:%s" % arm, sorted(bias.keys()) == ["s0", "s1", "s2"], str(sorted(bias.keys())))
    vals = [csi["s0"], csi["s1"], csi["s2"]]
    bvals = [bias["s0"], bias["s1"], bias["s2"]]
    per_seed[arm] = {"csi": vals, "bias": bvals}
    med = statistics.median(vals)
    medb = statistics.median(bvals)
    check("ss_median_csi:%s" % arm, abs(med - rec.get("median_csi_at_2.0", 1e9)) < 1e-9,
          "recomp=%r stored=%r" % (med, rec.get("median_csi_at_2.0")))
    check("ss_median_bias:%s" % arm, abs(medb - rec.get("median_bias_at_2.0", 1e9)) < 1e-9,
          "recomp=%r stored=%r" % (medb, rec.get("median_bias_at_2.0")))
    npass = sum(1 for v in vals if v >= FLOOR)
    check("ss_pass_rate:%s" % arm, rec.get("pass_rate_over_floor") == "%d/3" % npass,
          "recomp=%d/3 stored=%r" % (npass, rec.get("pass_rate_over_floor")))
    spread = max(vals) / min(vals)
    check("ss_spread:%s" % arm, abs(spread - rec.get("seed_spread_csi_at_2.0", -1)) < 1e-6,
          "recomp=%r stored=%r" % (spread, rec.get("seed_spread_csi_at_2.0")))
    norc = sum(1 for v in vals if v > ORACLE)
    info.append("ARM %s median_csi=%.6f range=[%.6f,%.6f] median_bias=%.4f floor_pass=%d/3 over_oracle=%d/3 spread=%.3fx"
                % (arm, med, min(vals), max(vals), medb, npass, norc, spread))

# ---------------- 3. scores_all 与 summary 互校（21 条） ----------------
# scores_all 顶层直接是 "<臂>_s<seed>_ep00" 键（无 tags 子层，那是 struct_metrics_all 的结构）
sa_tags = sa if isinstance(sa, dict) else None
cmp_n = 0
if isinstance(sa_tags, dict):
    for arm in ARMS7:
        for i, s in enumerate((0, 1, 2)):
            key = "%s_s%d_ep00" % (arm, s)
            rec = sa_tags.get(key)
            if rec is None:
                continue
            cmp_n += 1
            c = rec.get("csi_at_2.0")
            b = rec.get("bias_at_2.0")
            check("sa_vs_ss_csi:%s" % key, c is not None and abs(c - per_seed[arm]["csi"][i]) < 1e-9,
                  "sa=%r ss=%r" % (c, per_seed[arm]["csi"][i]))
            check("sa_vs_ss_bias:%s" % key, b is not None and abs(b - per_seed[arm]["bias"][i]) < 1e-9,
                  "sa=%r ss=%r" % (b, per_seed[arm]["bias"][i]))
    check("sa_record_count_21", cmp_n == 21, "compared=%d" % cmp_n)
else:
    check("sa_tags_present", False, "scores_all 顶层=%s" % list(sa.keys())[:8])

# ---------------- 4. S3 等偏置对照独立复核 ----------------
check("s3_threshold_2mm", abs(float(s3.get("threshold_mm", -1)) - 2.0) < 1e-9, str(s3.get("threshold_mm")))
check("s3_control_B0", s3.get("control") == "B0", str(s3.get("control")))
check("s3_nboot_10000", s3.get("n_boot") == 10000, str(s3.get("n_boot")))
check("s3_arms_set", set(s3.get("arms", {}).keys()) == set(S3ARMS), str(sorted(s3.get("arms", {}).keys())))
for arm in S3ARMS:
    a = s3["arms"][arm]
    for blk in ("raw", "equal_bias"):
        b = a.get(blk, {})
        check("s3_%s_%s_nboot" % (arm, blk), b.get("n_boot") == 10000, str(b.get("n_boot")))
        check("s3_%s_%s_nseed" % (arm, blk), b.get("n_seeds") == 3, str(b.get("n_seeds")))
        check("s3_%s_%s_nwin" % (arm, blk), b.get("n_windows") == 538, str(b.get("n_windows")))
        lo, hi = b.get("ci_lo"), b.get("ci_hi")
        recomputed = (lo is not None and hi is not None and lo <= 0.0 <= hi)
        check("s3_%s_%s_crosses_zero_recompute" % (arm, blk), recomputed == b.get("crosses_zero"),
              "recomp=%r stored=%r ci=[%r,%r]" % (recomputed, b.get("crosses_zero"), lo, hi))
        neg("s3_%s_%s_ci_crosses_zero" % (arm, blk), not recomputed,
            "ci=[%r,%r] 不跨零" % (lo, hi))
    check("s3_verdict:%s" % arm, a.get("s3_verdict") == EXPECT_S3_VERDICT[arm],
          "got=%r expect=%r" % (a.get("s3_verdict"), EXPECT_S3_VERDICT[arm]))

# ---------------- 5. 机制读数（逐臂逐 seed，从三份原始探针重取） ----------------
def ro(arm, key, seed_idx):
    return dps[seed_idx]["arms"][arm]["readouts"].get(key)


gate = {}
for arm in ("A1", "A2"):
    cma = [ro(arm, "corr_mask_amp", i) for i in range(3)]
    mpeak = [ro(arm, "mask_at_peak", i) for i in range(3)]
    check("gate_corr_nonneg:%s" % arm, all(v is not None and v >= 0 for v in cma), str(cma))
    check("gate_mask_at_peak:%s" % arm, all(v is not None and v > 0.5 for v in mpeak), str(mpeak))
    neg("gate_corr_nonneg_viol:%s" % arm, any(v is None or v < 0 for v in cma), str(cma))
    neg("gate_mask_at_peak_viol:%s" % arm, any(v is None or v <= 0.5 for v in mpeak), str(mpeak))
    gate[arm] = (cma, mpeak)
    info.append("ARM %s corr_mask_amp=%s mask_at_peak=%s" % (arm, cma, mpeak))

topo = {}
for arm in ("A3a", "A3b"):
    share = [ro(arm, "grad_top1pct_share", i) for i in range(3)]
    fmax = [ro(arm, "field_max_mm", i) for i in range(3)]
    pass_pat = [(v is not None and v < 0.5) and (f is not None and f <= 30.0) for v, f in zip(share, fmax)]
    pat("topo_pass_pattern:%s" % arm, pass_pat, [False, False, False])
    neg("topo_no_seed_passes:%s" % arm, any(pass_pat), "pass=%s share=%s fmax=%s" % (pass_pat, share, fmax))
    check("topo_fmax_gt30:%s" % arm, all(f is not None and f > 30.0 for f in fmax), str(fmax))
    topo[arm] = (share, fmax, pass_pat)
    info.append("ARM %s grad_top1pct_share=%s field_max_mm=%s topo_pass=%s" % (arm, share, fmax, pass_pat))

scale = {}
for arm in ("A4", "A4b"):
    cv = [ro(arm, "scale_cv", i) for i in range(3)]
    tc = [ro(arm, "scale_truth_corr", i) for i in range(3)]
    isc = [ro(arm, "scale_is_constant", i) for i in range(3)]
    mot = [ro(arm, "mean_out_equals_T", i) for i in range(3)]
    check("scale_cv_ge_0p05:%s" % arm, all(v is not None and v >= 0.05 for v in cv), str(cv))
    # scale_truth_corr 逐 seed 期望模式：A4 三 seed 全过；A4b 第 3 个种子已知退化（0.2789 < 0.3）
    tc_pass = [(v is not None and v >= 0.3) for v in tc]
    pat("scale_truth_corr_pattern:%s" % arm, tc_pass,
        [True, True, True] if arm == "A4" else [True, True, False])
    if arm == "A4":
        neg("scale_truth_corr_viol:A4", any(not p for p in tc_pass), str(tc))
    pat("scale_is_constant_pattern:%s" % arm, isc, [False, False, False] if arm == "A4" else [False, False, True])
    pat("mean_out_equals_T_pattern:%s" % arm, mot, [True, False, True] if arm == "A4" else [False, False, False])
    scale[arm] = (cv, tc, isc, mot)
    info.append("ARM %s scale_cv=%s scale_truth_corr=%s is_constant=%s mean_out_equals_T=%s"
                % (arm, cv, tc, isc, mot))

for arm in ("A1", "A2"):
    mr = [ro(arm, "maxratio_as_trained", i) for i in range(3)]
    neg("maxratio_as_trained_all_nan:%s" % arm, any(not is_nan(v) for v in mr), str(mr))

# ---------------- 6. 晋级合取判定 ----------------
qual = []
for arm in S3ARMS:
    a = s3["arms"][arm]
    raw_cross = a["raw"]["ci_lo"] <= 0.0 <= a["raw"]["ci_hi"]
    eq_cross = a["equal_bias"]["ci_lo"] <= 0.0 <= a["equal_bias"]["ci_hi"]
    all_floor = ss[arm].get("pass_rate_over_floor") == "3/3"
    if (not raw_cross) and (not eq_cross) and all_floor:
        qual.append(arm)
check("promotion_qualified_set_empty", qual == [], "qualified=%s" % qual)
info.append("PROMOTION qualified arms = %s (空集)" % (qual if qual else "[]"))

# ---------------- 7. 文档状态与时效性 ----------------
def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return f.read().splitlines()

master = read_lines(os.path.join(ROOT, "docs", "MASTER.md"))
index = read_lines(os.path.join(ROOT, "docs", "experiments", "INDEX.md"))
handover = read_lines(os.path.join(os.path.dirname(ROOT), ".workbuddy", "memory", "HANDOVER_E45.md"))

ver_rows = [l for l in master if l.lstrip().startswith("|") and "最后更新" in l]
check("master_version_row_present", len(ver_rows) >= 1, str(len(ver_rows)))
check("master_version_v245", any("v2.45" in l for l in ver_rows), str(ver_rows[:1]))
check("master_447_card", any(re.search(r"4\.47", l) and "E45" in l for l in master), "无 §4.47 卡片")
e45_rows = [l for l in master if re.match(r"^\|\s*\*{0,2}E45", l)]
check("master_section9_e45_row", len(e45_rows) >= 1, str(len(e45_rows)))
if e45_rows:
    cells = [c for c in e45_rows[0].split("|") if c.strip() != ""]
    check("master_section9_e45_row_3cells", len(cells) == 3, "cells=%d" % len(cells))
chg = [l for l in master if re.match(r"^\|\s*2026-11-11\s*\|", l)]
check("master_changelog_v244", any("v2.44" in l for l in chg), "无 v2.44 变更日志行")
check("master_changelog_v245", any("v2.45" in l for l in chg), "无 v2.45 变更日志行")
for v in ("v2.44", "v2.45"):
    rows = [l for l in chg if v in l]
    if rows:
        cells = [c for c in rows[0].split("|") if c.strip() != ""]
        check("master_changelog_cells4:%s" % v, len(cells) == 4, "cells=%d" % len(cells))
idx_rows = [l for l in index if re.match(r"^\|\s*\*{0,2}E45", l)]
check("index_e45_row", len(idx_rows) >= 1, str(len(idx_rows)))
check("master_mentions_floor", any("0.0924" in l for l in master), "MASTER 缺地板 0.0924")
check("master_mentions_oracle", any("0.1653" in l for l in master), "MASTER 缺 Oracle 0.1653")
check("master_mentions_fieldmax", any("402.43" in l for l in master), "MASTER 缺场最大值 402.43")
h0 = "\n".join(handover)
check("handover_says_finished", ("已全部跑完" in h0) and ("RUNNING" not in h0.split("### 真正的下一步")[0]), "交接文档现状段陈旧")
check("handover_nextstep_section", "### 真正的下一步" in h0, "交接文档缺「真正的下一步」段")

# ---------------- 8. 报告 ----------------
print("=" * 78)
print("E45 ROUND-25 INDEPENDENT VERIFICATION")
print("=" * 78)
for line in info:
    print("  " + line)
print("-" * 78)
print("POS %d/%d  NEG %d/%d  PAT %d/%d"
      % (len(pos_ok), len(pos_ok) + len(pos_bad), len(neg_ok), len(neg_ok) + len(neg_bad),
         len(pat_ok), len(pat_ok) + len(pat_bad)))
for n, d in pos_bad:
    print("  [POS-FAIL] %s  %s" % (n, d))
for n, d in neg_bad:
    print("  [NEG-FAIL] %s  %s" % (n, d))
for n, g, e in pat_bad:
    print("  [PAT-FAIL] %s  got=%s expect=%s" % (n, g, e))
print("VERDICT:", "PASS" if not (pos_bad or neg_bad or pat_bad) else "FAIL")
