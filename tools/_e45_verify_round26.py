# -*- coding: utf-8 -*-
"""E45 第 26 轮独立复核（不 import 任何旧脚本，全部从原始产物重取）。

本轮新增两项此前未做的检查：
  (A) E45 结案文档「机制读数（三 seed 全部读出）」表 ↔ 原始探针 JSON **逐值对读**
      （覆盖此前只抽查过的 corr_mask_amp / mask_at_peak / grad_top1pct_share /
        field_max_mm / scale_cv / scale_truth_corr，本轮扩到全部 12 行 × 2 臂 × 3 seed = 72 值，
        并新增 mask_dry_frac_lt_0p1 / corr_mask_resid / tail_max_over_p99 三行）；
  (B) 用**独立的 S1 结构探针**（struct_metrics_all.json，tools/_struct_probe_one_case.py::metrics）
      交叉验证场最大值 —— 这是与机制探针完全不同的工具链。

断言三通道：POS（应为真）/ NEG（违规事件不应发生）/ PAT（逐 seed 期望模式）。
计数断言齐备，避免「空集合空洞通过」。
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
MODEL_FIELD_CLAMP_MM = math.expm1(6.0)  # 402.4288... 模型 log 输出的 clamp 上界
EXPECT_S3_VERDICT = {
    "A1": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A4": "GAIN_WAS_AREA_OR_AMPLITUDE",
    "A2": "NO_GAIN_TO_BEGIN_WITH",
    "A4b": "NO_GAIN_TO_BEGIN_WITH",
}
# 文档标签 → 产物键
DOC_READING_KEY = {
    "corr(mask,amp)": "corr_mask_amp",
    "mask_at_peak": "mask_at_peak",
    "maxratio_as_trained": "maxratio_as_trained",
    "mask_dry_frac_lt_0p1": "mask_dry_frac_lt_0p1",
    "corr_mask_resid": "corr_mask_resid",
    "grad_top1pct_share": "grad_top1pct_share",
    "field_max_mm": "field_max_mm",
    "tail_max_over_p99": "tail_max_over_p99",
    "scale_cv": "scale_cv",
    "scale_truth_corr": "scale_truth_corr",
    "scale_is_constant": "scale_is_constant",
    "mean_out_equals_T": "mean_out_equals_T",
}
BOOL_READINGS = {"scale_is_constant", "mean_out_equals_T"}

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


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return f.read().splitlines()


# ---------------- 1. 产物存在性 ----------------
need = ["scores_summary.json", "scores_all.json", "struct_metrics_all.json", "s3_equalbias.json",
        "diag_probe_s0.json", "diag_probe_s1.json", "diag_probe_s2.json",
        "scores_summary_s0.json", "scores_summary_s1.json", "scores_summary_s2.json",
        "struct_metrics_s0.json", "struct_metrics_s1.json", "struct_metrics_s2.json"]
missing = [f for f in need if not os.path.exists(os.path.join(EV, f))]
check("artifact_exists_all13", not missing, "missing=%s" % missing)
if missing:
    print("FATAL: 产物缺失", missing)
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
    info.append("ARM %-4s median_csi=%.6f range=[%.6f,%.6f] median_bias=%.4f floor=%d/3 over_oracle=%d/3 spread=%.3fx"
                % (arm, med, min(vals), max(vals), medb, npass, norc, spread))

# ---------------- 3. scores_all ↔ summary 互校（21 条计数断言） ----------------
cmp_n = 0
for arm in ARMS7:
    for i, s in enumerate((0, 1, 2)):
        key = "%s_s%d_ep00" % (arm, s)
        rec = sa.get(key)
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
        neg("s3_%s_%s_ci_crosses_zero" % (arm, blk), not recomputed, "ci=[%r,%r] 不跨零" % (lo, hi))
    check("s3_verdict:%s" % arm, a.get("s3_verdict") == EXPECT_S3_VERDICT[arm],
          "got=%r expect=%r" % (a.get("s3_verdict"), EXPECT_S3_VERDICT[arm]))

# ---------------- 5. 机制读数（逐臂逐 seed，从三份原始探针重取） ----------------
def ro(arm, key, seed_idx):
    return dps[seed_idx]["arms"].get(arm, {}).get("readouts", {}).get(key)


for arm in ("A1", "A2"):
    cma = [ro(arm, "corr_mask_amp", i) for i in range(3)]
    mpeak = [ro(arm, "mask_at_peak", i) for i in range(3)]
    check("gate_corr_nonneg:%s" % arm, all(v is not None and v >= 0 for v in cma), str(cma))
    check("gate_mask_at_peak:%s" % arm, all(v is not None and v > 0.5 for v in mpeak), str(mpeak))
    neg("gate_corr_nonneg_viol:%s" % arm, any(v is None or v < 0 for v in cma), str(cma))
    neg("gate_mask_at_peak_viol:%s" % arm, any(v is None or v <= 0.5 for v in mpeak), str(mpeak))
    mr = [ro(arm, "maxratio_as_trained", i) for i in range(3)]
    neg("maxratio_as_trained_all_nan:%s" % arm, any(not is_nan(v) for v in mr), str(mr))
    info.append("ARM %s corr_mask_amp=%s mask_at_peak=%s maxratio_as_trained=%s"
                % (arm, ["%.4f" % v for v in cma], ["%.4f" % v for v in mpeak], mr))

for arm in ("A3a", "A3b"):
    share = [ro(arm, "grad_top1pct_share", i) for i in range(3)]
    fmax = [ro(arm, "field_max_mm", i) for i in range(3)]
    tail = [ro(arm, "tail_max_over_p99", i) for i in range(3)]
    pass_pat = [(v is not None and v < 0.5) and (f is not None and f <= 30.0) for v, f in zip(share, fmax)]
    pat("topo_pass_pattern:%s" % arm, pass_pat, [False, False, False])
    neg("topo_no_seed_passes:%s" % arm, any(pass_pat), "pass=%s" % pass_pat)
    check("topo_fmax_gt30:%s" % arm, all(f is not None and f > 30.0 for f in fmax), str(fmax))
    check("topo_tail_gt10:%s" % arm, all(t is not None and t > 10.0 for t in tail), str(tail))
    info.append("ARM %s grad_top1pct_share=%s field_max_mm=%s tail_max_over_p99=%s topo_pass=%s"
                % (arm, ["%.4f" % v for v in share], ["%.4f" % v for v in fmax],
                   ["%.2f" % v for v in tail], pass_pat))

for arm in ("A4", "A4b"):
    cv = [ro(arm, "scale_cv", i) for i in range(3)]
    tc = [ro(arm, "scale_truth_corr", i) for i in range(3)]
    isc = [ro(arm, "scale_is_constant", i) for i in range(3)]
    mot = [ro(arm, "mean_out_equals_T", i) for i in range(3)]
    check("scale_cv_ge_0p05:%s" % arm, all(v is not None and v >= 0.05 for v in cv), str(cv))
    tc_pass = [(v is not None and v >= 0.3) for v in tc]
    pat("scale_truth_corr_pattern:%s" % arm, tc_pass,
        [True, True, True] if arm == "A4" else [True, True, False])
    if arm == "A4":
        neg("scale_truth_corr_viol:A4", any(not p for p in tc_pass), str(tc))
    pat("scale_is_constant_pattern:%s" % arm, isc, [False, False, False] if arm == "A4" else [False, False, True])
    pat("mean_out_equals_T_pattern:%s" % arm, mot, [True, False, True] if arm == "A4" else [False, False, False])
    info.append("ARM %s scale_cv=%s scale_truth_corr=%s is_constant=%s mean_out_equals_T=%s"
                % (arm, ["%.3f" % v for v in cv], ["%.4f" % v for v in tc], isc, mot))

# ---------------- 6. 【新】A. E45 结案文档读数表 ↔ 原始产物逐值对读 ----------------
e45_doc = read_lines(os.path.join(ROOT, "docs", "experiments",
                                   "E45-gate-direction-amount-separation-baseline.md"))
start = None
end = None
for i, l in enumerate(e45_doc):
    if start is None and "机制读数（三 seed 全部读出）" in l:
        start = i
    if start is not None and "结案判读（逐机制族）" in l:
        end = i
        break
check("e45_reading_table_region_found", start is not None and end is not None and end > start,
      "start=%r end=%r" % (start, end))

doc_rows = {}
cur_arms = None
if start is not None and end is not None:
    for l in e45_doc[start:end]:
        if not l.startswith("|"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        heads = re.findall(r"(A[0-9a-b]+)（s0/s1/s2）", l)
        if heads:
            cur_arms = heads
            continue
        label = cells[0].strip("`* ").strip()
        if label not in DOC_READING_KEY or cur_arms is None:
            continue
        for k, arm in enumerate(cur_arms):
            if k + 2 >= len(cells):
                continue
            doc_rows[(arm, label)] = cells[k + 2]

check("e45_doc_rows_parsed_24", len(doc_rows) == 24, "parsed=%d" % len(doc_rows))

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_doc_token(tok):
    """返回 ('num', value, ndec) / ('bool', value) / ('missing', None)。"""
    c = tok.replace("✔", "").replace("✘", "").replace("**", "").strip()
    if ("空值" in c) or ("未产出" in c) or (c == "—"):
        return ("missing", None)
    if c.startswith("True") or c.startswith("False"):
        return ("bool", c.startswith("True"))
    m = NUM_RE.search(c)
    if not m:
        return ("missing", None)
    t = m.group(0)
    ndec = len(t.split(".")[1]) if "." in t else 0
    return ("num", float(t), ndec)


def parse_doc_cell(cell):
    """文档单元格是 'v0／v1／v2'（全角斜杠分隔三 seed）；'未产出' 之类只有一段时按缺失三 seed 处理。"""
    toks = [t for t in cell.split("／")]
    parsed = [parse_doc_token(t) for t in toks]
    if len(parsed) == 1:
        parsed = parsed * 3
    if len(parsed) != 3:
        parsed = (parsed + [("missing", None)] * 3)[:3]
    return parsed


doc_cmp = 0
doc_mismatch = []
for (arm, label), cell in sorted(doc_rows.items()):
    key = DOC_READING_KEY[label]
    parsed = parse_doc_cell(cell)
    for i in range(3):
        kind = parsed[i][0]
        v = ro(arm, key, i)
        doc_cmp += 1
        if kind == "num":
            dv, ndec = parsed[i][1], parsed[i][2]
            if is_nan(v):
                doc_mismatch.append((arm, label, i, "doc=%r artifact=NAN" % dv))
                continue
            doc_cmp_ok = abs(round(float(v), ndec) - dv) < 1e-9
        elif kind == "bool":
            doc_cmp_ok = (v is not None) and (bool(v) == parsed[i][1])
        else:  # missing
            doc_cmp_ok = is_nan(v)
        if not doc_cmp_ok:
            doc_mismatch.append((arm, label, i, "doc_cell=%r artifact=%r" % (cell, v)))
check("e45_doc_vs_artifact_72_values", doc_cmp == 72, "compared=%d" % doc_cmp)
check("e45_doc_vs_artifact_all_match", not doc_mismatch,
      "mismatch=%s" % doc_mismatch[:8])
info.append("DOC-vs-ARTIFACT compared=%d mismatches=%d" % (doc_cmp, len(doc_mismatch)))
for mm in doc_mismatch[:8]:
    info.append("  MISMATCH %s" % (mm,))

# ---------------- 7. 【新】B. S1 结构探针（独立工具链）交叉验证场最大值 ----------------
tags = sm.get("tags", {})
check("struct_tags_21_entries", len(tags) == 21, "n=%d" % len(tags))
expect_keys = {"%s_s%d_ep00" % (a, s) for a in ARMS7 for s in (0, 1, 2)}
check("struct_tags_keyset", set(tags.keys()) == expect_keys,
      "missing=%s extra=%s" % (sorted(expect_keys - set(tags.keys())), sorted(set(tags.keys()) - expect_keys)))
over30 = []
clamp_viol = []
for key in sorted(tags.keys()):
    rec = tags[key]
    pred = rec.get("pred", {})
    mx = pred.get("max")
    check("struct_pred_max_present:%s" % key, isinstance(mx, (int, float)), "max=%r" % mx)
    if isinstance(mx, (int, float)):
        if mx > MODEL_FIELD_CLAMP_MM + 1e-6:
            clamp_viol.append((key, mx))
        if mx > 30.0:
            over30.append((key, round(mx, 4)))
    check("struct_nwin_538:%s" % key, rec.get("n_windows_measured") == 538, str(rec.get("n_windows_measured")))
check("struct_max_within_model_clamp", not clamp_viol, "viol=%s" % clamp_viol[:5])
a3_over = [k for k, _ in over30 if k.startswith(("A3a", "A3b"))]
check("struct_a3_pred_max_over30", len(a3_over) == 6, "n=%d list=%s" % (len(a3_over), a3_over))
info.append("STRUCT-PROBE pred.max>30mm 的臂-种子: %s" % (sorted(over30) or "无"))
info.append("STRUCT-PROBE A3a/A3b 六条 pred.max 均 > 30mm ⇒ 拓扑支结掉在**独立工具链上同样成立**")

# ---------------- 8. 晋级合取判定 ----------------
qual = []
for arm in S3ARMS:
    a = s3["arms"][arm]
    raw_cross = a["raw"]["ci_lo"] <= 0.0 <= a["raw"]["ci_hi"]
    eq_cross = a["equal_bias"]["ci_lo"] <= 0.0 <= a["equal_bias"]["ci_hi"]
    all_floor = ss[arm].get("pass_rate_over_floor") == "3/3"
    if (not raw_cross) and (not eq_cross) and all_floor:
        qual.append(arm)
check("promotion_qualified_set_empty", qual == [], "qualified=%s" % qual)
info.append("PROMOTION qualified arms = %s" % (qual if qual else "[]（空集）"))

# ---------------- 9. 文档状态与时效性 ----------------
master = read_lines(os.path.join(ROOT, "docs", "MASTER.md"))
index = read_lines(os.path.join(ROOT, "docs", "experiments", "INDEX.md"))
handover = read_lines(os.path.join(os.path.dirname(ROOT), ".workbuddy", "memory", "HANDOVER_E45.md"))

ver_rows = [l for l in master if l.lstrip().startswith("|") and "最后更新" in l]
check("master_version_v245", any("v2.45" in l for l in ver_rows), str(ver_rows[:1]))
check("master_447_card", any(re.search(r"4\.47", l) and "E45" in l for l in master), "无 §4.47 卡片")
e45_rows = [l for l in master if re.match(r"^\|\s*\*{0,2}E45", l)]
check("master_section9_e45_row", len(e45_rows) >= 1, str(len(e45_rows)))
if e45_rows:
    cells = [c for c in e45_rows[0].split("|") if c.strip() != ""]
    check("master_section9_e45_row_3cells", len(cells) == 3, "cells=%d" % len(cells))
chg = [l for l in master if re.match(r"^\|\s*2026-11-11\s*\|", l)]
for v in ("v2.44", "v2.45"):
    rows = [l for l in chg if v in l]
    check("master_changelog_%s" % v, len(rows) >= 1, "n=%d" % len(rows))
    if rows:
        cells = [c for c in rows[0].split("|") if c.strip() != ""]
        check("master_changelog_cells4:%s" % v, len(cells) == 4, "cells=%d" % len(cells))
idx_rows = [l for l in index if re.match(r"^\|\s*\*{0,2}E45", l)]
check("index_e45_row", len(idx_rows) >= 1, str(len(idx_rows)))
for token, nm in (("0.0924", "floor"), ("0.1653", "oracle"), ("402.43", "field_max")):
    check("master_mentions_%s" % nm, any(token in l for l in master), "MASTER 缺 %s" % token)
h0 = "\n".join(handover)
check("handover_says_finished", ("已全部跑完" in h0) and ("RUNNING" not in h0.split("### 真正的下一步")[0]),
      "交接文档现状段陈旧")
check("handover_nextstep_section", "### 真正的下一步" in h0, "交接文档缺「真正的下一步」段")

# ---------------- 10. 报告 ----------------
print("=" * 78)
print("E45 ROUND-26 INDEPENDENT VERIFICATION")
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
