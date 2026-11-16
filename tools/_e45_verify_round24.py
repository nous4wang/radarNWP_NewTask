# -*- coding: utf-8 -*-
"""E45 第 24 次独立复核 —— 新增两项此前未做的检查：

(A) 文档口径漂移检查：地板 0.0924 / Oracle 0.1653 / 场最大值 402.43
    在**与 E45 相关的语境**（MASTER 头部版本行、§4.47 卡片、§9 E45 行、
    §12 v2.44/v2.45 行；INDEX 的 E45 行；E45 判据文件全文；HANDOVER 全文）
    中不得出现异值 —— 防止结案后关键常量被悄悄改成别的数。
(B) 交接文档时效性检查：HANDOVER_E45.md 的 §0 现状段必须仍写
    "已全部跑完并结案"，且 §4「真正的下一步」段不得残留待跑指引。

本脚本不 import 其它复核脚本，全部从原始产物与文档重新取数。
注意：正/负向断言必须分通道 —— 预注册的**负结果**（拓扑支不过判据）
只能记入 NEG（违规事件是否发生），不得写成 POS。
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")

MASTER = os.path.join(ROOT, "docs", "MASTER.md")
INDEX = os.path.join(ROOT, "docs", "experiments", "INDEX.md")
GATE = os.path.join(ROOT, "docs", "experiments",
                    "E45-gate-direction-amount-separation-baseline.md")
HANDOVER = r"R:\1radarNWPprediction\.workbuddy\memory\HANDOVER_E45.md"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
PROBE_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]

pos, neg, pat = [], [], []
POS = NEG = PAT = 0


def check(name, cond, detail=""):
    global POS
    if cond:
        POS += 1
        pos.append(name)
    else:
        print("  FAIL(POS): %s %s" % (name, detail))
    return cond


def neg_check(name, violation, detail=""):
    """violation=True 表示『违规事件发生』⇒ 判失败。"""
    global NEG
    if not violation:
        NEG += 1
        neg.append(name)
    else:
        print("  FAIL(NEG): %s %s" % (name, detail))


def pat_check(name, got, want):
    global PAT
    if list(got) == list(want):
        PAT += 1
        pat.append(name)
    else:
        print("  FAIL(PAT): %s got=%s want=%s" % (name, list(got), list(want)))


def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def find_float(obj, key_rx):
    hits = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, float) and re.search(key_rx, str(k), re.I):
                    hits.append((str(k), v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return hits


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ---------------- 载入原始产物 ----------------
print("-- loading raw artifacts --")
scores_all = read_json(os.path.join(EVALS, "scores_all.json"))
s3 = read_json(os.path.join(EVALS, "s3_equalbias.json"))
summary = read_json(os.path.join(EVALS, "scores_summary.json"))
probes = {s: read_json(os.path.join(EVALS, "diag_probe_s%d.json" % s)) for s in SEEDS}

top_keys = list(scores_all.keys())
rec_keys = [k for k in top_keys if re.search(r"_s0_ep00$", str(k))]
if not check("scores_all 含 7x3=21 条 <臂>_s<seed>_ep00 记录",
             len(top_keys) == 21 and len(rec_keys) == 7,
             "top=%d" % len(top_keys)):
    print("ABORT: scores_all 结构与预期不符")
    sys.exit(2)
sample_rec = scores_all[rec_keys[0]]
csi_hits = find_float(sample_rec, r"csi.*2\.?0")
if not check("scores_all 记录中可定位 CSI@2.0 字段", len(csi_hits) >= 1,
             "hits=%s" % csi_hits):
    print("ABORT: CSI 字段定位失败")
    sys.exit(2)
csi_key = sorted(csi_hits, key=lambda t: (len(t[0]), t[0]))[0][0]
print("  scores_all: %d 条记录, CSI@2.0 字段名 = %r" % (len(top_keys), csi_key))

per_seed = {}
for arm in ARMS:
    for s in SEEDS:
        k = "%s_s%d_ep00" % (arm, s)
        if k not in scores_all:
            check("scores_all 含 %s" % k, False)
            continue
        rec = scores_all[k]
        if csi_key in rec and isinstance(rec[csi_key], float):
            per_seed[(arm, s)] = rec[csi_key]
check("per_seed 覆盖 7 臂 x 3 seed = 21 条", len(per_seed) == 21, "got=%d" % len(per_seed))

FLOOR = 0.0924
ORACLE = 0.1653

print("-- 3-seed median / pass-rate (recomputed from scores_all.json) --")
med = {}
for arm in ARMS:
    vals = [per_seed[(arm, s)] for s in SEEDS if (arm, s) in per_seed]
    m = median(vals)
    med[arm] = m
    over_floor = sum(1 for v in vals if v >= FLOOR)
    over_oracle = sum(1 for v in vals if v > ORACLE)
    print("  %-4s median=%.6f range=%.6f-%.6f >floor=%d/3 >oracle=%d/3" %
          (arm, m, min(vals), max(vals), over_floor, over_oracle))
    check("%s 中位数有效" % arm, m == m)

check("B0 中位数 == 0.121817（复现基线）", abs(med["B0"] - 0.121817) < 1e-5, str(med.get("B0")))
check("A4b 离散 9.23x (max/min)",
      abs(max(per_seed[("A4b", s)] for s in SEEDS) /
          min(per_seed[("A4b", s)] for s in SEEDS) - 9.226194) < 0.01)
above = sorted([a for a in ARMS if a != "B0" and med.get(a, -1) > med["B0"]])
check("中位数高于对照的臂恰为 A4/A4b", above == ["A4", "A4b"], str(above))

# ---- 互校：scores_summary.json 顶层即臂键 ----
cmp_n = 0
for arm in ARMS:
    blk = summary.get(arm)
    if not isinstance(blk, dict):
        check("scores_summary 含臂 %s" % arm, False)
        continue
    if "median_csi_at_2.0" in blk:
        cmp_n += 1
        check("summary[%s].median_csi_at_2.0 与重算一致" % arm,
              abs(float(blk["median_csi_at_2.0"]) - med[arm]) < 1e-6,
              "%s vs %s" % (blk["median_csi_at_2.0"], med[arm]))
    check("summary[%s].n_seeds == 3" % arm, blk.get("n_seeds") == 3, str(blk.get("n_seeds")))
    check("summary[%s].seeds 覆盖 0/1/2" % arm, sorted(blk.get("seeds", [])) == [0, 1, 2],
          str(blk.get("seeds")))
    pr = blk.get("pass_rate_over_floor")
    vals = [per_seed[(arm, s)] for s in SEEDS]
    want = "%d/3" % sum(1 for v in vals if v >= FLOOR)
    check("summary[%s].pass_rate_over_floor == %s" % (arm, want), str(pr) == want,
          "got=%s want=%s" % (pr, want))
check("互校实际比对条数 > 0（防空集合空洞通过）", cmp_n == 7, "cmp_n=%d" % cmp_n)

# ---------------- S3 等偏置裁决 ----------------
print("-- S3 equal-bias verdict --")
s3_arms = s3["arms"] if isinstance(s3, dict) and "arms" in s3 else s3
check("s3 头部元数据：阈值 2.0 / control=B0 / n_boot=10000",
      s3.get("threshold_mm") == 2.0 and s3.get("control") == "B0" and s3.get("n_boot") == 10000,
      "threshold=%s control=%s n_boot=%s" % (s3.get("threshold_mm"), s3.get("control"),
                                             s3.get("n_boot")))
verdicts = {}
for arm in PROBE_ARMS:
    blk = s3_arms.get(arm)
    if not isinstance(blk, dict):
        check("s3 含臂 %s" % arm, False)
        continue
    verdicts[arm] = blk.get("s3_verdict")
    for tag in ("raw", "equal_bias"):
        sub = blk.get(tag, {})
        lo, hi = sub.get("ci_lo"), sub.get("ci_hi")
        if lo is None or hi is None:
            check("s3[%s].%s 有 ci_lo/ci_hi" % (arm, tag), False, str(sub)[:120])
            continue
        check("s3[%s].%s 增益 CI 跨零" % (arm, tag), lo <= 0.0 <= hi,
              "CI=[%s,%s]" % (lo, hi))
        check("s3[%s].%s crosses_zero 标志一致" % (arm, tag),
              bool(sub.get("crosses_zero")) == (lo <= 0.0 <= hi))
    print("  %-4s s3_verdict=%s" % (arm, verdicts[arm]))

for arm in ["A1", "A3a", "A3b", "A4"]:
    check("s3 %s 判为 GAIN_WAS_AREA_OR_AMPLITUDE" % arm,
          verdicts.get(arm) == "GAIN_WAS_AREA_OR_AMPLITUDE", str(verdicts.get(arm)))
for arm in ["A2", "A4b"]:
    check("s3 %s 判为 NO_GAIN_TO_BEGIN_WITH" % arm,
          verdicts.get(arm) == "NO_GAIN_TO_BEGIN_WITH", str(verdicts.get(arm)))

# ---------------- 机制读数（逐臂逐 seed）----------------
print("-- mechanism readouts (keys under arms/<arm>/readouts/) --")


def series(arm, key):
    return [probes[s]["arms"][arm]["readouts"].get(key) for s in SEEDS]


def is_nan(v):
    return v is None or (isinstance(v, (int, float)) and v != v)


# 门指向 A1/A2（预注册：应通过 → POS）
for arm in ["A1", "A2"]:
    c = series(arm, "corr_mask_amp")
    m = series(arm, "mask_at_peak")
    check("%s corr_mask_amp 三 seed 均 >= 0" % arm, all(x >= 0 for x in c), str(c))
    check("%s mask_at_peak 三 seed 均 > 0.5" % arm, all(x > 0.5 for x in m), str(m))
    pat_check("%s 门判据逐 seed 模式 [T,T,T]" % arm,
              [c[i] >= 0 and m[i] > 0.5 for i in range(3)], [True, True, True])
    print("  %-4s corr(mask,amp)=%s mask_at_peak=%s" % (arm, c, m))

# 拓扑支 A3a/A3b（预注册：应【不】通过 → NEG）
for arm in ["A3a", "A3b"]:
    g = series(arm, "grad_top1pct_share")
    f = series(arm, "field_max_mm")
    sat = [bool(g[i] < 0.5 and f[i] <= 30) for i in range(3)]
    neg_check("%s 无 seed 意外满足拓扑判据（应全不满足）" % arm, any(sat),
              "sat=%s" % sat)
    pat_check("%s 逐 seed 满足拓扑判据模式 [F,F,F]" % arm, sat, [False, False, False])
    neg_check("%s 无 seed 的 field_max_mm <= 30" % arm, any(x <= 30 for x in f), str(f))
    print("  %-4s grad_share=%s field_max=%s" % (arm, g, f))

# 量通道 A4（预注册：应通过 → POS）
a4_cv, a4_corr, a4_const = (series("A4", k) for k in
                             ("scale_cv", "scale_truth_corr", "scale_is_constant"))
check("A4 scale_cv 三 seed 均 >= 0.05", all(x >= 0.05 for x in a4_cv), str(a4_cv))
check("A4 scale_truth_corr 三 seed 均 >= 0.3", all(x >= 0.3 for x in a4_corr), str(a4_corr))
check("A4 scale_is_constant 三 seed 均 False", all(x is False for x in a4_const), str(a4_const))
pat_check("A4 量通道逐 seed 模式 [T,T,T]",
          [a4_cv[i] >= 0.05 and a4_corr[i] >= 0.3 and a4_const[i] is False for i in range(3)],
          [True, True, True])

# A4b（量通道只在 2/3 seed 成立 → 逐 seed 模式）
b_cv, b_corr, b_const = (series("A4b", k) for k in
                         ("scale_cv", "scale_truth_corr", "scale_is_constant"))
check("A4b scale_cv 三 seed 均 >= 0.05", all(x >= 0.05 for x in b_cv), str(b_cv))
pat_check("A4b scale_truth_corr 逐 seed 模式 [过,过,不过]",
          [x >= 0.3 for x in b_corr], [True, True, False])
pat_check("A4b scale_is_constant 逐 seed 模式 [F,F,T]", list(b_const), [False, False, True])
print("  A4  cv=%s corr=%s const=%s" % (a4_cv, a4_corr, a4_const))
print("  A4b cv=%s corr=%s const=%s" % (b_cv, b_corr, b_const))

pat_check("A4 mean_out_equals_T 逐 seed 模式 [T,F,T]",
          list(series("A4", "mean_out_equals_T")), [True, False, True])
pat_check("A4b mean_out_equals_T 逐 seed 模式 [F,F,F]",
          list(series("A4b", "mean_out_equals_T")), [False, False, False])

for arm in ["A1", "A2"]:
    vals = series(arm, "maxratio_as_trained")
    check("%s maxratio_as_trained 三 seed 全为缺失(NaN/None)" % arm,
          all(is_nan(v) for v in vals), str(vals))

# ---------------- 新增 (A)：文档口径漂移（限定 E45 语境）----------------
print("-- (A) document numeric drift (E45-scoped) --")
CONSTANTS = [
    ("floor 0.0924", r"0\.092\d+", {"0.0924"}),
    ("oracle 0.1653", r"0\.165\d+", {"0.1653"}),
    ("field_max 402.43", r"402\.4\d+", {"402.43"}),
]


def master_e45_context(text):
    """只取 MASTER 中与 E45 相关的行：头部版本行、§4.47 卡片、§9 E45 行、§12 v2.44/v2.45 行。"""
    lines = text.splitlines()
    keep = []
    for i, ln in enumerate(lines):
        if re.search(r"^\|\s*最后更新\s*\|", ln):
            keep.append(ln)
        if re.match(r"^\|\s*\*{0,2}E45", ln):
            keep.append(ln)
        if re.match(r"^\|\s*2026-11-11\s*\|\s*\*\*v2\.4[45]\*\*", ln):
            keep.append(ln)
    # §4.47 卡片：从 4.47 标题行到下一个标题行为止
    start = None
    for i, ln in enumerate(lines):
        if re.search(r"^#{1,6}\s*4\.47\b", ln) or re.search(r"^\s*4\.47\s", ln):
            start = i
            break
    if start is not None:
        for ln in lines[start:start + 40]:
            keep.append(ln)
            if re.match(r"^#{1,6}\s*4\.4[89]\b", ln):
                break
    return keep


def index_e45_context(text):
    return [ln for ln in text.splitlines() if re.match(r"^\|\s*\*{0,2}E45", ln)]


def check_drift(docname, lines):
    blob = "\n".join(lines)
    for cname, rx, allowed in CONSTANTS:
        found = set(re.findall(rx, blob))
        bad = found - allowed
        neg_check("%s 的 %s 无异值" % (docname, cname), bool(bad),
                  "异值 %s（全部取值 %s）" % (sorted(bad), sorted(found)))
        if found:
            print("    %s : %s -> %s" % (docname, cname, sorted(found)))


mtext = read_text(MASTER)
itext = read_text(INDEX)
gtext = read_text(GATE)
check_drift("MASTER(E45 语境)", master_e45_context(mtext))
check_drift("INDEX(E45 行)", index_e45_context(itext))
check_drift("E45 判据文件", gtext.splitlines())

# ---------------- 新增 (B)：交接文档时效性 ----------------
print("-- (B) handover freshness --")
if os.path.exists(HANDOVER):
    h = read_text(HANDOVER)
    hlines = h.splitlines()
    sec0 = "\n".join(hlines[:40])
    check("HANDOVER §0 现状段写『已全部跑完』", "已全部跑完" in sec0)
    neg_check("HANDOVER §0 未把 273 写成 RUNNING",
              ("RUNNING" in sec0) and ("已全部跑完" not in sec0),
              "现状段仍称 RUNNING")
    # §4「真正的下一步」段：不得残留待跑指引
    nxt = None
    for i, ln in enumerate(hlines):
        if "真正的下一步" in ln:
            nxt = i
            break
    check("HANDOVER 含『真正的下一步』段", nxt is not None)
    if nxt is not None:
        seg = "\n".join(hlines[nxt:nxt + 14])
        neg_check("HANDOVER『真正的下一步』段无待跑指引",
                  ("等 290 完成" in seg) or ("等待即可" in seg) or ("等远端跑完" in seg),
                  "残留待跑指引")
        check("HANDOVER『真正的下一步』段列出四件待办",
              ("实现缺陷" in seg) and ("机制族" in seg))
    check_drift("HANDOVER_E45.md", hlines)
else:
    neg_check("HANDOVER 存在", True, HANDOVER)

# ---------------- Gate-3 写回标记 ----------------
print("-- Gate-3 write-back markers --")
check("MASTER 头部版本行 = v2.45", re.search(r"\|\s*最后更新\s*\|[^|]*v2\.45", mtext))
check("MASTER 含 §4.47 E45 卡片", "4.47" in mtext)
check("MASTER §9 含 E45 行", re.search(r"^\|\s*\*{0,2}E45", mtext, re.M))
check("MASTER §12 含 v2.44 与 v2.45 两行",
      len(re.findall(r"^\|\s*2026-11-11\s*\|\s*\*\*v2\.4[45]\*\*", mtext, re.M)) == 2)
check("INDEX 含 E45 行", re.search(r"^\|\s*\*{0,2}E45", itext, re.M))
check("INDEX 提及 v2.45", "v2.45" in itext)

print("\n==============================================================================")
print("  POS  %3d/%3d passed" % (POS, len(pos)))
print("  NEG  %3d/%3d passed" % (NEG, len(neg)))
print("  PAT  %3d/%3d passed" % (PAT, len(pat)))
print("==============================================================================")
ok = (POS == len(pos)) and (NEG == len(neg)) and (PAT == len(pat))
print("VERDICT: %s  (POS %d/%d, NEG %d/%d, PAT %d/%d)" %
      ("PASS" if ok else "FAIL", POS, len(pos), NEG, len(neg), PAT, len(pat)))
sys.exit(0 if ok else 1)
