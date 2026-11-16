"""E45 第 22 轮独立复核（2026-11-12）。

原则（沿用历轮教训）：
  * 不 import 任何旧复核脚本，一切从原始产物重取，避免旧口径 bug 静默传染；
  * 结构先行：读数用递归查找，不猜键路径；
  * 三态判定：None / NaN / 有效值分开处理；
  * POS（应成立）与 NEG（应不成立）分两条通道，NEG 用「违规事件是否发生」作参数；
  * 互校断言必须校验实际比对条数（N=0 直接失败），不允许空洞通过；
  * 逐 seed 期望模式必须绑定到具体读数，不能按臂套用。

远端 md5 取自本轮 remote.py 实拉（2026-11-12 13:58）。
"""
import hashlib
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "tools", "_evals", "e45")
DOCS = os.path.join(ROOT, "docs")
UPPER = r"R:\1radarNWPprediction\.workbuddy\memory"

FLOOR = 0.0924
ORACLE = 0.1653
ARMS6 = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ARMS7 = ["B0"] + ARMS6
SEEDS = [0, 1, 2]

# 本轮实拉的远端 md5（runs/e45/*.json，13 份）
REMOTE_MD5 = {
    "diag_probe_s0.json": "da016521a6bb5dfd85da4f518e3fb763",
    "diag_probe_s1.json": "cf69079c3fdef206fe3e3fedaa12685b",
    "diag_probe_s2.json": "01de640b578d5980264259586c1adf18",
    "s3_equalbias.json": "3aba999d5ef6dcc4a775999be6585eb2",
    "scores_all.json": "80655963622b09cd5231588e26468a28",
    "scores_summary.json": "57bcd511a537ab85494ee41b029731a8",
    "scores_summary_s0.json": "bac63c96975fdd5e8ec46dcac4ec2463",
    "scores_summary_s1.json": "35f1142dfda6042bcb3e5a74b60e78d7",
    "scores_summary_s2.json": "80655963622b09cd5231588e26468a28",
    "struct_metrics_all.json": "cd3052bbb8185e6b15fa27afd663db01",
    "struct_metrics_s0.json": "4bc259d01d1450f6c8ca1e43391dd1a4",
    "struct_metrics_s1.json": "75489dcb98d373b34ceb5e7d8636dab3",
    "struct_metrics_s2.json": "077ac3dc307f701866cff695e881cbd2",
}

# MASTER v2.45 已定稿的判读数字（用于互校，不是裁决来源）
EXPECT_MEDIAN_CSI = {"B0": 0.1218, "A1": 0.1184, "A2": 0.1120, "A3a": 0.1100,
                     "A3b": 0.1161, "A4": 0.1439, "A4b": 0.1376}
EXPECT_MEDIAN_BIAS = {"B0": 0.5095, "A1": 0.4430, "A2": 0.4490, "A3a": 0.6394,
                      "A3b": 0.6188, "A4": 0.8362, "A4b": 2.8215}
EXPECT_PASS = {"B0": 2, "A1": 2, "A2": 2, "A3a": 3, "A3b": 3, "A4": 2, "A4b": 2}
EXPECT_OVER_ORACLE = {"B0": 0, "A1": 0, "A2": 0, "A3a": 0, "A3b": 1, "A4": 1, "A4b": 0}
EXPECT_VERDICT = {"A1": "GAIN_WAS_AREA_OR_AMPLITUDE", "A2": "NO_GAIN_TO_BEGIN_WITH",
                  "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE", "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
                  "A4": "GAIN_WAS_AREA_OR_AMPLITUDE", "A4b": "NO_GAIN_TO_BEGIN_WITH"}

pos_pass, pos_fail, neg_pass, neg_fail = [], [], [], []


def check(name, cond, detail=""):
    (pos_pass if cond else pos_fail).append((name, detail))


def neg_check(name, violation, detail=""):
    """neg_check(name, cond)：cond 为真表示「违规事件发生」⇒ 判负结果失败。"""
    (neg_fail if violation else neg_pass).append((name, detail))


def is_missing(x):
    return x is None or (isinstance(x, float) and x != x)


def find_key(node, key):
    """递归查找 key，返回 (found, value)；找不到返回 (False, None)。"""
    if isinstance(node, dict):
        if key in node:
            return True, node[key]
        for v in node.values():
            f, val = find_key(v, key)
            if f:
                return True, val
    elif isinstance(node, list):
        for v in node:
            f, val = find_key(v, key)
            if f:
                return True, val
    return False, None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    with open(os.path.join(EV, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


print("=" * 78)
print("E45 第 22 轮独立复核  ·  2026-11-12")
print("=" * 78)

# ---------------------------------------------------------------- A. md5
print("\n[A] 远端/本地 md5 逐份比对")
local_files = sorted(f for f in os.listdir(EV) if f.endswith(".json"))
check("md5 文件集一致", set(local_files) == set(REMOTE_MD5),
      "local=%d remote=%d" % (len(local_files), len(REMOTE_MD5)))
for fn in sorted(REMOTE_MD5):
    p = os.path.join(EV, fn)
    if not os.path.exists(p):
        check("md5 存在:" + fn, False)
        continue
    got = md5(p)
    check("md5 一致:" + fn, got == REMOTE_MD5[fn], "local=%s remote=%s" % (got, REMOTE_MD5[fn]))
check("md5 比对条数=13", len(local_files) == 13 and len(REMOTE_MD5) == 13,
      "n=%d" % len(local_files))

# ---------------------------------------------------------------- B. S3
print("\n[B] S3 等偏置对照独立重算")
s3 = load("s3_equalbias.json")
check("S3 元数据", s3.get("threshold_mm") == 2.0 and s3.get("control") == "B0"
      and s3.get("n_boot") == 10000 and s3.get("epoch") == "00",
      "thr=%s ctrl=%s nboot=%s epoch=%s" % (s3.get("threshold_mm"), s3.get("control"),
                                             s3.get("n_boot"), s3.get("epoch")))
check("S3 臂集合", sorted(s3["arms"].keys()) == sorted(ARMS6),
      str(sorted(s3["arms"].keys())))
s3_blocks = 0
for arm in ARMS6:
    blk = s3["arms"][arm]
    for stat in ("raw", "equal_bias"):
        b = blk[stat]
        s3_blocks += 1
        lo, hi = b["ci_lo"], b["ci_hi"]
        recomputed = (lo <= 0.0 <= hi)
        check("S3 %s/%s crosses_zero 标志与重算一致" % (arm, stat),
              recomputed == b["crosses_zero"],
              "stored=%s recomputed=%s" % (b["crosses_zero"], recomputed))
        # 预注册负结果：所有增益 CI 必须跨零
        neg_check("S3 %s/%s CI 跨零" % (arm, stat), not recomputed,
                  "ci=[%.6f, %.6f]" % (lo, hi))
        check("S3 %s/%s 元数据" % (arm, stat),
              b["n_boot"] == 10000 and b["n_seeds"] == 3 and b["n_windows"] == 538,
              "nboot=%s nseeds=%s nwin=%s" % (b["n_boot"], b["n_seeds"], b["n_windows"]))
    check("S3 %s 判决标签" % arm, blk.get("s3_verdict") == EXPECT_VERDICT[arm],
          "%s" % blk.get("s3_verdict"))
    check("S3 %s 三个 seed 齐全" % arm, sorted(blk["seeds"].keys()) == ["0", "1", "2"],
          str(sorted(blk["seeds"].keys())))
check("S3 统计块数=12", s3_blocks == 12, "n=%d" % s3_blocks)

# ---------------------------------------------------------------- C. 3-seed 统计
print("\n[C] 3-seed 中位数 / 过地板通过率 / 超 Oracle（与 scores_summary 互校）")
scores_all = load("scores_all.json")
summary = load("scores_summary.json")
check("scores_all 记录数=21", len(scores_all) == 21, "n=%d" % len(scores_all))
check("scores_summary 臂集合", sorted(summary.keys()) == sorted(ARMS7),
      str(sorted(summary.keys())))
stat_rows = []
for arm in ARMS7:
    csi = [scores_all["%s_s%d_ep00" % (arm, s)]["csi_at_2.0"] for s in SEEDS]
    bias = [scores_all["%s_s%d_ep00" % (arm, s)]["bias_at_2.0"] for s in SEEDS]
    med_csi = statistics.median(csi)
    med_bias = statistics.median(bias)
    n_pass = sum(1 for v in csi if v >= FLOOR)
    n_over = sum(1 for v in csi if v > ORACLE)
    spread = max(csi) / min(csi)
    blk = summary[arm]
    check("summary 中位数一致:" + arm, abs(blk["median_csi_at_2.0"] - med_csi) < 5e-4,
          "summary=%.6f recomputed=%.6f" % (blk["median_csi_at_2.0"], med_csi))
    check("summary BIAS 中位数一致:" + arm, abs(blk["median_bias_at_2.0"] - med_bias) < 5e-4,
          "summary=%.6f recomputed=%.6f" % (blk["median_bias_at_2.0"], med_bias))
    pr = blk["pass_rate_over_floor"]
    pr_num, pr_den = (int(x) for x in pr.split("/")) if isinstance(pr, str) else \
        (round(pr * 3), 3)
    check("summary 过地板通过率一致:" + arm,
          pr_num == n_pass and pr_den == 3,
          "summary=%s recomputed=%d/3" % (pr, n_pass))
    check("summary 离散度=极比:" + arm,
          abs(blk["seed_spread_csi_at_2.0"] - spread) < 1e-6,
          "summary=%.6f recomputed=%.6f" % (blk["seed_spread_csi_at_2.0"], spread))
    check("summary 种子覆盖:" + arm, sorted(blk["seeds"]) == [0, 1, 2]
          and blk.get("n_seeds") == 3, str(blk["seeds"]))
    # 与 MASTER v2.45 定稿数字互校
    check("MASTER 中位数互校:" + arm, abs(med_csi - EXPECT_MEDIAN_CSI[arm]) < 5e-4,
          "master=%.4f recomputed=%.4f" % (EXPECT_MEDIAN_CSI[arm], med_csi))
    check("MASTER BIAS 中位数互校:" + arm, abs(med_bias - EXPECT_MEDIAN_BIAS[arm]) < 5e-4,
          "master=%.4f recomputed=%.4f" % (EXPECT_MEDIAN_BIAS[arm], med_bias))
    check("MASTER 过地板计数互校:" + arm, n_pass == EXPECT_PASS[arm],
          "master=%d recomputed=%d" % (EXPECT_PASS[arm], n_pass))
    check("MASTER 超 Oracle 计数互校:" + arm, n_over == EXPECT_OVER_ORACLE[arm],
          "master=%d recomputed=%d" % (EXPECT_OVER_ORACLE[arm], n_over))
    stat_rows.append((arm, med_csi, med_bias, n_pass, n_over, spread,
                      min(csi), max(csi)))
print("  %-4s %-9s %-10s %-8s %-8s %-8s %s" %
      ("臂", "中位CSI", "中位BIAS", "过地板", "超Ora", "离散度", "范围"))
for r in stat_rows:
    print("  %-4s %-9.4f %-10.4f %-8s %-8s %-8.2f %.4f-%.4f" %
          (r[0], r[1], r[2], "%d/3" % r[3], "%d/3" % r[4], r[5], r[6], r[7]))

# 晋级资格合取判定：3/3 过地板 且 S3 配平后增益不跨零（且中位数高于对照）
base_med = dict((r[0], r[1]) for r in stat_rows)["B0"]
promoted = []
for arm in ARMS6:
    r = [x for x in stat_rows if x[0] == arm][0]
    reasons = []
    if r[3] != 3:
        reasons.append("未 3/3 过地板（%d/3）" % r[3])
    if r[1] <= base_med:
        reasons.append("中位数不高于对照 B0")
    eq = s3["arms"][arm]["equal_bias"]
    if eq["crosses_zero"]:
        reasons.append("S3 配平后增益跨零")
    if reasons:
        print("  晋级判定 %-4s ⇒ 不晋级：%s" % (arm, "；".join(reasons)))
    else:
        promoted.append(arm)
check("无臂满足晋级合取条件", promoted == [], "promoted=%s" % promoted)

# ---------------------------------------------------------------- D. 机制读数
print("\n[D] 机制读数逐臂逐 seed 独立重算")
probes = {s: load("diag_probe_s%d.json" % s) for s in SEEDS}


def readout(arm, seed, key):
    found, val = find_key(probes[seed]["arms"][arm], key)
    return val if found else None


# D1 门指向：A1/A2 corr(mask,amp) >= 0 且 mask_at_peak > 0.5
for arm in ("A1", "A2"):
    for s in SEEDS:
        ca = readout(arm, s, "corr_mask_amp")
        mp = readout(arm, s, "mask_at_peak")
        check("门 %s s%d corr(mask,amp)>=0" % (arm, s), ca is not None and ca >= 0.0,
              "corr=%s" % ca)
        check("门 %s s%d mask_at_peak>0.5" % (arm, s), mp is not None and mp > 0.5,
              "mask_at_peak=%s" % mp)
# D2 拓扑支：A3a/A3b 需 grad_top1pct_share < 0.5 且 field_max_mm <= 30（预注册应不成立）
for arm in ("A3a", "A3b"):
    for s in SEEDS:
        gs = readout(arm, s, "grad_top1pct_share")
        fm = readout(arm, s, "field_max_mm")
        ok = (gs is not None and fm is not None and gs < 0.5 and fm <= 30.0)
        neg_check("拓扑 %s s%d 未过预注册判据" % (arm, s), ok,
                  "grad=%s field_max=%s" % (gs, fm))
        neg_check("拓扑 %s s%d field_max_mm>30" % (arm, s),
                  not (fm is not None and fm > 30.0), "field_max=%s" % fm)
# D3 量通道：A4 三 seed 全成立；A4b scale_truth_corr 仅 2/3
for arm in ("A4", "A4b"):
    # 逐 seed 期望模式：A4 三 seed 全未退化；A4b 第 3 个 seed 退化（scale_truth_corr<0.3）
    ic_pat = {"A4": [False, False, False], "A4b": [False, False, True]}[arm]
    for s in SEEDS:
        cv = readout(arm, s, "scale_cv")
        tc = readout(arm, s, "scale_truth_corr")
        ic = readout(arm, s, "scale_is_constant")
        check("量通道 %s s%d scale_cv>=0.05" % (arm, s), cv is not None and cv >= 0.05,
              "scale_cv=%s" % cv)
        check("量通道 %s s%d scale_is_constant 模式" % (arm, s),
              (ic is True) == ic_pat[s], "is_constant=%s expected=%s" % (ic, ic_pat[s]))
        if arm == "A4":
            check("量通道 A4 s%d scale_truth_corr>=0.3" % s, tc is not None and tc >= 0.3,
                  "corr=%s" % tc)
        else:
            # 逐 seed 期望模式 [过, 过, 不过]
            exp = (s == 2)
            check("量通道 A4b s%d scale_truth_corr>=0.3" % s, (tc >= 0.3) == (not exp),
                  "corr=%s expected_pass=%s" % (tc, not exp))
# D4 mean(out)=T：A4 [True,False,True]，A4b 三 seed 全 False ⇒ 实现缺陷
pat = {"A4": [True, False, True], "A4b": [False, False, False]}
for arm, p in pat.items():
    for s in SEEDS:
        v = readout(arm, s, "mean_out_equals_T")
        check("mean(out)=T %s s%d 模式" % (arm, s), (v is True) == p[s],
              "value=%s expected=%s" % (v, p[s]))
# D5 maxratio_as_trained 三 seed 全缺读数（NaN/None）
for arm in ARMS6:
    for s in SEEDS:
        v = readout(arm, s, "maxratio_as_trained")
        neg_check("maxratio_as_trained %s s%d 读数缺失" % (arm, s), not is_missing(v),
                  "value=%r" % (v,))

# ---------------------------------------------------------------- E. 文档锚点
print("\n[E] 文档锚点与写回在位")
master = open(os.path.join(DOCS, "MASTER.md"), encoding="utf-8").read()
index = open(os.path.join(DOCS, "INDEX.md"), "r", encoding="utf-8").read() if False else \
    open(os.path.join(DOCS, "experiments", "INDEX.md"), encoding="utf-8").read()
mem = open(os.path.join(UPPER, "MEMORY.md"), encoding="utf-8").read()
mlines = master.splitlines()
hdr = [l for l in mlines[:20] if l.startswith("| 最后更新 |")]
check("MASTER 头部版本行 v2.45", bool(hdr) and "v2.45" in hdr[0],
      hdr[0][:80] if hdr else "未找到头部行")
check("MASTER §4.47 卡片在位", any("4.47" in l for l in mlines))
e45_rows = [l for l in mlines if re.match(r"^\|\s*\*{0,2}E45", l)]
check("MASTER §9 E45 行在位", len(e45_rows) >= 1, "rows=%d" % len(e45_rows))
check("MASTER §9 E45 行为 3 列", all(len(l.split("|")) - 2 == 3 for l in e45_rows),
      str([len(l.split("|")) - 2 for l in e45_rows]))
chg = [l for l in mlines if re.match(r"^\|\s*2026-11-11\s*\|\s*\*\*v2\.4[45]\*\*", l)]
check("MASTER §12 v2.44/v2.45 两行在位", len(chg) == 2, "n=%d" % len(chg))
check("MASTER §12 行均为 4 列", all(len(l.split("|")) - 2 == 4 for l in chg),
      str([len(l.split("|")) - 2 for l in chg]))
ilines = index.splitlines()
irow = [l for l in ilines if l.strip().startswith("| E45")]
check("INDEX E45 行在位且含 v2.45", bool(irow) and "v2.45" in irow[0],
      irow[0][:90] if irow else "未找到")
check("上层 MEMORY.md 指针 v2.45", "v2.45" in mem)
# 禁写措辞 lint（区分断言与禁令引用：仅查 E45 相关行，且前 2 行内有禁止标记则放行）
BAN = ["已实现 SOTA", "已解决 novelty", "LABEL 有孤立针状尖峰", "C0 被推翻"]
for l in mlines:
    if "E45" not in l:
        continue
    idx = mlines.index(l)
    ctx = "\n".join(mlines[max(0, idx - 2): idx + 1])
    prohibited = any(k in ctx for k in ["禁写", "不得写", "禁止写", "勿写", "不能写"])
    for b in BAN:
        if b in l:
            neg_check("禁写措辞 lint: " + b, not prohibited, l.strip()[:70])

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 78)
print("POS（应成立）: %d 通过 / %d 失败" % (len(pos_pass), len(pos_fail)))
for n, d in pos_fail:
    print("  FAIL  %s :: %s" % (n, d))
print("NEG（应不成立）: %d 通过 / %d 失败" % (len(neg_pass), len(neg_fail)))
for n, d in neg_fail:
    print("  FAIL  %s :: %s" % (n, d))
ok = (len(pos_fail) == 0) and (len(neg_fail) == 0)
print("VERDICT: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
