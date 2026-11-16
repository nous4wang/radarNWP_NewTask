# -*- coding: utf-8 -*-
"""E45 第 17 轮独立复核（2026-11-12）。

原则（承接前十六轮经验）：
1. 不 import 旧复核脚本的函数 —— 避免旧口径 bug 静默传染；全部从原始产物重算。
2. 结构先行：先打印真实键名，再取键；键名比对两端都小写。
3. 负结果用独立通道（NEG），不与"应通过"混在同一红绿通道。
4. NaN / None / 有效值三态区分。
5. 互校断言必须同时校验"实际比对了 N 条"，N=0 直接失败。
"""
import json
import os
import re
import subprocess
import sys
import hashlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(REPO, "tools", "_evals", "e45")
FLOOR = 0.0924
ORACLE = 0.1653
ARMS6 = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ARMS7 = ["B0"] + ARMS6

ok_rows = []      # 应通过的检查
neg_rows = []     # 预注册的负结果（"该失败的没失败"要单独告警）
fails = []


def check(name, cond, detail=""):
    ok_rows.append((name, bool(cond), detail))
    if not cond:
        fails.append("POS %s | %s" % (name, detail))


def neg(name, cond, detail=""):
    """cond 为 True 表示"符合预期的负结果"（例如增益 CI 跨零）。"""
    neg_rows.append((name, bool(cond), detail))
    if not cond:
        fails.append("NEG %s | %s" % (name, detail))


def is_missing(x):
    return x is None or (isinstance(x, (int, float)) and x != x)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def lmap(d):
    """键名小写化后的查找表（两端都小写，避免 mean_out_equals_T 取不到）。"""
    return {str(k).lower(): v for k, v in d.items()}


def jload(name):
    with open(os.path.join(LOCAL, name), "r", encoding="utf-8") as f:
        return json.load(f)


print("=" * 78)
print("E45 第 17 轮独立复核 —— 2026-11-12")
print("=" * 78)

# ---------------------------------------------------------------- 1. 远端 md5
print("\n[1] 远端 md5 × 本地 md5 核对")
REMOTE_CMD = "cd /home/wf/radarNWP && md5sum runs/e45/*.json"
proc = subprocess.run(
    [sys.executable, os.path.join(REPO, "tools", "remote.py"), "--", REMOTE_CMD],
    cwd=REPO, capture_output=True, text=True,
)
remote = {}
for line in proc.stdout.splitlines():
    m = re.search(r"([0-9a-fA-F]{32})\s+(\S+\.json)\s*$", line)
    if m:
        remote[os.path.basename(m.group(2))] = m.group(1).lower()
print("  远端 json 数：%d" % len(remote))
check("remote_md5_parsed", len(remote) >= 13, "远端只解析到 %d 个 json" % len(remote))

local = {}
for fn in sorted(os.listdir(LOCAL)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(LOCAL, fn), "rb") as f:
        local[fn] = hashlib.md5(f.read()).hexdigest()
print("  本地 json 数：%d" % len(local))
check("local_md5_parsed", len(local) >= 13, "本地只 %d 个 json" % len(local))

same, diff, missing = 0, [], []
for fn, h in sorted(local.items()):
    if fn not in remote:
        missing.append(fn)
    elif remote[fn] == h:
        same += 1
    else:
        diff.append(fn)
print("  md5 相同 %d / 差异 %s / 远端缺 %s" % (same, diff or "无", missing or "无"))
check("md5_all_identical", same == len(local) and not diff and not missing,
      "same=%d diff=%s missing=%s" % (same, diff, missing))

# ---------------------------------------------------------------- 2. S3 裁决
print("\n[2] S3 等偏置对照独立重算")
s3 = jload("s3_equalbias.json")
print("  meta: threshold_mm=%s epoch=%s control=%s n_boot=%s rng_seed=%s"
      % (s3.get("threshold_mm"), s3.get("epoch"), s3.get("control"),
         s3.get("n_boot"), s3.get("rng_seed")))
check("s3_meta", s3.get("threshold_mm") == 2.0 and s3.get("control") == "B0"
      and s3.get("n_boot") == 10000, json.dumps({k: v for k, v in s3.items() if k != "arms"}))
s3_arms = s3["arms"]
print("  arms(%d): %s" % (len(s3_arms), list(s3_arms.keys())))
check("s3_has_six_arms", sorted(s3_arms.keys()) == sorted(ARMS6), str(list(s3_arms.keys())))

for arm in ARMS6:
    a = s3_arms[arm]
    raw = a["raw"]
    eb = a["equal_bias"]
    # 独立重算跨零
    raw_cz = (raw["ci_lo"] <= 0.0 <= raw["ci_hi"])
    eb_cz = (eb["ci_lo"] <= 0.0 <= eb["ci_hi"])
    check("s3_flag_raw_" + arm, raw["crosses_zero"] == raw_cz,
          "产物=%s 重算=%s" % (raw["crosses_zero"], raw_cz))
    check("s3_flag_eq_" + arm, eb["crosses_zero"] == eb_cz,
          "产物=%s 重算=%s" % (eb["crosses_zero"], eb_cz))
    neg("s3_raw_crosses_zero_" + arm, raw_cz,
        "raw CI=[%.4f, %.4f]" % (raw["ci_lo"], raw["ci_hi"]))
    neg("s3_eqbias_crosses_zero_" + arm, eb_cz,
        "eq CI=[%.4f, %.4f]" % (eb["ci_lo"], eb["ci_hi"]))
    print("  %-4s raw %+.4f CI[%+.4f,%+.4f] %s | eqbias %+.4f CI[%+.4f,%+.4f] %s | %s"
          % (arm, raw["point"], raw["ci_lo"], raw["ci_hi"],
             "跨零" if raw_cz else "不跨零",
             eb["point"], eb["ci_lo"], eb["ci_hi"],
             "跨零" if eb_cz else "不跨零", a["s3_verdict"]))
    if eb["crosses_zero"]:
        neg("s3_verdict_" + arm,
            a["s3_verdict"] in ("GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH"),
            "verdict=%s" % a["s3_verdict"])

# ---------------------------------------------------------------- 3. 3-seed 统计
print("\n[3] 3-seed 中位数 / 过地板通过率（由 scores_all.json 重算，与 scores_summary.json 互校）")
allsc = jload("scores_all.json")
print("  scores_all 键样例：%s" % list(allsc.keys())[:6])
check("scores_all_21_records", len(allsc) == 21, "只有 %d 条" % len(allsc))
per_seed = {}
for k, v in allsc.items():
    toks = k.split("_")
    check("scores_all_keyfmt", len(toks) == 3 and toks[1].startswith("s") and toks[2].startswith("ep"),
          "键名格式异常：%s" % k)
    arm = toks[0]
    seed = toks[1]
    per_seed[(arm, seed)] = v["csi_at_2.0"]
check("per_seed_21", len(per_seed) == 21, "只解析到 %d 条" % len(per_seed))

ss = jload("scores_summary.json")
print("  scores_summary 臂：%s" % list(ss.keys()))
check("scores_summary_seven_arms", sorted(ss.keys()) == sorted(ARMS7), str(list(ss.keys())))

print("  %-4s %-8s %-8s %-9s %-9s %-8s" % ("臂", "中位CSI", "过地板", "中位BIAS", "超Oracle", "离散"))
stat = {}
for arm in ARMS7:
    csi = [per_seed[(arm, "s%d" % s)] for s in range(3)]
    bias = []
    for s in range(3):
        toks = [k for k in allsc if k == "%s_s%d_ep00" % (arm, s)]
        bias.append(allsc[toks[0]]["bias_at_2.0"])
    m = median(csi)
    mb = median(bias)
    npass = sum(1 for x in csi if x >= FLOOR)
    norc = sum(1 for x in csi if x > ORACLE)
    spread = max(csi) / min(csi) if min(csi) > 0 else float("inf")
    stat[arm] = dict(median=m, npass=npass, mbias=mb, norc=norc, spread=spread,
                     csi=csi, bias=bias)
    # 与 scores_summary 互校
    s = ss[arm]
    check("ss_csi_median_" + arm, abs(s["median_csi_at_2.0"] - m) < 1e-12,
          "summary=%.6f 重算=%.6f" % (s["median_csi_at_2.0"], m))
    check("ss_bias_median_" + arm, abs(s["median_bias_at_2.0"] - mb) < 1e-12,
          "summary=%.6f 重算=%.6f" % (s["median_bias_at_2.0"], mb))
    check("ss_passrate_" + arm, s["pass_rate_over_floor"] == "%d/3" % npass,
          "summary=%s 重算=%d/3" % (s["pass_rate_over_floor"], npass))
    for sd in range(3):
        key = "s%d" % sd
        check("ss_csi_%s_%s" % (arm, key), abs(s["csi_at_2.0"][key] - csi[sd]) < 1e-12,
              "summary=%.6f 重算=%.6f" % (s["csi_at_2.0"][key], csi[sd]))
    print("  %-4s %-8.4f %-8s %-9.4f %-9s %-8.2fx"
          % (arm, m, "%d/3" % npass, mb, "%d/3" % norc, spread))

b0 = stat["B0"]["median"]
higher = [a for a in ARMS6 if stat[a]["median"] > b0]
print("  中位数高于对照 B0 的臂：%s" % (higher or "无"))
for a in higher:
    neg("higher_than_control_has_bias_blowup_" + a,
        stat[a]["mbias"] > stat["B0"]["mbias"],
        "%s 中位BIAS=%.4f vs 对照 %.4f" % (a, stat[a]["mbias"], stat["B0"]["mbias"]))
neg("control_floor_unstable", stat["B0"]["npass"] < 3,
    "对照 B0 自身过地板 %d/3（地板口径不稳）" % stat["B0"]["npass"])
neg("no_arm_full_3of3_plus_higher",
    not any(stat[a]["npass"] == 3 and stat[a]["median"] > b0 for a in ARMS6),
    "无一臂同时满足 3/3 过地板且中位数高于对照 ⇒ 不投十 seed")

# ---------------------------------------------------------------- 4. 机制读数
print("\n[4] 机制读数逐臂逐 seed 重算（diag_probe_s{0,1,2}.json，键在 arms/<臂>/readouts/）")
probes = {s: jload("diag_probe_s%d.json" % s) for s in range(3)}
print("  探针 arms(%d): %s（不含 B0，B0 只在 S3 里作 control）" %
      (len(probes[0]["arms"]), list(probes[0]["arms"].keys())))
check("probe_six_arms_no_B0", sorted(probes[0]["arms"].keys()) == sorted(ARMS6),
      str(list(probes[0]["arms"].keys())))

# 先打印真实键名（结构先行）
for arm in ("A3a", "A4"):
    print("  %s readouts 键：%s" % (arm, sorted(lmap(probes[0]["arms"][arm]["readouts"]).keys())))

def readout(arm, seed, key):
    m = lmap(probes[seed]["arms"][arm]["readouts"])
    return m.get(key.lower())

# 门：corr_mask_amp >= 0 且 mask_at_peak > 0.5
print("  -- 门指向（A1/A2）--")
for arm in ("A1", "A2"):
    for s in range(3):
        c = readout(arm, s, "corr_mask_amp")
        p = readout(arm, s, "mask_at_peak")
        check("gate_%s_s%d" % (arm, s), c is not None and c >= 0.0 and p is not None and p > 0.5,
              "corr_mask_amp=%s mask_at_peak=%s" % (c, p))
    print("    %s corr_mask_amp=%s | mask_at_peak=%s"
          % (arm, [round(readout(arm, s, "corr_mask_amp"), 4) for s in range(3)],
             [round(readout(arm, s, "mask_at_peak"), 4) for s in range(3)]))

# 拓扑支：预注册期望 A3a/A3b 均 0/3 通过（场最大值三 seed 全 402.43 mm > 30）⇒ 走负向通道
print("  -- 拓扑支（A3a/A3b）：期望 0/3 通过 ⇒ 拓扑支结掉（负向通道）--")
for arm in ("A3a", "A3b"):
    n_pass = 0
    for s in range(3):
        g = readout(arm, s, "grad_top1pct_share")
        f = readout(arm, s, "field_max_mm")
        seed_pass = (g is not None and g < 0.5 and f is not None and f <= 30.0)
        n_pass += 1 if seed_pass else 0
        neg("topo_%s_s%d_not_pass" % (arm, s), not seed_pass,
            "grad_top1pct_share=%s field_max_mm=%s" % (g, f))
    neg("topo_arm_%s_zero_of_three" % arm, n_pass == 0,
        "%s 通过 %d/3（期望 0/3）" % (arm, n_pass))
    print("    %s grad_top1pct_share=%s | field_max_mm=%s ⇒ 通过 %d/3（期望 0/3）"
          % (arm, [round(readout(arm, s, "grad_top1pct_share"), 4) for s in range(3)],
             [round(readout(arm, s, "field_max_mm"), 2) for s in range(3)], n_pass))

# 量通道：A4 期望 3/3 通过且未退化；A4b 期望 2/3（seed2 量通道退化）⇒ 混合通道
print("  -- 量通道（A4/A4b）：A4 期望 3/3；A4b 期望 2/3（seed2 退化）--")
EXPECT_AMT = {"A4": dict(n_pass=3, const=[False, False, False], seed_pass=[True, True, True]),
              "A4b": dict(n_pass=2, const=[False, False, True], seed_pass=[True, True, False])}
for arm in ("A4", "A4b"):
    exp = EXPECT_AMT[arm]
    n_pass = 0
    for s in range(3):
        cv = readout(arm, s, "scale_cv")
        tc = readout(arm, s, "scale_truth_corr")
        isc = readout(arm, s, "scale_is_constant")
        seed_pass = (cv is not None and cv >= 0.05 and tc is not None and tc >= 0.3)
        n_pass += 1 if seed_pass else 0
        # 逐 seed 期望模式比对（seed2 期望不过：A4b 量通道退化）
        neg("amt_%s_s%d_pass_pattern" % (arm, s), seed_pass == exp["seed_pass"][s],
            "scale_cv=%s scale_truth_corr=%s ⇒ 本 seed %s判据（期望 %s）"
            % (cv, tc, "" if seed_pass else "不", "过" if exp["seed_pass"][s] else "不过"))
        # scale_is_constant：期望序列由 EXPECT_AMT 给定（True=退化成常数）
        neg("amt_%s_s%d_is_constant" % (arm, s), isc is exp["const"][s],
            "scale_is_constant=%s（期望 %s）" % (isc, exp["const"][s]))
    if arm == "A4":
        check("amt_arm_A4_three_of_three", n_pass == 3, "A4 通过 %d/3（期望 3/3）" % n_pass)
    else:
        neg("amt_arm_A4b_two_of_three", n_pass == 2, "A4b 通过 %d/3（期望 2/3）" % n_pass)
    print("    %s scale_cv=%s | scale_truth_corr=%s | is_constant=%s ⇒ 通过 %d/3（期望 %d/3）"
          % (arm, [round(readout(arm, s, "scale_cv"), 4) for s in range(3)],
             [round(readout(arm, s, "scale_truth_corr"), 4) for s in range(3)],
             [readout(arm, s, "scale_is_constant") for s in range(3)],
             n_pass, exp["n_pass"]))
    # mean(out)=T：A4 s0/s2 True、s1 False；A4b 三 seed 全 False ⇒ 实现缺陷
    mot = [readout(arm, s, "mean_out_equals_T") for s in range(3)]
    neg("mean_out_equals_T_not_all_true_" + arm, not all(x is True for x in mot),
        "%s mean_out_equals_T=%s ⇒ 仍属实现缺陷" % (arm, mot))
    print("    %s mean_out_equals_T=%s" % (arm, mot))
    # maxratio_as_trained 三 seed 全 NaN（读数缺失）
    mr = [readout(arm, s, "maxratio_as_trained") for s in range(3)]
    neg("maxratio_as_trained_all_missing_" + arm, all(is_missing(x) for x in mr),
        "%s maxratio_as_trained=%s" % (arm, mr))

# 门那组 maxratio 三 seed 全空值
for arm in ("A1", "A2"):
    mr = [readout(arm, s, "maxratio_as_trained") for s in range(3)]
    neg("maxratio_as_trained_all_missing_" + arm, all(is_missing(x) for x in mr),
        "%s maxratio_as_trained=%s" % (arm, mr))
    print("    %s maxratio_as_trained=%s" % (arm, mr))

# ---------------------------------------------------------------- 5. 汇总
print("\n[5] 汇总")
n_ok = sum(1 for _, c, _ in ok_rows if c)
n_neg = sum(1 for _, c, _ in neg_rows if c)
print("  正向断言 %d/%d 通过" % (n_ok, len(ok_rows)))
print("  负向断言 %d/%d 符合预期（预注册的负结果）" % (n_neg, len(neg_rows)))
if fails:
    print("\n  失败项（%d）：" % len(fails))
    for f in fails:
        print("   - %s" % f)
    print("\nVERDICT: FAIL")
    sys.exit(1)
print("\nVERDICT: PASS")
