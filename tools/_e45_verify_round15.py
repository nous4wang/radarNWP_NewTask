"""E45 第 15 次独立复核：从原始产物重算 S3 裁决 / 3-seed 统计 / 机制读数 / md5 一致性。

设计约束（沿用历轮教训，不继承任何旧脚本函数）：
  1) 不 import 旧复核脚本 —— 避免旧口径 bug 静默传染；
  2) 结构先行 —— 先打印产物真实键集，再自适应取键；
  3) 键名比对两端都小写（否则 `mean_out_equals_T` 恒取 None，伪装成"读数缺失"）；
  4) 臂名按路径段 token 精确匹配（子串匹配会让 A4b 串台到 A4）；
  5) 预注册的负结果走独立 NEG 通道，不刷红 VERDICT；
  6) S3 的 CI 键为 point/ci_lo/ci_hi/crosses_zero，并同时重算与自带标志比对。
"""
import hashlib
import json
import os
import statistics

BASE = os.path.join("tools", "_evals", "e45")
FLOOR = 0.0924
ORACLE = 0.1653
N_SEEDS = 3
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]

# 本轮实测远端 md5（remote: md5sum runs/e45/*.json）
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

fails = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        fails.append(msg)
        print(f"  FAIL  {msg}")


def neg(cond, msg):
    """预注册负结果：cond 为真表示与预注册负结果一致（NEG），不计入 VERDICT 失败。"""
    if cond:
        print(f"  NEG   {msg}  <- 与预注册负结果一致")
    else:
        fails.append(f"{msg}（与预注册负结果不一致）")
        print(f"  FAIL  {msg}  <- 与预注册负结果不一致")


def load(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return json.load(f)


def pick(d, *subs, default=None):
    """键名与待查子串**两端都小写**，取第一个全部命中的值。"""
    if not isinstance(d, dict):
        return default
    subs_l = [str(s).lower() for s in subs]
    for k, v in d.items():
        kl = str(k).lower()
        if all(s in kl for s in subs_l):
            return v
    return default


def walk_dicts(o, path=""):
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from walk_dicts(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk_dicts(v, f"{path}[{i}]")


def arm_of(path):
    """路径段按 '_' 切分，首 token 精确等于臂名才算命中。"""
    for seg in path.split("/"):
        if not seg:
            continue
        tok = seg.split("_")[0].split("[")[0]
        if tok in ARMS:
            return tok
    return None


def seed_of(path):
    for i in range(N_SEEDS):
        for pat in (f"_s{i}", f"seed_{i}", f"seed{i}"):
            if pat in path:
                return i
    return None


def stat_tuple(block):
    if not isinstance(block, dict):
        return None, None, None, None
    point = pick(block, "point")
    if point is None:
        point = pick(block, "gain")
    lo = pick(block, "ci_lo")
    if lo is None:
        lo = pick(block, "low")
    hi = pick(block, "ci_hi")
    if hi is None:
        hi = pick(block, "high")
    cz = pick(block, "cross")
    return point, lo, hi, cz


def crosses(lo, hi):
    if lo is None or hi is None:
        return None
    return (lo <= 0.0 <= hi) or (lo >= 0.0 >= hi)


print("=== [0] 产物 md5 远端/本地一致性 ===")
local_md5 = {}
for name, exp in REMOTE_MD5.items():
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        check(False, f"本地缺少 {name}")
        continue
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    local_md5[name] = h
    check(h == exp, f"{name} md5 一致 ({h[:12]}…)")
extra = sorted(set(n for n in os.listdir(BASE) if n.endswith(".json")) - set(REMOTE_MD5))
check(not extra, f"本地无多余 json（多出: {extra}）")

print("\n=== [1] S3 等偏置对照独立重算 ===")
s3 = load("s3_equalbias.json")
print("  S3 顶层键:", list(s3.keys()))
print("  S3 元信息: control=%s n_boot=%s thr=%s epoch=%s rng=%s" % (
    s3.get("control"), s3.get("n_boot"), s3.get("threshold_mm"),
    s3.get("epoch"), s3.get("rng_seed")))

arm_rows = {}
for path, d in walk_dicts(s3):
    ks = {str(k).lower() for k in d}
    if not (any("raw" in k for k in ks) and any("equal" in k for k in ks)):
        continue
    a = arm_of(path)
    if a is None:
        continue
    raw = pick(d, "raw")
    eq = pick(d, "equal_bias")
    if eq is None:
        eq = pick(d, "equalbias")
    rp, rl, rh, rc = stat_tuple(raw)
    ep, el, eh, ec = stat_tuple(eq)
    arm_rows[a] = dict(verdict=pick(d, "s3_verdict"), raw=(rp, rl, rh, rc),
                       eq=(ep, el, eh, ec), raw_xz=crosses(rl, rh), eq_xz=crosses(el, eh))

print("  S3 覆盖的臂:", sorted(arm_rows.keys()))
for a in ["A1", "A2", "A3a", "A3b", "A4", "A4b"]:
    r = arm_rows.get(a)
    if r is None:
        check(False, f"{a} 在 S3 产物中缺失")
        continue
    (rp, rl, rh, rc), (ep, el, eh, ec) = r["raw"], r["eq"]
    print("  %-4s raw=(%+.6f, [%+.6f, %+.6f]) xz=%s flag=%s | eq=(%+.6f, [%+.6f, %+.6f]) xz=%s flag=%s | %s" % (
        a, rp, rl, rh, r["raw_xz"], rc, ep, el, eh, r["eq_xz"], ec, r["verdict"]))
    check(r["raw_xz"] is True, f"{a} 原始增益 CI 跨零")
    check(r["eq_xz"] is True, f"{a} 配平后增益 CI 跨零")
    if rc is not None:
        check(bool(rc) == bool(r["raw_xz"]), f"{a} raw crosses_zero 标志与重算一致")
    if ec is not None:
        check(bool(ec) == bool(r["eq_xz"]), f"{a} equal_bias crosses_zero 标志与重算一致")

VERDICT_MAP = {"A1": "GAIN_WAS_AREA_OR_AMPLITUDE", "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE",
               "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE", "A4": "GAIN_WAS_AREA_OR_AMPLITUDE",
               "A2": "NO_GAIN_TO_BEGIN_WITH", "A4b": "NO_GAIN_TO_BEGIN_WITH"}
for a, exp in VERDICT_MAP.items():
    if a in arm_rows:
        check(arm_rows[a]["verdict"] == exp,
              f"{a} 判词 = {exp}（实际 {arm_rows[a]['verdict']}）")

print("\n=== [2] 3-seed 中位数 / 过地板通过率（地板 %.4f, Oracle %.4f）===" % (FLOOR, ORACLE))
sa = load("scores_all.json")
print("  scores_all 顶层键:", list(sa.keys()))
recs = {}
for path, d in walk_dicts(sa):
    ks = {str(k).lower() for k in d}
    if not (any("csi" in k for k in ks) and any("bias" in k for k in ks)):
        continue
    a, s = arm_of(path), seed_of(path)
    if a and s is not None:
        recs[(a, s)] = d
print("  scores_all 记录数:", len(recs))


def csi_at(d, thr):
    v = pick(d, "csi_at", thr)
    if v is None:
        c = pick(d, "csi")
        if isinstance(c, dict):
            v = c.get(thr)
    return v


def bias_at(d, thr):
    v = pick(d, "bias_at", thr)
    if v is None:
        c = pick(d, "bias")
        if isinstance(c, dict):
            v = c.get(thr)
    return v


stats = {}
for a in ARMS:
    csis, biases = [], []
    for s in range(N_SEEDS):
        d = recs.get((a, s))
        if d is None:
            continue
        c, b = csi_at(d, "2.0"), bias_at(d, "2.0")
        if isinstance(c, (int, float)):
            csis.append(c)
        if isinstance(b, (int, float)):
            biases.append(b)
    if not csis:
        check(False, f"{a} 无 3-seed 记录")
        continue
    med = statistics.median(csis)
    stats[a] = dict(med=med, lo=min(csis), hi=max(csis), n=len(csis),
                    npass=sum(1 for c in csis if c > FLOOR),
                    n_over=sum(1 for c in csis if c > ORACLE),
                    bias_med=(statistics.median(biases) if biases else None))
    s = stats[a]
    print("  %-4s median=%.4f range=%.4f-%.4f pass=%d/%d bias_med=%.4f over_oracle=%d/%d" % (
        a, med, min(csis), max(csis), s["npass"], len(csis), s["bias_med"], s["n_over"], len(csis)))
    check(len(csis) == N_SEEDS, f"{a} 有 3 个 seed 的 CSI 记录")

for a in ARMS:
    if a in stats:
        check(stats[a]["npass"] >= 2, f"{a} 至少 2/3 seed 过地板 {FLOOR}")
hi_med = [a for a in stats if a != "B0" and stats[a]["med"] > stats["B0"]["med"]]
print("  中位数高于对照 B0 的臂:", hi_med)
for a in hi_med:
    check(stats[a]["bias_med"] > stats["B0"]["bias_med"],
          f"{a} 中位 BIAS({stats[a]['bias_med']:.4f}) 高于对照({stats['B0']['bias_med']:.4f}) ⇒ 疑过预报")
check(stats["B0"]["npass"] < 3, "对照 B0 自己并非 3/3 过地板 ⇒ 地板口径不稳")

ss = load("scores_summary.json")
print("  scores_summary 顶层键:", list(ss.keys()))
mismatch = []
for a in ARMS:
    if a not in ss or a not in stats:
        continue
    blk = ss[a]
    med_ss = pick(blk, "median")
    if isinstance(med_ss, dict):
        med_ss = pick(med_ss, "csi_at", "2.0")
        if med_ss is None:
            med_ss = pick(med_ss, "2.0")
    if isinstance(med_ss, (int, float)) and abs(med_ss - stats[a]["med"]) > 1e-6:
        mismatch.append((a, med_ss, stats[a]["med"]))
check(not mismatch, f"scores_summary 与 scores_all 的中位数互校一致（不一致: {mismatch}）")

print("\n=== [3] 机制读数逐臂逐 seed 重算（读数在 arms/<臂>/readouts/）===")
probe = {}
for s in range(N_SEEDS):
    p = load(f"diag_probe_s{s}.json")
    probe[s] = p.get("arms", p)
print("  探针 arms 键:", list(probe[0].keys()), "（B0 为对照，只在 S3 中作 control 出现）")


def readout(arm, seed, key):
    a = probe.get(seed, {}).get(arm)
    if not isinstance(a, dict):
        return None
    r = a.get("readouts")
    if not isinstance(r, dict):
        r = a
    return pick(r, key)


def row(arm, key):
    vals = [readout(arm, s, key) for s in range(N_SEEDS)]
    txt = " / ".join("None" if not isinstance(v, (int, float)) else f"{v:+.4f}" for v in vals)
    return vals, txt


print("  -- 门指向（A1/A2）--")
for a in ["A1", "A2"]:
    v, t = row(a, "corr_mask_amp")
    print("   %s corr(mask,amp)      = %s   (判据 >=0)" % (a, t))
    check(all(isinstance(x, (int, float)) and x >= 0 for x in v), f"{a} corr(mask,amp) 全部 >=0")
    v, t = row(a, "mask_at_peak")
    print("   %s mask_at_peak       = %s   (判据 >0.5)" % (a, t))
    check(all(isinstance(x, (int, float)) and x > 0.5 for x in v), f"{a} mask_at_peak 全部 >0.5")
    v, t = row(a, "maxratio_as_trained")
    print("   %s maxratio_as_trained = %s   (读数缺失/缺陷)" % (a, v))
    neg(all(x is None or (isinstance(x, (int, float)) and x != x) for x in v),
        f"{a} maxratio_as_trained 应为缺失读数（None/NaN）（实际 {v}）")

print("  -- 拓扑支（A3a/A3b）--")
for a in ["A3a", "A3b"]:
    v, t = row(a, "grad_top1pct_share")
    g_ok = all(isinstance(x, (int, float)) and x < 0.5 for x in v)
    print("   %s grad_top1pct_share = %s   (判据 <0.5)" % (a, t))
    v2, t2 = row(a, "field_max_mm")
    f_ok = all(isinstance(x, (int, float)) and x <= 30 for x in v2)
    print("   %s field_max_mm       = %s   (判据 <=30)" % (a, t2))
    neg(not g_ok, f"{a} 梯度集中判据应不成立（实际 {t}）")
    neg(not f_ok, f"{a} 场最大值判据应不成立（实际 {t2}）")
    check(not (g_ok and f_ok), f"{a} 整体不过预注册判据 ⇒ 拓扑支结掉")

print("  -- 量通道（A4/A4b）--")
for a in ["A4", "A4b"]:
    v, t = row(a, "scale_cv")
    print("   %s scale_cv          = %s   (判据 >=0.05)" % (a, t))
    check(all(isinstance(x, (int, float)) and x >= 0.05 for x in v), f"{a} scale_cv 全部 >=0.05")
    v, t = row(a, "scale_truth_corr")
    corr_ok = all(isinstance(x, (int, float)) and x >= 0.3 for x in v)
    print("   %s scale_truth_corr  = %s   (判据 >=0.3)" % (a, t))
    v_const, t_const = row(a, "scale_is_constant")
    print("   %s scale_is_constant = %s" % (a, v_const))
    const_ok = all(x is False for x in v_const)
    if a == "A4":
        check(corr_ok, "A4 scale_truth_corr 3/3 >=0.3（预注册正结果）")
        check(const_ok, "A4 量通道三 seed 均未退化成常数（预注册正结果）")
    else:
        neg(not corr_ok, f"A4b scale_truth_corr 应有 seed 不过 0.3（实际 {t}）")
        neg(not const_ok, f"A4b 量通道应有 seed 退化成常数（实际 {v_const}）")
    v, t = row(a, "mean_out_equals_T")
    print("   %s mean_out_equals_T = %s   (判据 True)" % (a, v))
    neg(not all(x is True for x in v), f"{a} mean(out)=T 应不成立（实际 {v}）⇒ 实现缺陷")

print("\n=== VERDICT: " + ("PASS" if not fails else "FAIL") + " ===")
for f in fails:
    print("  FAILED CHECK:", f)
