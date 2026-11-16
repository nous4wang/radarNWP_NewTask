"""E45 第14次独立复核（v2）：从原始产物重算 S3 裁决 / 3-seed 统计 / 机制读数。

修正上一版两处自身 bug：
  1) pick() 把待查子串小写化（此前只小写化键名，导致 `mean_out_equals_T` 恒取 None）；
  2) arm_of() 改为 token 精确匹配（此前用子串匹配，`A4b` 被 `/A4` 误吞、A4/A4b 记录串台）；
  3) 判词键显式取 `s3_verdict`（产物真实键名）。
不 import 任何旧复核脚本，结构自适应取键，取不到时打印真实结构。
"""
import json
import os
import statistics

BASE = os.path.join("tools", "_evals", "e45")
FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
N_SEEDS = 3


def load(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return json.load(f)


def pick(d, *subs, default=None):
    """按键名子串（全部命中，忽略大小写）取第一个值。"""
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
    """token 精确匹配：路径段按 '_' 切分，首 token 等于臂名才算命中。"""
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


fails = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        fails.append(msg)
        print(f"  FAIL  {msg}")


def neg(cond_neg_holds, msg):
    """预注册的负结果（判据应不成立）——cond_neg_holds 为真时记 NEG，不计入 VERDICT 失败。"""
    if cond_neg_holds:
        print(f"  NEG   {msg}  ← 与预注册负结果一致")
    else:
        fails.append(f"{msg}（与预注册负结果不一致）")
        print(f"  FAIL  {msg}  ← 与预注册负结果不一致")


# ============================ 1. S3 等偏置裁决 ============================
print("=== [1] S3 等偏置对照独立重算 ===")
s3 = load("s3_equalbias.json")
print("  S3 顶层键:", list(s3.keys()), "| control =", s3.get("control"),
      "| n_boot =", s3.get("n_boot"), "| thr =", s3.get("threshold_mm"),
      "| epoch =", s3.get("epoch"), "| rng =", s3.get("rng_seed"))

arm_rows = {}
for path, d in walk_dicts(s3):
    ks = {str(k).lower() for k in d}
    if any("raw" in k for k in ks) and any("equal" in k for k in ks):
        a = arm_of(path)
        if a is None:
            continue
        raw = pick(d, "raw")
        eq = pick(d, "equal_bias")
        if eq is None:
            eq = pick(d, "equalbias")
        rp, rl, rh, rc = stat_tuple(raw)
        ep, el, eh, ec = stat_tuple(eq)
        verdict = pick(d, "s3_verdict")
        arm_rows[a] = dict(verdict=verdict, raw=(rp, rl, rh, rc), eq=(ep, el, eh, ec),
                            raw_xz=crosses(rl, rh), eq_xz=crosses(el, eh))

print("  S3 覆盖的臂:", sorted(arm_rows.keys()))
for a in ["A1", "A2", "A3a", "A3b", "A4", "A4b"]:
    r = arm_rows.get(a)
    if r is None:
        print(f"  FAIL  {a} 在 S3 产物中缺失")
        fails.append(f"{a} S3 缺失")
        continue
    (rp, rl, rh, rc), (ep, el, eh, ec) = r["raw"], r["eq"]
    print(f"  {a:4s} raw=({rp:+.6f}, [{rl:+.6f}, {rh:+.6f}]) xz={r['raw_xz']} flag={rc} | "
          f"eq=({ep:+.6f}, [{el:+.6f}, {eh:+.6f}]) xz={r['eq_xz']} flag={ec} | {r['verdict']}")
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
              f"{a} 判词={exp}（实际 {arm_rows[a]['verdict']}）")

# ============================ 2. 3-seed 分数统计 ============================
print("\n=== [2] 3-seed 中位数 / 过地板通过率 ===")
sa = load("scores_all.json")
recs = {}
for path, d in walk_dicts(sa):
    ks = {str(k).lower() for k in d}
    if any("csi" in k for k in ks) and any("bias" in k for k in ks):
        a = arm_of(path)
        s = seed_of(path)
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
        print(f"  FAIL  {a} 无 3-seed 记录")
        fails.append(f"{a} 无 3-seed 记录")
        continue
    med = statistics.median(csis)
    stats[a] = dict(med=med, lo=min(csis), hi=max(csis), n=len(csis),
                    npass=sum(1 for c in csis if c > FLOOR),
                    n_over=sum(1 for c in csis if c > ORACLE),
                    bias_med=(statistics.median(biases) if biases else None))
    s = stats[a]
    print(f"  {a:4s} median={med:.4f} range={min(csis):.4f}-{max(csis):.4f} "
          f"pass={s['npass']}/{len(csis)} bias_med={s['bias_med']:.4f} "
          f"over_oracle={s['n_over']}/{len(csis)}")

for a in ARMS:
    if a in stats:
        check(stats[a]["npass"] >= 2, f"{a} 至少 2/3 seed 过地板 {FLOOR}")
hi_med = [a for a in stats if a != "B0" and stats[a]["med"] > stats["B0"]["med"]]
print("  中位数高于对照 B0 的臂:", hi_med)
for a in hi_med:
    check(stats[a]["bias_med"] > stats["B0"]["bias_med"], f"{a} 中位 BIAS 高于对照（爆表⇒疑过预报）")
check(stats["B0"]["npass"] < 3, "对照 B0 自己并非 3/3 过地板（地板口径不稳）")

# 与 scores_summary.json 互校
ss = load("scores_summary.json")
print("  scores_summary 顶层键:", list(ss.keys()))
mismatch = []
for a in ARMS:
    if a not in ss or a not in stats:
        continue
    blk = ss[a]
    med_ss = pick(blk, "median")
    if isinstance(med_ss, dict):
        med_ss = pick(med_ss, "csi_at", "2.0") or pick(med_ss, "2.0")
    if isinstance(med_ss, (int, float)) and abs(med_ss - stats[a]["med"]) > 1e-6:
        mismatch.append((a, med_ss, stats[a]["med"]))
check(not mismatch, f"scores_summary 与 scores_all 的中位数互校一致（不一致: {mismatch}）")

# ============================ 3. 机制读数逐臂逐 seed ============================
print("\n=== [3] 机制读数逐臂逐 seed 重算（读数在 arms/<臂>/readouts/） ===")
probe = {}
for s in range(N_SEEDS):
    p = load(f"diag_probe_s{s}.json")
    probe[s] = p.get("arms", p)
print("  探针 arms 键:", list(probe[0].keys()), "（B0 为对照，只在 S3 中出现）")


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
    print(f"   {a} corr(mask,amp)   = {t}   (判据 >=0)")
    check(all(isinstance(x, (int, float)) and x >= 0 for x in v), f"{a} corr(mask,amp) 全部 >=0")
    v, t = row(a, "mask_at_peak")
    print(f"   {a} mask_at_peak    = {t}   (判据 >0.5)")
    check(all(isinstance(x, (int, float)) and x > 0.5 for x in v), f"{a} mask_at_peak 全部 >0.5")
    v, t = row(a, "maxratio_as_trained")
    print(f"   {a} maxratio_as_trained = {v}   (读数缺失)")

print("  -- 拓扑支（A3a/A3b）--")
for a in ["A3a", "A3b"]:
    v, t = row(a, "grad_top1pct_share")
    print(f"   {a} grad_top1pct_share = {t}   (判据 <0.5)")
    g_ok = all(isinstance(x, (int, float)) and x < 0.5 for x in v)
    v2, t2 = row(a, "field_max_mm")
    print(f"   {a} field_max_mm       = {t2}   (判据 <=30)")
    f_ok = all(isinstance(x, (int, float)) and x <= 30 for x in v2)
    neg(not g_ok, f"{a} 梯度集中判据应不成立（实际 {t}）")
    neg(not f_ok, f"{a} 场最大值判据应不成立（实际 {t2}）")
    check(not (g_ok and f_ok), f"{a} 整体不过预注册判据 ⇒ 拓扑支结掉")

print("  -- 量通道（A4/A4b）--")
for a in ["A4", "A4b"]:
    v, t = row(a, "scale_cv")
    print(f"   {a} scale_cv         = {t}   (判据 >=0.05)")
    check(all(isinstance(x, (int, float)) and x >= 0.05 for x in v), f"{a} scale_cv 全部 >=0.05")
    v, t = row(a, "scale_truth_corr")
    print(f"   {a} scale_truth_corr = {t}   (判据 >=0.3)")
    corr_ok = all(isinstance(x, (int, float)) and x >= 0.3 for x in v)
    v_const, t_const = row(a, "scale_is_constant")
    print(f"   {a} scale_is_constant = {v_const}")
    const_ok = all(x is False for x in v_const)
    if a == "A4":
        check(corr_ok, "A4 scale_truth_corr 3/3 >=0.3（预注册正结果）")
        check(const_ok, "A4 量通道三 seed 均未退化成常数（预注册正结果）")
    else:
        neg(not corr_ok, f"A4b scale_truth_corr 应有 seed 不过 0.3（实际 {t}）")
        neg(not const_ok, f"A4b 量通道应有 seed 退化成常数（实际 {v_const}）")
    v, t = row(a, "mean_out_equals_T")
    print(f"   {a} mean_out_equals_T = {v}   (判据 True)")
    neg(not all(x is True for x in v), f"{a} mean(out)=T 应不成立（实际 {v}）⇒ 实现缺陷")

print("\n=== VERDICT: " + ("PASS" if not fails else "FAIL") + " ===")
for f in fails:
    print("  FAILED CHECK:", f)
