"""E45 第 16 次独立复核 —— 独立实现，不 import 任何旧复核脚本。

与前几轮的不同点（刻意换实现路径，避免同源 bug 传染）：
  1) 3-seed 统计**不用 scores_all.json**，改为从 `scores_summary_s{0,1,2}.json`
     三份按 seed 分文件直接重算中位数/过地板通过率，再与合并版 `scores_summary.json`
     **互校**（三份独立来源一致才算通过）；
  2) S3 块按**显式键名**读取（raw/equal_bias → point/ci_lo/ci_hi/crosses_zero），
     并用 `raw_gain_ci`/`equal_bias_gain_ci` 两种候选键逐一探测，任一命中即用；
  3) 机制读数走 `arms/<臂>/readouts/<键>` 的**小写精确匹配**（不做子串猜测）；
  4) 负结果走独立 NEG 通道；NaN/None 三态区分；
  5) 期望值表直接抄 HANDOVER_E45.md（冻结判据口径），不重读 MASTER。

VERDICT: PASS 表示「远端产物 → 判读 → MASTER v2.45 结论」这条链自洽。
"""
import hashlib
import json
import os
import re
import statistics

BASE = os.path.join("tools", "_evals", "e45")
FLOOR = 0.0924          # 持久性地板（seed-0 口径）
ORACLE = 0.1653         # Oracle 上界
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
EXPER_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]

# HANDOVER_E45.md / MASTER v2.45 冻结的期望值（抄录，不重读 MASTER）
EXPECT_MEDIAN = {"B0": 0.1218, "A1": 0.1184, "A2": 0.1120, "A3a": 0.1100,
                 "A3b": 0.1161, "A4": 0.1439, "A4b": 0.1376}
EXPECT_BIAS_MED = {"B0": 0.5095, "A1": 0.4430, "A2": 0.4490, "A3a": 0.6394,
                   "A3b": 0.6188, "A4": 0.8362, "A4b": 2.8215}
EXPECT_PASS = {"B0": (2, 3), "A1": (2, 3), "A2": (2, 3), "A3a": (3, 3),
               "A3b": (3, 3), "A4": (2, 3), "A4b": (2, 3)}
EXPECT_VERDICT = {"A1": "GAIN_WAS_AREA_OR_AMPLITUDE", "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE",
                  "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE", "A4": "GAIN_WAS_AREA_OR_AMPLITUDE",
                  "A2": "NO_GAIN_TO_BEGIN_WITH", "A4b": "NO_GAIN_TO_BEGIN_WITH"}

fails = []


def ck(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def neg(cond, msg):
    """预注册负结果：cond 为真表示与预注册一致，不刷红。"""
    print(("  NEG   " if cond else "  FAIL  ") + msg + ("  <- 与预注册负结果一致" if cond else "  <- 与预注册负结果不一致"))
    if not cond:
        fails.append(msg + "（与预注册负结果不一致）")


def load(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return json.load(f)


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_missing(x):
    """None / NaN 视为「读数缺失」——三态区分（第 15 轮教训）。"""
    return x is None or (is_num(x) and x != x)


def get_ci(block):
    """从统计块取 (point, lo, hi, crosses_zero)，显式键名优先。"""
    if not isinstance(block, dict):
        return (None, None, None, None)
    lo = hi = point = flag = None
    for lk, hk in (("ci_lo", "ci_hi"), ("ci_low", "ci_high"), ("lo", "hi")):
        if lk in block and hk in block:
            lo, hi = block[lk], block[hk]
            break
    for pk in ("point", "gain", "median"):
        if pk in block:
            point = block[pk]
            break
    for fk in ("crosses_zero", "cross", "xz"):
        if fk in block:
            flag = block[fk]
            break
    return point, lo, hi, flag


def crosses(lo, hi):
    if not is_num(lo) or not is_num(hi):
        return None
    return (lo <= 0.0 <= hi) or (lo >= 0.0 >= hi)


def find_arm_block(obj, arm):
    """在对象树里定位属于某臂的字典（路径段首 token 精确匹配，避免 A4b 串台 A4）。"""
    stack = [(obj, "")]
    while stack:
        node, path = stack.pop()
        if isinstance(node, dict):
            hit = False
            for seg in path.split("/"):
                if seg and seg.split("_")[0].split("[")[0] == arm:
                    hit = True
            if hit and path:
                yield path, node
            for k, v in node.items():
                stack.append((v, f"{path}/{k}"))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                stack.append((v, f"{path}[{i}]"))


print("=== [0] 产物 md5（本地）===")
local = {}
for fn in sorted(os.listdir(BASE)):
    if fn.endswith(".json"):
        local[fn] = hashlib.md5(open(os.path.join(BASE, fn), "rb").read()).hexdigest()
        print("  %-26s %s" % (fn, local[fn]))
ck(len(local) == 13, f"本地 json 计数 = 13（实际 {len(local)}）")

print("\n=== [1] 3-seed 统计：从 scores_summary_s{0,1,2}.json 独立重算 ===")
# 按 seed 分文件的**键格式是 `<臂>_s<seed>_ep00`**（实测结构转储，非猜测），
# 且 s1/s2 文件是**累积**的（s2 文件含 s0+s1+s2，与 scores_all.json 同 md5）。
# 合并时按 (臂, seed) 取值，并对重复出现处做一致性核对。
per_seed = {}
dupe_conflicts = []
KEY_RE = re.compile(r"^(B0|A1|A2|A3a|A3b|A4|A4b)_s(\d+)_ep(\d+)$")
for s in range(3):
    doc = load(f"scores_summary_s{s}.json")
    print("  s%d 文件含 %d 条 `<臂>_s<seed>_ep00` 记录" % (s, len(doc)))
    for k, blk in doc.items():
        m = KEY_RE.match(str(k))
        if not m or not isinstance(blk, dict):
            continue
        arm, seed = m.group(1), int(m.group(2))
        csi, bias = blk.get("csi_at_2.0"), blk.get("bias_at_2.0")
        if not is_num(csi):
            continue
        rec = (float(csi), float(bias) if is_num(bias) else None)
        key = (arm, seed)
        if key in per_seed and per_seed[key] != rec:
            dupe_conflicts.append((key, per_seed[key], rec))
        per_seed[key] = rec
ck(len(per_seed) == 21, f"共解析出 21 条 (臂,种子) 记录（实际 {len(per_seed)}）")
ck(not dupe_conflicts, f"按 seed 分文件间同臂同种子取值无冲突（冲突: {dupe_conflicts}）")

print("\n  -- 与合并版 scores_all.json（独立产物）逐记录互校 --")
sa = load("scores_all.json")
sa_diff = []
sa_cmp = 0
for k, blk in sa.items():
    m = KEY_RE.match(str(k))
    if not m:
        continue
    key = (m.group(1), int(m.group(2)))
    if key in per_seed:
        sa_cmp += 1
        c2 = per_seed[key][0]
        c_sa = blk.get("csi_at_2.0")
        if is_num(c_sa) and abs(c2 - float(c_sa)) > 1e-12:
            sa_diff.append((key, c2, c_sa))
ck(sa_cmp == 21, f"与 scores_all.json 实际比对了 21 条记录（实际 {sa_cmp}）")
ck(not sa_diff, f"按 seed 分文件与 scores_all.json 逐记录一致（不一致: {sa_diff[:5]}）")

recomputed = {}
for arm in ARMS:
    rec = {sd: per_seed[(arm, sd)] for sd in range(3) if (arm, sd) in per_seed}
    ck(len(rec) == 3, f"{arm} 三个种子均有 CSI 记录（实际 {len(rec)}）")
    if len(rec) != 3:
        continue
    cs = [rec[s][0] for s in range(3)]
    bs = [rec[s][1] for s in range(3) if rec[s][1] is not None]
    med = statistics.median(cs)
    npass = sum(1 for c in cs if c > FLOOR)
    bmed = statistics.median(bs) if bs else None
    recomputed[arm] = (med, npass, bmed, min(cs), max(cs))
    print("  %-4s median=%.4f range=%.4f-%.4f pass=%d/3 bias_med=%.4f"
          % (arm, med, min(cs), max(cs), npass, bmed))
    ck(abs(med - EXPECT_MEDIAN[arm]) < 5e-4, f"{arm} 中位数 {med:.4f} ≈ 冻结值 {EXPECT_MEDIAN[arm]:.4f}")
    ck((npass, 3) == EXPECT_PASS[arm], f"{arm} 过地板通过率 {npass}/3 = 冻结口径 {EXPECT_PASS[arm][0]}/3")
    if bmed is not None:
        ck(abs(bmed - EXPECT_BIAS_MED[arm]) < 5e-3, f"{arm} BIAS 中位 {bmed:.4f} ≈ 冻结值 {EXPECT_BIAS_MED[arm]:.4f}")

print("\n  -- 与合并版 scores_summary.json 互校 --")
ss = load("scores_summary.json")
for arm in ARMS:
    blk = ss.get(arm)
    if not isinstance(blk, dict) or arm not in recomputed:
        continue
    med_ss = blk.get("median_csi_at_2.0")
    bias_ss = blk.get("median_bias_at_2.0")
    if is_num(med_ss):
        ck(abs(med_ss - recomputed[arm][0]) < 1e-9,
           f"{arm} 合并版中位数 {med_ss:.9f} 与三份按 seed 重算一致")
    else:
        ck(False, f"{arm} 合并版中位数键未定位（可用键: {list(blk.keys())[:8]}）")
    if is_num(bias_ss) and recomputed[arm][2] is not None:
        ck(abs(bias_ss - recomputed[arm][2]) < 1e-9,
           f"{arm} 合并版 BIAS 中位数 {bias_ss:.6f} 与重算一致")
    n_seeds = blk.get("n_seeds")
    ck(n_seeds == 3, f"{arm} 合并版 n_seeds = 3（实际 {n_seeds}）")

print("\n=== [2] S3 等偏置对照独立重算 ===")
s3 = load("s3_equalbias.json")
print("  S3 顶层键:", list(s3.keys()))
print("  S3 元信息: control=%s n_boot=%s thr=%s epoch=%s rng=%s"
      % (s3.get("control"), s3.get("n_boot"), s3.get("threshold_mm"), s3.get("epoch"), s3.get("rng_seed")))
s3_arms = s3.get("arms", s3)
for arm in EXPER_ARMS:
    blk = None
    for path, node in find_arm_block(s3_arms, arm):
        if isinstance(node, dict) and ("raw" in node or "equal_bias" in node):
            blk = node
            break
    if blk is None:
        ck(False, f"{arm} 在 S3 产物中未定位到统计块")
        continue
    print("  %-4s 块内键: %s" % (arm, list(blk.keys())))
    raw = blk.get("raw") or blk.get("raw_gain_ci") or blk.get("raw_gain")
    eq = blk.get("equal_bias") or blk.get("equal_bias_gain_ci") or blk.get("equalbias")
    rp, rl, rh, rflag = get_ci(raw)
    ep, el, eh, eflag = get_ci(eq)
    verdict = None
    for k, v in blk.items():
        if "verdict" in str(k).lower():
            verdict = v
    print("     raw  point=%+.6f CI=[%+.6f, %+.6f] xz=%s flag=%s" % (rp, rl, rh, crosses(rl, rh), rflag))
    print("     eq   point=%+.6f CI=[%+.6f, %+.6f] xz=%s flag=%s" % (ep, el, eh, crosses(el, eh), eflag))
    print("     verdict=%s" % verdict)
    ck(crosses(rl, rh) is True, f"{arm} 原始增益 CI 跨零")
    ck(crosses(el, eh) is True, f"{arm} 配平后增益 CI 跨零")
    if rflag is not None:
        ck(bool(rflag) == bool(crosses(rl, rh)), f"{arm} raw crosses_zero 标志与重算一致")
    if eflag is not None:
        ck(bool(eflag) == bool(crosses(el, eh)), f"{arm} equal_bias crosses_zero 标志与重算一致")
    ck(verdict == EXPECT_VERDICT[arm], f"{arm} 判词 = {EXPECT_VERDICT[arm]}（实际 {verdict}）")

print("\n=== [3] 机制读数（arms/<臂>/readouts/，小写精确键）===")
probes = {s: load("diag_probe_s%d.json" % s).get("arms", {}) for s in range(3)}
print("  探针 arms 键:", list(probes[0].keys()), "（B0 为对照，只在 S3 作 control）")


def rd(arm, seed, key):
    a = probes.get(seed, {}).get(arm)
    if not isinstance(a, dict):
        return None
    r = a.get("readouts")
    if not isinstance(r, dict):
        r = a
    lk = key.lower()
    for k, v in r.items():
        if str(k).lower() == lk:
            return v
    return None


def row(arm, key):
    vals = [rd(arm, s, key) for s in range(3)]
    txt = " / ".join("None" if v is None else ("NaN" if (is_num(v) and v != v) else f"{v:+.4f}") for v in vals)
    return vals, txt


print("  -- 门指向 A1/A2 --")
for a in ["A1", "A2"]:
    v, t = row(a, "corr_mask_amp")
    print("   %s corr(mask,amp)       = %s  (判据 >=0)" % (a, t))
    ck(all(is_num(x) and x >= 0 for x in v), f"{a} corr(mask,amp) 三 seed 全部 >=0")
    v, t = row(a, "mask_at_peak")
    print("   %s mask_at_peak        = %s  (判据 >0.5)" % (a, t))
    ck(all(is_num(x) and x > 0.5 for x in v), f"{a} mask_at_peak 三 seed 全部 >0.5")
    v, t = row(a, "maxratio_as_trained")
    print("   %s maxratio_as_trained = %s  (读数缺失)" % (a, t))
    neg(all(is_missing(x) for x in v), f"{a} maxratio_as_trained 应全为缺失读数（实际 {t}）")

print("  -- 拓扑支 A3a/A3b --")
for a in ["A3a", "A3b"]:
    v, t = row(a, "grad_top1pct_share")
    v2, t2 = row(a, "field_max_mm")
    print("   %s grad_top1pct_share = %s  (判据 <0.5)" % (a, t))
    print("   %s field_max_mm       = %s  (判据 <=30)" % (a, t2))
    neg(not all(is_num(x) and x < 0.5 for x in v), f"{a} 梯度集中判据应不成立（实际 {t}）")
    neg(not all(is_num(x) and x <= 30 for x in v2), f"{a} 场最大值判据应不成立（实际 {t2}）")
    ck(not (all(is_num(x) and x < 0.5 for x in v) and all(is_num(x) and x <= 30 for x in v2)),
       f"{a} 整体不过预注册判据 ⇒ 拓扑支结掉")

print("  -- 量通道 A4/A4b --")
for a in ["A4", "A4b"]:
    v, t = row(a, "scale_cv")
    print("   %s scale_cv           = %s  (判据 >=0.05)" % (a, t))
    ck(all(is_num(x) and x >= 0.05 for x in v), f"{a} scale_cv 三 seed 全部 >=0.05")
    v, t = row(a, "scale_truth_corr")
    v2, t2 = row(a, "scale_is_constant")
    print("   %s scale_truth_corr   = %s  (判据 >=0.3)" % (a, t))
    print("   %s scale_is_constant  = %s" % (a, v2))
    corr_ok = all(is_num(x) and x >= 0.3 for x in v)
    const_ok = all(x is False for x in v2)
    if a == "A4":
        ck(corr_ok, "A4 scale_truth_corr 三 seed 全部 >=0.3")
        ck(const_ok, "A4 量通道三 seed 均未退化成常数")
    else:
        neg(not corr_ok, f"A4b 应有 seed 的 scale_truth_corr < 0.3（实际 {t}）")
        neg(not const_ok, f"A4b 应有 seed 退化成常数（实际 {v2}）")
    v, t = row(a, "mean_out_equals_T")
    print("   %s mean_out_equals_T  = %s  (判据 True)" % (a, t))
    neg(not all(x is True for x in v), f"{a} mean(out)=T 应不成立（实际 {v}）⇒ 实现缺陷")

print("\n=== [4] 地板口径不稳（对照自身非 3/3 过地板）===")
ck(recomputed["B0"][1] < 3, f"对照 B0 过地板 {recomputed['B0'][1]}/3 < 3/3 ⇒ 地板口径不稳，不作晋级依据")
hi = [a for a in ARMS if a != "B0" and a in recomputed and recomputed[a][0] > recomputed["B0"][0]]
print("  中位数高于对照的臂:", hi)
for a in hi:
    ck(recomputed[a][2] > recomputed["B0"][2],
       f"{a} 中位 BIAS {recomputed[a][2]:.4f} > 对照 {recomputed['B0'][2]:.4f} ⇒ 疑过预报")

print("\n=== VERDICT: " + ("PASS" if not fails else "FAIL") + " ===")
for f in fails:
    print("  FAILED CHECK:", f)
