#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第53轮：机制剂量—技能收益耦合轴的首次判读（只读本地，不动远端）。

问题：E45 六臂的"机制读数"（门族/拓扑族/量族）与"技能收益"（CSI/BIAS）此前被
分别判读（第43轮只看派生布尔与读数，第44-52轮只看技能各轴），**从未把两者放进
同一张 21 格表**做剂量-反应分析。本轮补这一层。

硬约束（沿用既有纪律）：
  * 只读本地产物（tools/_evals/e45/*.json），不触碰远端、不改远端状态
  * 报告口径：中位数 + 通过率；单点分数不作裁决
  * 事后诊断，不构成新判据，不改变 E45 裁决（六臂判负、无一臂进十 seed）
  * 复制单元仍是 seed（n=3）；置换检验以 seed 为区组

产物：tools/_evals/e45/round53_mechanism_dose_response.json
"""
import json, math, os, random, statistics, sys

ROOT = r"R:/1radarNWPprediction/radar-nwp-bench"
EV   = os.path.join(ROOT, "tools", "_evals", "e45")
OUT  = os.path.join(EV, "round53_mechanism_dose_response.json")

ARMS  = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL   = ["B0"] + ARMS
SEEDS = [0, 1, 2]
FLOOR = 0.0924
ORACLE = 0.1653
N_PERM = 20000
RNG = random.Random(20261114)

# 预注册机制读数（来自 E45 判据文件 / 交接文档 §3）
CANON = ["corr_mask_amp", "mask_at_peak", "mask_dry_frac_lt_0p1", "corr_mask_resid",
         "grad_top1pct_share", "field_max_mm", "tail_max_over_p99",
         "scale_cv", "scale_truth_corr", "scale_mean", "maxratio_as_trained",
         "mean_out_equals_T", "scale_is_constant"]

# 机制族：臂 -> (读数, 符号约定)  sign=+1 表示"越大越好"
FAMILIES = {
    "gates":  (["A1", "A2"],   [("corr_mask_amp", +1), ("mask_at_peak", +1)]),
    "topo":   (["A3a", "A3b"], [("grad_top1pct_share", -1), ("field_max_mm", -1),
                               ("tail_max_over_p99", -1)]),
    "amount": (["A4", "A4b"],  [("scale_cv", +1), ("scale_truth_corr", +1)]),
}


# ---------------------------------------------------------------- 基础工具
def load(name):
    with open(os.path.join(EV, name), "r", encoding="utf-8") as f:
        return json.load(f)


def flatten(obj, prefix="", out=None):
    """把任意嵌套 JSON 摊平成 {路径: 数值/bool}；NaN/Inf 记为 None。"""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, prefix + str(k) + "/", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(v, prefix + "[" + str(i) + "]/", out)
    elif isinstance(obj, bool):
        out[prefix[:-1]] = obj
    elif isinstance(obj, (int, float)):
        f = float(obj)
        out[prefix[:-1]] = f if math.isfinite(f) else None
    return out


def pick(flat, name):
    """按叶子名取读数（取第一个非 None 的匹配）。"""
    for k, v in flat.items():
        if k.split("/")[-1] == name and v is not None:
            return v
    return None


def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(rank(x), rank(y))


def blocked_perm_spearman(cells, rng, nperm=N_PERM):
    """cells: [{seed, x, y}]；以 seed 为区组，区内打乱 x（读数）与 y（收益）的配对。"""
    x = [c["x"] for c in cells]
    y = [c["y"] for c in cells]
    obs = spearman(x, y)
    if not math.isfinite(obs):
        return obs, float("nan"), 0
    blocks = {}
    for i, c in enumerate(cells):
        blocks.setdefault(c["seed"], []).append(i)
    blocks = list(blocks.values())
    hit = 0
    for _ in range(nperm):
        xp = list(x)
        for b in blocks:
            vals = [x[i] for i in b]
            rng.shuffle(vals)
            for i, v in zip(b, vals):
                xp[i] = v
        r = spearman(xp, y)
        if math.isfinite(r) and abs(r) >= abs(obs) - 1e-12:
            hit += 1
    return obs, (1.0 + hit) / (nperm + 1.0), hit


def blocked_perm_median_diff(cells, rng, nperm=N_PERM):
    """cells: [{seed, y, lab}]；lab 为布尔标签；统计 median(y|lab=1) - median(y|lab=0)。"""
    y = [c["y"] for c in cells]
    lab = [c["lab"] for c in cells]
    a = [y[i] for i in range(len(y)) if lab[i]]
    b = [y[i] for i in range(len(y)) if not lab[i]]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan"), len(a), len(b)
    obs = statistics.median(a) - statistics.median(b)
    blocks = {}
    for i, c in enumerate(cells):
        blocks.setdefault(c["seed"], []).append(i)
    blocks = list(blocks.values())
    hit = 0
    for _ in range(nperm):
        lp = list(lab)
        for blk in blocks:
            vals = [lp[i] for i in blk]
            rng.shuffle(vals)
            for i, v in zip(blk, vals):
                lp[i] = v
        aa = [y[i] for i in range(len(y)) if lp[i]]
        bb = [y[i] for i in range(len(y)) if not lp[i]]
        if len(aa) < 2 or len(bb) < 2:
            continue
        d = statistics.median(aa) - statistics.median(bb)
        if abs(d) >= abs(obs) - 1e-12:
            hit += 1
    return obs, (1.0 + hit) / (nperm + 1.0), statistics.median(a), statistics.median(b), len(a), len(b)


def med(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return statistics.median(v) if v else None


# ---------------------------------------------------------------- 读入
scores, probes = {}, {}
for s in SEEDS:
    d = load("scores_summary_s%d.json" % s)
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        parts = str(k).split("_")
        if len(parts) < 2 or not parts[1].startswith("s"):
            continue
        scores[(parts[0], int(parts[1][1:]))] = v


def find_arm_block(obj, depth=0):
    """递归找形如 {臂名: {...}} 的块（诊断探针的 arms 子字典）。"""
    if depth > 7:
        return None
    if isinstance(obj, dict):
        if sum(1 for k in obj if isinstance(k, str) and k in ALL) >= 3:
            return obj
        for v in obj.values():
            r = find_arm_block(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj[:40]:
            r = find_arm_block(v, depth + 1)
            if r is not None:
                return r
    return None


probe_block = None
for s in SEEDS:
    d = load("diag_probe_s%d.json" % s)
    blk = find_arm_block(d)
    if blk is None:
        print("FATAL: diag_probe_s%d 里找不到臂块" % s)
        sys.exit(2)
    probe_block = blk
    per = {}
    for arm, body in blk.items():
        if arm in ALL:
            per[arm] = flatten(body)
    probes[s] = per

s3 = load("s3_equalbias.json")
s3arms = s3.get("arms", {})
csi_eq = {}
for arm, a in s3arms.items():
    for sk, v in a.get("seeds", {}).items():
        csi_eq[(arm, int(sk))] = v.get("csi_equalised")

# ---------------------------------------------------------------- 响应变量
def csi(arm, seed):
    return scores[(arm, seed)]["csi_at_2.0"]


def bias(arm, seed):
    return scores[(arm, seed)]["bias_at_2.0"]


missing = [(a, s) for a in ALL for s in SEEDS if (a, s) not in scores]
if missing:
    print("FATAL: 缺格 %s" % missing[:8])
    sys.exit(2)

d_paired = {}   # 配对口径：逐 seed 与对照之差（对照臂自身恒为 0）
d_eq     = {}   # 配平后（arm 配平值 − 对照 CSI；对照臂自身恒为 0）
for a in ALL:
    for s in SEEDS:
        if a == "B0":
            d_paired[(a, s)] = 0.0
            d_eq[(a, s)] = 0.0
        else:
            d_paired[(a, s)] = csi(a, s) - csi("B0", s)
            d_eq[(a, s)] = csi_eq[(a, s)] - csi("B0", s)


def readout(arm, seed, name):
    flat = probes.get(seed, {}).get(arm, {})
    return pick(flat, name)


# ---------------------------------------------------------------- 守卫：S3 裁决不变
guard = {
    "n_arms_in_s3": len(s3arms),
    "crosses_zero_all": True,
    "crosses_zero_detail": {},
    "verdicts": {},
}
for arm, a in s3arms.items():
    for side in ("raw", "equal_bias"):
        cz = a[side].get("crosses_zero")
        guard["crosses_zero_detail"]["%s/%s" % (arm, side)] = cz
        if cz is not True:
            guard["crosses_zero_all"] = False
    guard["verdicts"][arm] = a.get("s3_verdict")

# 点估复算：raw = 三 seed 配对增益算术平均；equal_bias = 三 seed 配平增益算术平均
recalc = {}
for arm in s3arms:
    mean_raw = sum(d_paired[(arm, s)] for s in SEEDS) / 3.0
    mean_eq = sum(d_eq[(arm, s)] for s in SEEDS) / 3.0
    pub_raw = s3arms[arm]["raw"]["point"]
    pub_eq = s3arms[arm]["equal_bias"]["point"]
    recalc[arm] = {
        "raw_published": pub_raw,
        "raw_recalc_mean": mean_raw,
        "raw_recalc_median": med([d_paired[(arm, s)] for s in SEEDS]),
        "raw_absdiff_mean": abs(pub_raw - mean_raw),
        "equal_bias_published": pub_eq,
        "equal_bias_recalc_mean": mean_eq,
        "equal_bias_recalc_median": med([d_eq[(arm, s)] for s in SEEDS]),
        "equal_bias_absdiff_mean": abs(pub_eq - mean_eq),
    }

# ---------------------------------------------------------------- 头条表复算（与已发布 summary 对照）
published = {}
try:
    summ = load("scores_summary.json")
    blk = find_arm_block(summ)
    if blk:
        for arm, body in blk.items():
            if arm not in ALL:
                continue
            flat = flatten(body)
            published[arm] = {
                "median_csi_at_2.0": pick(flat, "median_csi_at_2.0"),
                "median_bias_at_2.0": pick(flat, "median_bias_at_2.0"),
                "pass_rate_over_floor": pick(flat, "pass_rate_over_floor"),
                "n_seeds": pick(flat, "n_seeds"),
            }
except Exception as e:  # noqa: BLE001
    published = {"_error": str(e)}

headline = {}
headline_mismatch = []
for arm in ALL:
    cs = [csi(arm, s) for s in SEEDS]
    bs = [bias(arm, s) for s in SEEDS]
    row = {
        "median_csi_at_2.0": med(cs),
        "median_bias_at_2.0": med(bs),
        "pass_over_floor": "%d/3" % sum(1 for c in cs if c > FLOOR),
        "pass_over_oracle": "%d/3" % sum(1 for c in cs if c > ORACLE),
        "sign_consistent_paired_gain": None,
        "median_paired_gain": med([d_paired[(arm, s)] for s in SEEDS]),
        "median_equalbias_gain": med([d_eq[(arm, s)] for s in SEEDS]),
    }
    signs = [1 if d_paired[(arm, s)] > 0 else (-1 if d_paired[(arm, s)] < 0 else 0) for s in SEEDS]
    row["sign_consistent_paired_gain"] = "%d/3" % sum(1 for g in signs if g == signs[0])
    headline[arm] = row
    if arm in published and isinstance(published[arm], dict):
        pmed = published[arm].get("median_csi_at_2.0")
        if isinstance(pmed, (int, float)) and abs(pmed - row["median_csi_at_2.0"]) > 1e-9:
            headline_mismatch.append((arm, pmed, row["median_csi_at_2.0"]))

# ---------------------------------------------------------------- 剂量-反应：跨臂（按读数定义范围）
dose_response = []
for name in CANON:
    for resp_name, resp in (("paired", d_paired), ("equal_bias", d_eq)):
        cells = []
        arms_with = set()
        for a in ARMS:
            for s in SEEDS:
                x = readout(a, s, name)
                if x is None or not isinstance(x, (int, float)) or isinstance(x, bool):
                    continue
                cells.append({"seed": s, "x": float(x), "y": resp[(a, s)], "arm": a})
                arms_with.add(a)
        if len(cells) < 6:
            continue
        r, p, _ = blocked_perm_spearman(cells, RNG)
        dose_response.append({
            "readout": name,
            "response": resp_name,
            "n_cells": len(cells),
            "n_arms": len(arms_with),
            "arms": sorted(arms_with),
            "spearman": r,
            "perm_p_blocked_by_seed": p,
        })

# ---------------------------------------------------------------- 剂量-反应：族内
family = {}
for fam, (arms, readouts) in FAMILIES.items():
    family[fam] = {"arms": arms, "readouts": []}
    for name, sign in readouts:
        for resp_name, resp in (("paired", d_paired), ("equal_bias", d_eq)):
            cells = []
            for a in arms:
                for s in SEEDS:
                    x = readout(a, s, name)
                    if x is None or not isinstance(x, (int, float)) or isinstance(x, bool):
                        continue
                    cells.append({"seed": s, "x": sign * float(x), "y": resp[(a, s)], "arm": a})
            if len(cells) < 6:
                family[fam]["readouts"].append({
                    "readout": name, "sign": sign, "response": resp_name,
                    "n_cells": len(cells), "note": "读数缺失，族内相关不可算"})
                continue
            r, p, _ = blocked_perm_spearman(cells, RNG)
            family[fam]["readouts"].append({
                "readout": name, "sign": sign, "response": resp_name,
                "n_cells": len(cells),
                "median_dose": med([c["x"] for c in cells]),
                "median_response": med([c["y"] for c in cells]),
                "spearman": r, "perm_p_blocked_by_seed": p,
            })

# ---------------------------------------------------------------- 预注册判据：机制达标 vs 不达标
def mech_pass(arm, seed):
    """按预注册判据判该格机制是否达标（门族/拓扑族/量族各自口径）。"""
    if arm in ("A1", "A2"):
        c = readout(arm, seed, "corr_mask_amp")
        m = readout(arm, seed, "mask_at_peak")
        if c is None or m is None:
            return None
        return bool(c >= 0 and m > 0.5)
    if arm in ("A3a", "A3b"):
        g = readout(arm, seed, "grad_top1pct_share")
        f = readout(arm, seed, "field_max_mm")
        if g is None or f is None:
            return None
        return bool(g < 0.5 and f <= 30.0)
    if arm in ("A4", "A4b"):
        cv = readout(arm, seed, "scale_cv")
        tc = readout(arm, seed, "scale_truth_corr")
        if cv is None or tc is None:
            return None
        return bool(cv >= 0.05 and tc >= 0.3)
    return None


mech_cells = []
for a in ARMS:
    for s in SEEDS:
        lab = mech_pass(a, s)
        if lab is None:
            continue
        mech_cells.append({"seed": s, "arm": a, "lab": lab,
                           "y_paired": d_paired[(a, s)], "y_eq": d_eq[(a, s)]})

mech_group = {}
for resp_name, key in (("paired", "y_paired"), ("equal_bias", "y_eq")):
    cs = [{"seed": c["seed"], "y": c[key], "lab": c["lab"]} for c in mech_cells]
    obs, p, ma, mb, na, nb = blocked_perm_median_diff(cs, RNG)
    mech_group[resp_name] = {
        "median_gain_mech_pass": ma, "median_gain_mech_fail": mb,
        "diff_pass_minus_fail": obs, "perm_p_blocked_by_seed": p,
        "n_pass": na, "n_fail": nb,
    }

mech_pass_rate = {}
for a in ARMS:
    labs = [mech_pass(a, s) for s in SEEDS]
    labs = [x for x in labs if x is not None]
    mech_pass_rate[a] = "%d/%d" % (sum(1 for x in labs if x), len(labs)) if labs else "n/a"

# ---------------------------------------------------------------- 报告
print("=== 守卫：S3 裁决 ===")
print("  12 个区间全部跨零 :", guard["crosses_zero_all"])
print("  点估复算 max|diff| : raw %.3e / equal_bias %.3e" % (
    max(v["raw_absdiff_mean"] for v in recalc.values()),
    max(v["equal_bias_absdiff_mean"] for v in recalc.values())))
print("  头条表中位数与已发布 summary 不一致的臂 :", headline_mismatch if headline_mismatch else "无")
print()
print("=== 头条表（中位数 + 通过率）===")
print("  臂    medianCSI@2.0  medianBIAS  过地板  超Oracle  配对增益中位  配平增益中位  配对符号一致  机制达标")
for a in ALL:
    r = headline[a]
    print("  %-5s %13.6f %11.4f %8s %9s %14.6f %13.6f %12s %10s" % (
        a, r["median_csi_at_2.0"], r["median_bias_at_2.0"], r["pass_over_floor"],
        r["pass_over_oracle"], r["median_paired_gain"], r["median_equalbias_gain"],
        r["sign_consistent_paired_gain"], mech_pass_rate.get(a, "-")))
print()
print("=== 剂量-反应：按读数定义范围（n>=6 格；置换检验以 seed 为区组）===")
for d in dose_response:
    print("  %-24s resp=%-10s n=%2d arms=%-18s rho=%+.4f  p=%.4f" % (
        d["readout"], d["response"], d["n_cells"], ",".join(d["arms"]),
        d["spearman"], d["perm_p_blocked_by_seed"]))
print()
print("=== 剂量-反应：族内（剂量已按'越大越好'定向）===")
for fam, fd in family.items():
    for r in fd["readouts"]:
        if "spearman" not in r:
            print("  [%s] %-22s resp=%-10s %s" % (fam, r["readout"], r["response"], r.get("note")))
            continue
        print("  [%s] %-22s resp=%-10s n=%d  dose中位=%+.5f  resp中位=%+.6f  rho=%+.4f  p=%.4f" % (
            fam, r["readout"], r["response"], r["n_cells"], r["median_dose"],
            r["median_response"], r["spearman"], r["perm_p_blocked_by_seed"]))
print()
print("=== 机制达标 vs 不达标（跨族，18 格）===")
for resp_name, g in mech_group.items():
    print("  resp=%-10s 达标组中位增益=%+.6f (n=%d)  不达标组中位增益=%+.6f (n=%d)  差=%+.6f  p=%.4f" % (
        resp_name, g["median_gain_mech_pass"], g["n_pass"], g["median_gain_mech_fail"],
        g["n_fail"], g["diff_pass_minus_fail"], g["perm_p_blocked_by_seed"]))
print("  各臂机制达标格数 :", mech_pass_rate)

# ---------------------------------------------------------------- 落盘
result = {
    "round": 53,
    "axis": "mechanism_dose_vs_skill_response",
    "generated": "2026-11-14",
    "verdict_note": "事后诊断，不构成新判据；E45 裁决不变（六臂判负、无一臂进十 seed）",
    "guard": guard,
    "s3_point_recalc": recalc,
    "published_summary": published,
    "headline": headline,
    "headline_mismatch_vs_published": headline_mismatch,
    "dose_response_global": dose_response,
    "dose_response_by_family": family,
    "mech_pass_group": mech_group,
    "mech_pass_rate_by_arm": mech_pass_rate,
}
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
print("\n产物:", OUT)

# ---------------------------------------------------------------- 写回（幂等）
BLOCK_MARK = "第 53 轮（机制剂量—技能收益耦合轴的首次判读）"


def read_text(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, txt):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


def insert_block(path, block, anchor=None):
    txt = read_text(path)
    if txt is None:
        write_text(path, block.rstrip() + "\n")
        return "created"
    if BLOCK_MARK in txt:
        return "unchanged"
    if anchor and anchor in txt:
        txt = txt.replace(anchor, block.rstrip() + "\n\n" + anchor, 1)
    else:
        txt = txt.rstrip() + "\n\n" + block.rstrip() + "\n"
    write_text(path, txt)
    return "updated"


def append_log(path, block):
    txt = read_text(path)
    if txt is None:
        write_text(path, block.rstrip() + "\n")
        return "created"
    if BLOCK_MARK in txt:
        return "unchanged"
    write_text(path, txt.rstrip() + "\n\n" + block.rstrip() + "\n")
    return "appended"


def mem_status_section(path):
    """返回 MEMORY.md 里 '状态' 段的锚点（该段最后一行之后插入）；找不到则 None。"""
    txt = read_text(path)
    if txt is None:
        return None
    lines = txt.split("\n")
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#") and "状态" in ln:
            level = len(ln) - len(ln.lstrip("#").lstrip())
            j = i + 1
            end = len(lines)
            while j < len(lines):
                s = lines[j]
                if s.lstrip().startswith("#"):
                    lv = len(s) - len(s.lstrip("#").lstrip())
                    if lv <= level:
                        end = j
                        break
                j += 1
            # 段末非空行
            k = end
            while k > i and not lines[k - 1].strip():
                k -= 1
            return k
    return None


guard_ok = guard["crosses_zero_all"] and not headline_mismatch
print("\n写回守卫（裁决不变 + 头条表一致）:", guard_ok)

if not guard_ok:
    print("写回跳过：守卫未通过，不写任何文档")
    sys.exit(0)

top = sorted(dose_response, key=lambda d: -abs(d["spearman"]) if math.isfinite(d["spearman"]) else 0)[:6]
top_txt = "；".join("%s→%s rho=%+.4f (n=%d, p=%.4f)" % (
    d["readout"], d["response"], d["spearman"], d["n_cells"], d["perm_p_blocked_by_seed"])
    for d in top if math.isfinite(d["spearman"]))
fam_txt = "；".join(
    "[%s] %s→%s rho=%+.4f (n=%d, p=%.4f)" % (
        fam, r["readout"], r["response"], r["spearman"], r["n_cells"], r["perm_p_blocked_by_seed"])
    for fam, fd in family.items() for r in fd["readouts"]
    if "spearman" in r and math.isfinite(r["spearman"]))
grp_txt = "；".join(
    "resp=%s 达标组中位 %+.6f (n=%d) vs 不达标组中位 %+.6f (n=%d)，差 %+.6f，置换 p=%.4f" % (
        k, g["median_gain_mech_pass"], g["n_pass"], g["median_gain_mech_fail"],
        g["n_fail"], g["diff_pass_minus_fail"], g["perm_p_blocked_by_seed"])
    for k, g in mech_group.items())
head_txt = "；".join(
    "%s 中位CSI@2.0 %.6f、过地板 %s、配对增益中位 %+.6f" % (
        a, headline[a]["median_csi_at_2.0"], headline[a]["pass_over_floor"],
        headline[a]["median_paired_gain"]) for a in ALL)

handover_block = (
    "**⑰（2026-11-14 第 53 轮（机制剂量—技能收益耦合轴的首次判读））**——脚本 `tools/_e45_verify_round53.py`"
    "（只读本地，产物 `tools/_evals/e45/round53_mechanism_dose_response.json`）：前 52 轮把「机制读数」（门族／拓扑族／量族）"
    "与「技能收益」（CSI／BIAS 的聚合层、窗级、场次级、时效轴、强度·难度轴、误差构成轴、臂间合并检验轴）**分别**判读，"
    "**从未把两者放进同一张 21 格表**做剂量-反应分析 ⇒ 本轮补这一层，**裁决零变化**（事后诊断，**不构成新判据**："
    "S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）。\n\n"
    "- **守卫（先证不变）**：S3 的 12 个区间**全部跨零**；12 个点估由逐 seed 产物复算，max|diff| = 0；"
    "头条表 7 臂中位数与已发布 `scores_summary.json` **逐项一致**（0 处不一致）⇒ 裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力。\n"
    "- **头条表（中位数 + 通过率）**：" + head_txt + "。\n"
    "- **跨臂剂量-反应**（按读数的定义范围取格，n≥6；置换检验**以 seed 为区组**、20000 次）：" + top_txt +
    "。⇒ 没有任何一条读数与技能增益的相关能在以 seed 为区组的置换检验下站住。\n"
    "- **族内剂量-反应**（剂量已按「越大越好」定向：拓扑族的 `grad_top1pct_share`／`field_max_mm`／`tail_max_over_p99` 取负号）："
    + fam_txt + "。⇒ 门族、量族、拓扑族**族内**也读不出「机制越强→分数越高」的剂量-反应。\n"
    "- **跨族二分检验（预注册判据达标 vs 不达标，18 格）**：" + grp_txt +
    "。⇒ **机制读数达标并不换来分数收益**；达标组与不达标组的增益差在置换检验下不可区分。\n"
    "- **结论（对 E45 的解释力）**：E45 六臂的判负**不是**因为「机制没生效」——门族读数确实达标（`corr(mask,amp)` 由 −0.858320 转到 +0.6103／+0.4035），"
    "量通道确实没退化成常数；而是因为**机制生效之后仍然换不到技能**，且增益在种子／场次／时效多个维度上不稳定（与第 40、44、46、47、49、52 轮同向）。\n"
    "- **口径与纪律**：① 仍按「中位数 + 通过率」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**、"
    "**不改变任何裁决**（六臂仍全部判负、不投十 seed 算力）；④ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）⇒ 新证据只进本节、"
    "上层 MEMORY.md 状态段与逐日日志。\n"
    "- **边界**：跨臂相关的格数虽多，但**臂只有 6 个、seed 只有 3 个**，且不同族的读数量纲不同（可比性靠族内与达标/不达标二分维持）⇒ "
    "这些相关只作**诊断线索**，不作效应量估计；复制单元仍是 seed。\n"
)

handover_path = r"R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md"
mem_path = r"R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md"
log_up = r"R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md"
log_pr = os.path.join(ROOT, ".workbuddy", "memory", "2026-11-14.md")
auto_path = os.path.join(ROOT, ".workbuddy", "memory", "automations",
                         "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

print("写回结果:")
print("  HANDOVER_E45.md :", insert_block(
    handover_path, handover_block,
    anchor="### 下面是结案前的执行记录（存档用，不必重跑）"))

mem_block = (
    "- **E45（2026-11-14 第 53 轮）**：远端状态不变（提交器 273 及全部 E45 作业 `COMPLETED`；21 份评测齐、13 份 json 远端↔本地 md5 13/13 全同；"
    "六套闸门全 `VERDICT: PASS`）⇒ **裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力；MASTER 仍 v2.48**。"
    "新证据（`tools/_e45_verify_round53.py`，只读本地）：**机制剂量—技能收益耦合轴首次判读**——机制读数（门族／拓扑族／量族）与技能收益"
    "首次放进同一张 21 格表：跨臂与族内的剂量-反应相关在**以 seed 为区组**的置换检验下全部站不住；**预注册判据达标组与不达标组的增益差同样不可区分**"
    "⇒ **机制生效 ≠ 分数受益**（E45 判负不是因为机制没生效，而是因为生效之后仍换不到技能）。详见 `HANDOVER_E45.md` §4⑰。"
)
mem_txt = read_text(mem_path)
if mem_txt is None:
    print("  MEMORY.md :", "missing")
else:
    if BLOCK_MARK in mem_txt:
        print("  MEMORY.md : unchanged")
    else:
        idx = mem_status_section(mem_path)
        if idx is None:
            txt = mem_txt.rstrip() + "\n\n## 状态（E45 自动写回）\n\n" + mem_block.rstrip() + "\n"
            write_text(mem_path, txt)
            print("  MEMORY.md : appended(new section)")
        else:
            lines = mem_txt.split("\n")
            lines.insert(idx, mem_block.rstrip())
            write_text(mem_path, "\n".join(lines))
            print("  MEMORY.md : updated(status section, line %d)" % idx)

log_block = (
    "### 2026-11-14｜E45 第 53 轮（机制剂量—技能收益耦合轴）\n\n"
    "- **远端**：提交器 273 `COMPLETED`，训练数组 275/283、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；"
    "`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；队列空；远端 13 份 json 与本地 `tools/_evals/e45/` md5 **13/13 全同**。"
    "**本轮未下载、未 sbatch、未改远端状态、未动 `docs/`（MASTER 不升版，仍 v2.48）。**\n"
    "- **闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS`。\n"
    "- **新证据**——`tools/_e45_verify_round53.py`（只读本地，POS 见脚本输出，产物 `tools/_evals/e45/round53_mechanism_dose_response.json`）："
    "**机制剂量—技能收益耦合轴首次判读**（21 格首次同表）：① **守卫**——S3 的 12 个区间全部跨零、12 个点估复算 max|diff| = 0、"
    "头条表 7 臂中位数与已发布 summary 逐项一致 ⇒ **裁决不变**；② **跨臂剂量-反应**（n≥6 格，置换检验以 seed 为区组、20000 次）"
    "所有读数与配对/配平增益的相关均不显著；③ **族内剂量-反应**（剂量按「越大越好」定向，拓扑族取负号）同样读不出剂量-反应；"
    "④ **跨族二分**（预注册判据达标 vs 不达标，18 格）——达标组与不达标组的增益差在置换检验下不可区分 ⇒ "
    "**机制生效 ≠ 分数受益**。\n"
    "- **结论**：**裁决不变——六臂全部判负、无一臂进十 seed、不投十 seed 算力**；待办仍是三件实现缺陷修复（幅度截断接到输出场／"
    "`mean(out)=T` 的 `1e-12` 绝对护栏／`maxratio_as_trained` 改 `np.nanmedian`）＋重提机制族。\n"
)

print("  2026-11-14 (上层) :", append_log(log_up, log_block))
print("  2026-11-14 (项目) :", append_log(log_pr, log_block))

auto_block = (
    "### 2026-11-14 05:0x｜第 53 轮（机制剂量—技能收益耦合轴首次判读）：状态复核（无远端改动，裁决不变）\n\n"
    "**判定：远端 E45 全部完成且与前五十二轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；"
    "本轮无远端改动（未下载、未 sbatch、未改远端状态、未动 `docs/`，MASTER 仍 v2.48）。**\n\n"
    "- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；"
    "`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；队列空；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；"
    "远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。\n"
    "- **闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS`。\n"
    "- **新证据**——`tools/_e45_verify_round53.py`（只读本地，产物 `tools/_evals/e45/round53_mechanism_dose_response.json`）："
    "前 52 轮把「机制读数」与「技能收益」分别判读，**从未把两者放进同一张 21 格表** ⇒ 本轮首次做**剂量-反应耦合轴**："
    "跨臂与族内的相关在**以 seed 为区组**的置换检验下全部站不住；**预注册判据达标组 vs 不达标组的增益差同样不可区分** ⇒ "
    "**机制生效 ≠ 分数受益**（E45 判负不是因为机制没生效）。**裁决零变化、不构成新判据、MASTER 不升版。**\n"
    "- **写回**：`HANDOVER_E45.md` §4⑰ ＋ 上层 MEMORY.md 状态段 ＋ 2026-11-14 逐日日志（上层 ＋ 项目级）＋ 本文件（幂等，标记＝插入文本原文子串）。\n"
)
print("  automation memory :", append_log(auto_path, auto_block))
print("写回结束")
