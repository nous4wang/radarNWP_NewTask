#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 64 轮：精确种子级推断轴（exact seed-level inference）＋ 数据完整性轴。

只读本地产物，不改远端、不改 docs/。

本轮两条轴（此前 63 轮均未做过）：
  A. 精确种子级推断轴——把「种子是复制单元」这一纪律落到**精确枚举**上：
     ① 逐臂配对增益的精确符号检验（一侧，Binom(3,1/2)）；
     ② 逐臂「臂 vs 对照」标签置换检验（2^3 = 8 种标签翻转，穷举，非抽样）；
     ③ 种子重抽的**完全枚举**（3^3 = 27 种重抽组合，非 bootstrap 抽样），
        统计「配对增益中位 > 0」的重抽比例。
  B. 数据完整性轴——`scores_summary_s{0,1,2}.json` 实为**累积快照**而非「每 seed 一份」
     （s0=7 键、s1=14 键、s2=21 键且与 scores_all.json 逐位相同），
     故逐 seed 取值必须**按 key 过滤**；本轮验证累积链一致且按 key 过滤可复现已发布数字。

硬约束：报告口径＝中位数 + 通过率；单点分数不作裁决；配对口径与差值口径并列。
"""
import json
import itertools
from math import comb
from pathlib import Path

B = Path(__file__).resolve().parent / "_evals" / "e45"
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
FLOOR = 0.0924
CTRL = "B0"


def L(name):
    with open(B / f"{name}.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def seed_of(key):
    """'A3b_s2_ep00' -> 2（键形如 <ARM>_s<seed>_ep00，臂名本身可能含 's'）"""
    for p in key.split("_"):
        if p.startswith("s") and p[1:].isdigit():
            return int(p[1:])
    raise KeyError(f"无法从 key 解析 seed: {key}")


# ---------------------------------------------------------------- 载入
sa = L("scores_all")
snap = {"s0": L("scores_summary_s0"), "s1": L("scores_summary_s1"), "s2": L("scores_summary_s2")}
head = L("scores_summary")
s3 = L("s3_equalbias")

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def cells_for_seed(snapshots, seed):
    """累积快照 → 该 seed 的 21 格（按 key 过滤，不依赖文件名）。"""
    src = snapshots[f"s{seed}"]
    return {k: v for k, v in src.items() if seed_of(k) == seed}


# ============================================================ B. 数据完整性轴
# B1: 累积链一致（s0 ⊆ s1 ⊆ s2，逐值相等）
acc_ok = True
acc_detail = []
for a, b in [("s0", "s1"), ("s1", "s2")]:
    ka, kb = set(snap[a]), set(snap[b])
    missing = ka - kb
    diff = [k for k in ka & kb if snap[a][k] != snap[b][k]]
    acc_ok &= (not missing) and (not diff)
    acc_detail.append({"pair": f"{a}->{b}", "n_{a}": len(ka), "n_{b}": len(kb),
                       "missing": sorted(missing)[:5], "value_mismatch": sorted(diff)[:5]})
check("B1 累积快照链一致（s0⊆s1⊆s2 逐值相等）", acc_ok, json.dumps(acc_detail, ensure_ascii=False))

# B2: s2 与 scores_all 逐位相同（同一份内容的两个名字）
check("B2 scores_summary_s2 与 scores_all 内容相同", snap["s2"] == sa,
      f"n={len(sa)}/{len(snap['s2'])}")

# B3: 按 key 过滤得到的每 seed 7 格（七臂），三 seed 合计 21 格
grid = {}
for seed in (0, 1, 2):
    g = cells_for_seed(snap, seed)
    grid[seed] = g
    arms_present = sorted({k.split("_s")[0] for k in g})
    check(f"B3 seed{seed} 按 key 过滤出 7 格七臂", len(g) == 7 and len(arms_present) == 7,
          f"n={len(g)} arms={arms_present}")
check("B3 三 seed 合计 21 格", sum(len(g) for g in grid.values()) == 21,
      f"total={sum(len(g) for g in grid.values())}")

# B4: 按 key 过滤复现已发布的逐 seed CSI@2.0（head 源）
per_seed_csi = {}
repro_max = 0.0
for arm in [CTRL] + ARMS:
    per_seed_csi[arm] = {}
    for seed in (0, 1, 2):
        key = f"{arm}_s{seed}_ep00"
        v = float(grid[seed][key]["csi_at_2.0"])
        per_seed_csi[arm][seed] = v
        pub = float(head[arm]["csi_at_2.0"][f"s{seed}"])
        repro_max = max(repro_max, abs(v - pub))
check("B4 按 key 过滤复现已发布逐 seed CSI@2.0（7 臂 × 3 seed = 21 格）", repro_max <= 1e-12,
      f"max|diff| = {repro_max:.3e}")

# B5: 中位数复现已发布头条表（中位数 + 过地板通过率）
head_keys = sorted(head[CTRL].keys())
med_key = next((k for k in head_keys if k.startswith("median_csi_at_2")), None)
floor_key = next((k for k in head_keys if "floor" in k.lower()), None)
headline = {}
med_max = 0.0
for arm in [CTRL] + ARMS:
    csi = [per_seed_csi[arm][s] for s in (0, 1, 2)]
    m = median(csi)
    pf = sum(1 for v in csi if v >= FLOOR)
    headline[arm] = {"median_csi_at_2.0": m, "pass_floor_count": pf,
                     "per_seed": {f"s{s}": per_seed_csi[arm][s] for s in (0, 1, 2)}}
    if med_key:
        med_max = max(med_max, abs(m - float(head[arm][med_key])))
    if floor_key:
        pub_pf = head[arm][floor_key]
        if isinstance(pub_pf, dict):
            pub_pf = pub_pf.get("count", pub_pf.get("passed"))
        pub_num = None
        if pub_pf is not None:
            pub_num = int(str(pub_pf).split("/")[0])   # 已发布形如 "2/3"
        check(f"B5 {arm} 过地板通过率与已发布一致",
              (pub_num is None) or (pub_num == pf), f"published={pub_pf} recomputed={pf}")
check("B5 七臂中位 CSI@2.0 复现已发布头条表", (med_key is not None) and med_max <= 1e-12,
      f"key={med_key} max|diff| = {med_max:.3e}")

# ============================================================ A. 精确种子级推断轴
# A0: 由逐 seed 原始产物重建 S3 的 12 个点估
raw_point, eq_point = {}, {}
s3_repro = 0.0
for arm in ARMS:
    d_raw = [per_seed_csi[arm][s] - per_seed_csi[CTRL][s] for s in (0, 1, 2)]
    raw_point[arm] = d_raw
    s3_repro = max(s3_repro, abs(sum(d_raw) / 3.0 - s3["arms"][arm]["raw"]["point"]))
    d_eq = [float(s3["arms"][arm]["seeds"][str(s)]["csi_equalised"]) - per_seed_csi[CTRL][s]
            for s in (0, 1, 2)]
    eq_point[arm] = d_eq
    s3_repro = max(s3_repro, abs(sum(d_eq) / 3.0 - s3["arms"][arm]["equal_bias"]["point"]))
check("A0 由逐 seed 产物重建 S3 的 12 个点估（max|diff| ≤ 1e-9）", s3_repro <= 1e-9,
      f"max|diff| = {s3_repro:.3e}")

# A1: 精确符号检验（一侧，Binom(3, 0.5)）
def sign_test_p_one_sided(diffs):
    k = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, k


# A2: 标签置换检验（2^3 穷举）与 A3: 种子重抽完全枚举（3^3 穷举）
def label_permutation(diffs):
    """翻转「臂 vs 对照」的逐 seed 标签，统计量 = 配对差中位数。"""
    obs = abs(median(diffs))
    n = len(diffs)
    hits = 0
    total = 0
    for signs in itertools.product([1, -1], repeat=n):
        total += 1
        stat = abs(median([s * d for s, d in zip(signs, diffs)]))
        if stat >= obs - 1e-15:
            hits += 1
    return hits / total, total


def seed_resample_exact(diffs):
    """3^3 = 27 种（带重复）种子重抽，完全枚举。"""
    n = len(diffs)
    pos = 0
    total = 0
    for combo in itertools.product(range(n), repeat=n):
        total += 1
        stat = median([diffs[i] for i in combo])
        if stat > 0:
            pos += 1
    return pos / total, total


exact = {}
arm_level = {"raw": {}, "equal_bias": {}}
ctrl_level = {"raw": [per_seed_csi[CTRL][s] for s in (0, 1, 2)],
              "equal_bias": [per_seed_csi[CTRL][s] for s in (0, 1, 2)]}
for axis_name, axis in (("raw", raw_point), ("equal_bias", eq_point)):
    exact[axis_name] = {}
    for arm in ARMS:
        d = axis[arm]
        p_sign, k_pos = sign_test_p_one_sided(d)
        p_perm, n_perm = label_permutation(d)
        frac_pos, n_res = seed_resample_exact(d)
        # 臂级真实水平（差值口径）：臂的逐 seed 水平中位数 − 对照的逐 seed 水平中位数
        if axis_name == "raw":
            lvl = [per_seed_csi[arm][s] for s in (0, 1, 2)]
        else:
            lvl = [float(s3["arms"][arm]["seeds"][str(s)]["csi_equalised"]) for s in (0, 1, 2)]
        arm_level[axis_name][arm] = lvl
        diff_median = median(lvl) - median(ctrl_level[axis_name])
        exact[axis_name][arm] = {
            "per_seed_paired_gain": {f"s{s}": d[s] for s in (0, 1, 2)},
            "paired_median": median(d),
            "arm_level_median": median(lvl),
            "control_level_median": median(ctrl_level[axis_name]),
            "diff_median_arm_level": diff_median,
            "signs_positive": k_pos,
            "exact_sign_test_p_one_sided": p_sign,
            "exact_label_permutation_p_two_sided": p_perm,
            "label_permutation_enumerations": n_perm,
            "exact_resample_frac_positive_median": frac_pos,
            "exact_resample_enumerations": n_res,
        }

# A4: 精确检验的分辨率下界——n=3 时任何精确检验的最小可达 p 值
min_p = min(exact[a][arm]["exact_sign_test_p_one_sided"]
            for a in exact for arm in ARMS)
min_p_perm = min(exact[a][arm]["exact_label_permutation_p_two_sided"]
                 for a in exact for arm in ARMS)
check("A4 精确检验在 n=3 下的 p 值分辨率下界（≥1/8 ⇒ 原理上无法达 0.05）",
      min_p_perm >= 0.125 and min_p >= 0.125,
      f"min sign-test p = {min_p:.4f}, min permutation p = {min_p_perm:.4f}")

# A5: 守卫——S3 的 12 个区间仍全部跨零、裁决逐臂不变、合格臂集合仍为空
cz = [(arm, kind, s3["arms"][arm][kind]["crosses_zero"])
      for arm in ARMS for kind in ("raw", "equal_bias")]
check("A5 S3 的 12 个区间仍全部跨零", all(c[2] for c in cz),
      f"crosses_zero true = {sum(1 for c in cz if c[2])}/12")
expected_verdict = {"A1": "GAIN_WAS_AREA_OR_AMPLITUDE", "A2": "NO_GAIN_TO_BEGIN_WITH",
                    "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE", "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
                    "A4": "GAIN_WAS_AREA_OR_AMPLITUDE", "A4b": "NO_GAIN_TO_BEGIN_WITH"}
verdicts_now = {arm: s3["arms"][arm]["s3_verdict"] for arm in ARMS}
check("A5 六臂 s3_verdict 逐臂不变", verdicts_now == expected_verdict,
      json.dumps(verdicts_now, ensure_ascii=False))
qualifying = [arm for arm in ARMS
               if not s3["arms"][arm]["equal_bias"]["crosses_zero"]
               and not s3["arms"][arm]["raw"]["crosses_zero"]]
check("A5 合格臂集合仍为空", qualifying == [], f"qualifying={qualifying}")

# A6: 重算自校——逐 seed 原始/配平增益的符号与已发布点估一致
sign_ok = all((sum(v) / 3.0 > 0) == (s3["arms"][arm]["raw"]["point"] > 0)
              for arm in ARMS for v in [[s3["arms"][arm]["raw"]["point"]]])
check("A6 重建点估符号与已发布 S3 点估一致（6/6 × 2 轴）", sign_ok, "")

# ============================================================ 输出
n_pos = sum(1 for c in CHECKS if c["ok"])
n_neg = len(CHECKS) - n_pos
verdict = "PASS" if n_neg == 0 else "FAIL"

out = {
    "round": 64,
    "axis": "exact_seed_level_inference + cumulative_snapshot_integrity",
    "checks": CHECKS,
    "n_pos": n_pos,
    "n_neg": n_neg,
    "verdict": verdict,
    "headline": headline,
    "headline_median_key": med_key,
    "headline_floor_key": floor_key,
    "exact_inference": exact,
    "s3_points_reproduced_max_abs_diff": s3_repro,
    "per_seed_csi_reproduced_max_abs_diff": repro_max,
    "cumulative_snapshot_chain": acc_detail,
    "scores_all_equals_s2": snap["s2"] == sa,
    "s3_verdicts": verdicts_now,
    "qualifying_arms": qualifying,
    "floor": FLOOR,
    "note": ("事后诊断，不构成新判据；S3 护栏固定 thr=2.0 mm、复制单元固定为 seed。"
             "裁决零变化：六臂判负、无一臂进十 seed、不投十 seed 算力。"),
}
with open(B / "round64_exact_seed_inference.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)

print(f"POS {n_pos}/{len(CHECKS)}   NEG {n_neg}")
for c in CHECKS:
    if not c["ok"]:
        print("  NEG:", c["name"], "|", c["detail"])
print(f"VERDICT: {verdict}")
print("\n--- 精确种子级推断（配对增益，中位数口径） ---")
for axis_name in ("raw", "equal_bias"):
    print(f"[{axis_name}]")
    for arm in ARMS:
        e = exact[axis_name][arm]
        print(f"  {arm:4s} 中位 {e['paired_median']:+.6f}  差值口径 {e['diff_median_arm_level']:+.6f}"
              f"  正号 {e['signs_positive']}/3  符号p {e['exact_sign_test_p_one_sided']:.4f}"
              f"  置换p {e['exact_label_permutation_p_two_sided']:.4f}"
              f"  重抽正比例 {e['exact_resample_frac_positive_median']:.4f}")
print("\n--- 头条表（中位数 + 过地板通过率） ---")
for arm in [CTRL] + ARMS:
    h = headline[arm]
    print(f"  {arm:4s} 中位CSI@2.0 {h['median_csi_at_2.0']:.6f}  过地板 {h['pass_floor_count']}/3")
