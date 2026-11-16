#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 44 轮只读复核：多阈值技能剖面 + POD/FAR 分解 + 增益-偏置共变。

背景（前 43 轮都只做过的事）：
  - 第 40 轮：S3 裁决拆到逐 seed（只在 thr=2.0 上）
  - 第 41 轮：maxratio_as_trained > 0.9 逐 seed 判读
  - 第 42 轮：B0_s0 跨实验逐位可复现性
  - 第 43 轮：派生布尔 scale_is_constant 的分支分解

本轮新做的（前三者都没碰过）：
  已取回的产物里每格带 csi/bias/pod/far @ 0.5 / 2.0 / 4.0 三个阈值，
  但前 43 轮的判读全部只在 thr=2.0 上做。本脚本把六个臂的技能剖面
  摊到三个阈值上，看「增益是否随阈值单调放大」，以及增益与正偏置
  （过预报）是否共变 —— 这是「面积/幅度效应」这一裁决的独立佐证口径。

只读：只读本地 tools/_evals/e45/*.json，不触碰远端、不写任何文档。
口径：中位数 + 通过率；单点分数不得单独出现（均值只用于共变性相关系数）。
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(BASE, "_evals", "e45")

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
TREAT = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]          # 六个被裁决臂
# (阈值数值, 产物里的键后缀) —— 注意不能用 "%g" 拼键：2.0 会变成 "2" 而产物是 "2.0"
THRS = [(0.5, "0.5"), (2.0, "2.0"), (4.0, "4.0")]

# 记录值回归护栏（来自已冻结的交接文档/MASTER，防产物漂移）
GUARD = {
    ("B0", 0, "csi_at_2.0"): 0.12181703042952949,
    ("A4", 0, "csi_at_2.0"): 0.1755001059215604,
    ("A1", 0, "csi_at_2.0"): 0.1183974362784095,
    ("A3a", 0, "csi_at_2.0"): 0.14947025751330345,
    ("A3b", 0, "csi_at_2.0"): 0.16966079982919896,
    ("A4b", 0, "csi_at_2.0"): 0.13756654739888727,
    ("A2", 0, "csi_at_2.0"): 0.11200272630175023,
}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def pearson(pairs):
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (sx * sy)


def load(name):
    with open(os.path.join(EV, name), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    fails = []
    scores = load("scores_all.json")
    s3 = load("s3_equalbias.json")

    # ---- 解析 21 格 ----
    cells = {}
    for key, v in scores.items():
        parts = key.split("_")
        arm = parts[0]
        seed = int(parts[1].lstrip("s"))
        cells[(arm, seed)] = v
    if len(cells) != 21:
        fails.append("cell count %d != 21" % len(cells))
    for arm in ARMS:
        for s in (0, 1, 2):
            if (arm, s) not in cells:
                fails.append("missing cell %s_s%d" % (arm, s))

    # ---- 回归护栏 ----
    for (arm, seed, key), expect in GUARD.items():
        got = cells.get((arm, seed), {}).get(key)
        if got is None or abs(got - expect) > 1e-15:
            fails.append("GUARD %s_s%d.%s = %r != %r" % (arm, seed, key, got, expect))

    print("=" * 78)
    print("E45 第 44 轮 · 多阈值技能剖面（只读本地产物）")
    print("=" * 78)

    # ---- 1. 每阈值：各臂中位数 vs 同 seed 对照 B0 的中位数增益 + 正增益 seed 通过率 ----
    print("\n[1] 各阈值 CSI 中位数与相对同批对照 B0 的中位数增益（正增益 seed 通过率）")
    hdr = "arm   | " + " | ".join(
        "thr%.1f med / dmed (n+/3)" % t for t, ts in THRS)
    print(hdr)
    print("-" * len(hdr))
    gain_by_thr = {}
    for arm in ARMS:
        row = []
        for t, ts in THRS:
            k = "csi_at_%s" % ts
            med = median([cells[(arm, s)][k] for s in (0, 1, 2)])
            deltas = [cells[(arm, s)][k] - cells[("B0", s)][k] for s in (0, 1, 2)]
            npos = sum(1 for d in deltas if d > 0)
            gain_by_thr[(arm, t)] = (med, median(deltas), npos, deltas)
            row.append("%.4f / %+.4f (%d/3)" % (med, median(deltas), npos))
        print("%-5s | %s" % (arm, " | ".join(row)))

    print("\n  对照 B0 的三 seed CSI 中位数：" +
          " | ".join("thr%.1f=%.4f" % (t, gain_by_thr[("B0", t)][0]) for t, ts in THRS))

    print("\n[1b] 两种「中位数增益」口径并列（避免口径误读）")
    print("  配对口径 = 三 seed 逐 seed 差值的中位数；差值口径 = 两条中位数相减")
    print("  %-5s %-28s %-28s %-28s" % ("arm", "thr0.5 配对/差值", "thr2.0 配对/差值", "thr4.0 配对/差值"))
    for arm in ARMS:
        row = []
        for t, ts in THRS:
            paired = gain_by_thr[(arm, t)][1]
            diff = gain_by_thr[(arm, t)][0] - gain_by_thr[("B0", t)][0]
            row.append("%+.4f / %+.4f" % (paired, diff))
        print("  %-5s %-28s %-28s %-28s" % (arm, row[0], row[1], row[2]))

    # ---- 2. 阈值单调性：增益是否随阈值放大 ----
    print("\n[2] 增益随阈值的走向（增益是否只在高阈值出现 ⇒ 幅度/面积驱动而非技能驱动）")
    for arm in TREAT:
        ds = [gain_by_thr[(arm, t)][1] for t, ts in THRS]
        mono_up = ds[0] <= ds[1] <= ds[2]
        if all(d <= 0 for d in ds):
            tag = "三阈值全负 ⇒ 无增益"
        elif ds[0] <= 0 < ds[2]:
            tag = "仅高阈值正增益（低阈值为零/负） ⇒ 幅度/面积驱动"
        elif all(d > 0 for d in ds):
            tag = "三阈值皆正 ⇒ 唯一像真技能的形态"
        elif mono_up:
            tag = "随阈值单调放大 ⇒ 幅度/面积驱动"
        else:
            tag = "非单调"
        print("  %-4s dmed @0.5/2.0/4.0 = %+.4f / %+.4f / %+.4f  → %s"
              % (arm, ds[0], ds[1], ds[2], tag))

    # ---- 3. POD / FAR / BIAS 分解（产物里 POD/FAR 只落在 thr=2.0） ----
    print("\n[3] POD / FAR / BIAS 中位数分解（产物仅在 thr=2.0 记录 POD/FAR）")
    ts = "2.0"
    print("  --- thr=2.0 ---")
    print("  %-5s %8s %8s %8s %8s %8s" % ("arm", "POD", "FAR", "CSI", "BIAS", "dPOD"))
    for arm in ARMS:
        pod = median([cells[(arm, s)]["pod_at_%s" % ts] for s in (0, 1, 2)])
        far = median([cells[(arm, s)]["far_at_%s" % ts] for s in (0, 1, 2)])
        csi = median([cells[(arm, s)]["csi_at_%s" % ts] for s in (0, 1, 2)])
        bias = median([cells[(arm, s)]["bias_at_%s" % ts] for s in (0, 1, 2)])
        dpod = pod - median([cells[("B0", s)]["pod_at_%s" % ts] for s in (0, 1, 2)])
        print("  %-5s %8.4f %8.4f %8.4f %8.4f %+8.4f" % (arm, pod, far, csi, bias, dpod))

    print("\n  --- CSI / BIAS 中位数随阈值（对照 B0 同列）---")
    print("  %-5s %-22s %-22s %-22s" % ("arm", "CSI@0.5 / dCSI / BIAS",
                                        "CSI@2.0 / dCSI / BIAS", "CSI@4.0 / dCSI / BIAS"))
    for arm in ARMS:
        row = []
        for t, tsx in THRS:
            csi = gain_by_thr[(arm, t)][0]
            d = gain_by_thr[(arm, t)][1]
            bias = median([cells[(arm, s)]["bias_at_%s" % tsx] for s in (0, 1, 2)])
            row.append("%.4f / %+.4f / %+.4f" % (csi, d, bias))
        print("  %-5s %-22s %-22s %-22s" % (arm, row[0], row[1], row[2]))

    # ---- 4. 增益与正偏置共变（跨 21 格的 Pearson） ----
    print("\n[4] 跨 21 格的共变性（Pearson r，均值口径仅用于相关系数）")
    for t, ts in THRS:
        k_c = "csi_at_%s" % ts
        k_b = "bias_at_%s" % ts
        pairs_cb, pairs_cc = [], []
        for arm in TREAT:
            for s in (0, 1, 2):
                d_c = cells[(arm, s)][k_c] - cells[("B0", s)][k_c]
                d_b = cells[(arm, s)][k_b] - cells[("B0", s)][k_b]
                pairs_cb.append((d_c, d_b))
        for arm in TREAT:
            for s in (0, 1, 2):
                d2 = cells[(arm, s)]["csi_at_2.0"] - cells[("B0", s)]["csi_at_2.0"]
                d05 = cells[(arm, s)]["csi_at_0.5"] - cells[("B0", s)]["csi_at_0.5"]
                pairs_cc.append((d2, d05))
        print("  thr=%.1f : r(dCSI, dBIAS) = %+.4f   (n=18)" % (t, pearson(pairs_cb)))
    print("  r(dCSI@2.0, dCSI@0.5) = %+.4f   (n=18)" % pearson(pairs_cc))

    # ---- 5. S3 裁决复核：两侧 CI 是否仍全部跨零 ----
    print("\n[5] S3 等偏置对照复核（原始产物 s3_equalbias.json）")
    print("  thr=%s epoch=%s control=%s n_boot=%s"
          % (s3.get("threshold_mm"), s3.get("epoch"), s3.get("control"), s3.get("n_boot")))
    for arm in TREAT:
        a = s3["arms"][arm]
        out = []
        for side in ("raw", "equal_bias"):
            d = a[side]
            lo, hi = d["ci_lo"], d["ci_hi"]
            cross = bool(d.get("crosses_zero")) and lo <= 0.0 <= hi
            out.append("%s[%+.4f,%+.4f]%s" % (side, lo, hi, "跨零" if cross else "不跨零"))
            if not cross:
                fails.append("S3 %s %s does not cross zero (ci_lo=%.6f ci_hi=%.6f flag=%s)"
                             % (arm, side, lo, hi, d.get("crosses_zero")))
        print("  %-4s %s | %s | %s" % (arm, out[0], out[1], a.get("s3_verdict")))

    # ---- 6. 硬约束复核：无臂 3/3 seed 正增益 ----
    for arm in TREAT:
        if gain_by_thr[(arm, 2.0)][2] == 3:
            fails.append("%s has 3/3 positive seeds at thr=2.0" % arm)

    print("\n" + "=" * 78)
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        print("VERDICT: FAIL")
        return 1
    print("全部内部一致性检查通过：21 格齐全、记录值回归护栏命中、"
          "S3 两侧 CI 全部跨零、六臂无一 3/3 正增益。")
    print("裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力。")
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
