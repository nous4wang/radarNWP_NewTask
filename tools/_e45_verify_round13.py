"""E45 独立复核（第 13 轮）：从原始产物独立重算 S3 裁决 / 3-seed 统计 / 机制读数。

设计原则（继承前几轮教训）：
  1. 不 import 任何旧复核脚本的函数，避免旧口径 bug 静默传染；
  2. 结构先行：不凭猜写键路径，先打印真实键名再提取；
  3. 双产物互校：scores_all.json 与 scores_summary.json 互相独立重算比对；
  4. 机制读数逐臂逐 seed 对读 diag_probe_s{0,1,2}.json（读数在 arms/<臂>/readouts/ 下）；
  5. 只报中位数 + 通过率，单点分数不单独作为结论。

预注册判据（冻结，见 HANDOVER_E45.md / E45 实验文件）：
  A1/A2   : corr(mask,amp) >= 0 且 mask_at_peak > 0.5
  A3a/A3b : grad_top1pct_share < 0.5 且 field_max_mm <= 30
  A4/A4b  : scale_cv >= 0.05 且 scale_truth_corr >= 0.3 且 scale_is_constant is False
             （mean_out_equals_T 不成立 → 记实现缺陷，不作机制结论）
地板 0.0924（持久性，seed-0 口径）｜Oracle 0.1653
"""
import json
import math
import statistics
from pathlib import Path

BASE = Path("tools/_evals/e45")
SEEDS = [0, 1, 2]
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL7 = ["B0"] + ARMS
FLOOR = 0.0924
ORACLE = 0.1653


def load(name):
    with open(BASE / name, encoding="utf-8") as fh:
        return json.load(fh)


def num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def ci_of(d):
    """从一个 7 键统计块里取 (lo, hi) —— 不猜键名，按 'ci'+'lo'/'hi' 特征匹配。"""
    cands = [k for k in d if "ci" in k.lower()]
    lo = next((k for k in cands if "lo" in k.lower()), None)
    hi = next((k for k in cands if "hi" in k.lower()), None)
    if lo is None or hi is None:
        return None, None, cands
    return num(d[lo]), num(d[hi]), cands


def gain_of(d):
    g = [k for k in d if "gain" in k.lower() and "ci" not in k.lower() and "rel" not in k.lower()]
    return (num(d[g[0]]) if g else None), g


def main():
    ok_all = True

    # ---------- 0. 结构先行：打印 S3 统计块的真实键名 ----------
    s3 = load("s3_equalbias.json")
    probe_keys = list(next(iter(s3["arms"].values()))["raw"].keys())
    print("=" * 78)
    print("0. S3 统计块真实键名（raw / equal_bias 各 %d 键）" % len(probe_keys))
    print("=" * 78)
    print("  ", probe_keys)

    # ---------- A. S3 等偏置对照裁决（独立复核） ----------
    print()
    print("=" * 78)
    print("A. S3 等偏置对照：原始增益 CI 与配平后增益 CI 是否跨零")
    print("=" * 78)
    print(f"  阈值 {s3['threshold_mm']} mm｜epoch {s3['epoch']}｜对照 {s3['control']}"
          f"｜n_boot {s3['n_boot']}｜rng_seed {s3['rng_seed']}")
    s3_rows = {}
    for arm in ARMS:
        a = s3["arms"][arm]
        r_lo, r_hi, _ = ci_of(a["raw"])
        e_lo, e_hi, _ = ci_of(a["equal_bias"])
        r_g, _ = gain_of(a["raw"])
        e_g, _ = gain_of(a["equal_bias"])
        raw_cross = (r_lo is not None and r_hi is not None and r_lo <= 0.0 <= r_hi)
        eq_cross = (e_lo is not None and e_hi is not None and e_lo <= 0.0 <= e_hi)
        v = a.get("s3_verdict")
        # 判负的充要条件：原始或配平后至少一侧 CI 跨零
        neg = raw_cross or eq_cross
        consistent = (not neg) or (v is not None)
        if not consistent:
            ok_all = False
        s3_rows[arm] = (r_g, r_lo, r_hi, e_g, e_lo, e_hi, v)
        print(f"  {arm:4s} raw  gain={r_g} CI=[{r_lo}, {r_hi}] 跨零={raw_cross}")
        print(f"       equal gain={e_g} CI=[{e_lo}, {e_hi}] 跨零={eq_cross}  "
              f"verdict={v}  => 判负={'是' if neg else '否'}")
    n_neg = sum(1 for arm in ARMS if (s3_rows[arm][1] is not None and s3_rows[arm][2] is not None
                                     and s3_rows[arm][1] <= 0 <= s3_rows[arm][2])
                or (s3_rows[arm][4] is not None and s3_rows[arm][5] is not None
                    and s3_rows[arm][4] <= 0 <= s3_rows[arm][5]))
    print(f"  => 判负臂数 = {n_neg}/6")

    # ---------- B. 3-seed 统计：从 scores_all.json 独立重算，与 scores_summary.json 互校 ----------
    print()
    print("=" * 78)
    print("B. 3-seed 中位数 / 过地板通过率（独立重算 scores_all.json，再与 scores_summary.json 互校）")
    print("=" * 78)
    allsc = load("scores_all.json")
    summ = load("scores_summary.json")
    csi, bias = {}, {}
    for arm in ALL7:
        csi[arm] = [allsc[f"{arm}_s{s}_ep00"]["csi_at_2.0"] for s in SEEDS]
        bias[arm] = [allsc[f"{arm}_s{s}_ep00"]["bias_at_2.0"] for s in SEEDS]
    med_c = {a: statistics.median(csi[a]) for a in ALL7}
    med_b = {a: statistics.median(bias[a]) for a in ALL7}
    npass = {a: sum(1 for v in csi[a] if v > FLOOR) for a in ALL7}
    print(f"  {'臂':5s} {'median CSI@2.0':>15s} {'range':>17s} {'median BIAS':>12s} "
          f"{'过地板':>8s} {'超Oracle':>8s} {'离散倍数':>8s}  summary互校")
    for arm in ALL7:
        rng = f"{min(csi[arm]):.4f}-{max(csi[arm]):.4f}"
        spread = max(csi[arm]) / min(csi[arm])
        no = sum(1 for v in csi[arm] if v > ORACLE)
        s = summ.get(arm, {})
        match = (abs(s.get("median_csi_at_2.0", -1) - med_c[arm]) < 1e-12
                 and abs(s.get("median_bias_at_2.0", -1) - med_b[arm]) < 1e-12
                 and s.get("pass_rate_over_floor") == f"{npass[arm]}/3")
        if not match:
            ok_all = False
        print(f"  {arm:5s} {med_c[arm]:15.4f} {rng:>17s} {med_b[arm]:12.4f} "
              f"{npass[arm]}/3{'':4s} {no}/3{'':4s} {spread:8.2f}x  "
              f"{'一致' if match else '不一致!'}")
    higher = [a for a in ARMS if med_c[a] > med_c["B0"]]
    print(f"  => 中位数高于对照 B0 的臂：{higher if higher else '无'}")
    for a in higher:
        print(f"     {a}: median BIAS={med_b[a]:.4f}（对照 {med_b['B0']:.4f}）"
              f"  ⇒ {'BIAS 同时爆表' if med_b[a] > med_b['B0'] * 1.5 else 'BIAS 未爆表'}")
    print(f"  => 对照 B0 自己过地板 {npass['B0']}/3 ⇒ "
          f"{'地板口径不稳（不能单独用作晋级依据）' if npass['B0'] < 3 else '地板口径稳定'}")

    # S3 原始增益与「逐 seed 配对差的中位数」互校（自洽性）
    print()
    print("  S3 raw gain  vs  独立重算的逐 seed 配对差中位数 / 中位数之差：")
    for arm in ARMS:
        paired = statistics.median([csi[arm][i] - csi["B0"][i] for i in range(3)])
        diffmed = med_c[arm] - med_c["B0"]
        print(f"    {arm:4s} s3_raw={s3_rows[arm][0]}  paired_median={paired:+.4f}  "
              f"median_diff={diffmed:+.4f}")

    # ---------- C. 机制读数（逐臂逐 seed，对读原始探针） ----------
    print()
    print("=" * 78)
    print("C. 机制读数逐臂逐 seed（读数在 arms/<臂>/readouts/ 下）")
    print("=" * 78)
    probes = {s: load(f"diag_probe_s{s}.json") for s in SEEDS}
    print("  探针 arms 键：", list(probes[0].get("arms", {}).keys()), "（不含 B0，对照只在 S3 里）")
    readout_keys = sorted({k for s in SEEDS for a in probes[s]["arms"].values()
                            for k in (a.get("readouts") or {})})
    print("  读数键全集：", readout_keys)

    def pick(arm, seed, key):
        a = probes[seed].get("arms", {}).get(arm)
        if a is None:
            return None
        return (a.get("readouts") or {}).get(key)

    print()
    print("  C1 门指向 A1/A2：corr(mask,amp) >= 0 且 mask_at_peak > 0.5")
    for arm in ["A1", "A2"]:
        for s in SEEDS:
            c, m = num(pick(arm, s, "corr_mask_amp")), num(pick(arm, s, "mask_at_peak"))
            dry, res = num(pick(arm, s, "mask_dry_frac_lt_0p1")), num(pick(arm, s, "corr_mask_resid"))
            ok = c is not None and c >= 0 and m is not None and m > 0.5
            print(f"    {arm} s{s}: corr(mask,amp)={c}  mask_at_peak={m}  dry_frac={dry}  "
                  f"corr_mask_resid={res}  => {'PASS' if ok else 'FAIL'}")
    print()
    print("  C2 拓扑 A3a/A3b：grad_top1pct_share < 0.5 且 field_max_mm <= 30")
    for arm in ["A3a", "A3b"]:
        for s in SEEDS:
            g, f = num(pick(arm, s, "grad_top1pct_share")), num(pick(arm, s, "field_max_mm"))
            t = num(pick(arm, s, "tail_max_over_p99"))
            ok = g is not None and g < 0.5 and f is not None and f <= 30
            print(f"    {arm} s{s}: grad_top1pct_share={g}  field_max_mm={f}  "
                  f"tail_max_over_p99={t}  => {'PASS' if ok else 'FAIL'}")
    print()
    print("  C3 量通道 A4/A4b：scale_cv >= 0.05 且 scale_truth_corr >= 0.3 且 is_constant is False")
    for arm in ["A4", "A4b"]:
        for s in SEEDS:
            cv, tc = num(pick(arm, s, "scale_cv")), num(pick(arm, s, "scale_truth_corr"))
            ic, mt = pick(arm, s, "scale_is_constant"), pick(arm, s, "mean_out_equals_T")
            ok = (cv is not None and cv >= 0.05 and tc is not None and tc >= 0.3
                  and ic is False)
            print(f"    {arm} s{s}: scale_cv={cv}  scale_truth_corr={tc}  "
                  f"is_constant={ic}  mean_out_equals_T={mt}  => "
                  f"{'PASS' if ok else 'FAIL'}"
                  f"{'' if mt is True else '  [mean(out)=T 不成立→实现缺陷]'}")

    print()
    print("  C4 门那组第三读数 maxratio_as_trained（探针缺失检查）")
    for arm in ["A1", "A2"]:
        vals = [pick(arm, s, "maxratio_as_trained") for s in SEEDS]
        print(f"    {arm}: {vals}  => "
              f"{'全为 nan/缺失（读数缺失）' if all(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals) else '有值'}")

    # ---------- D. 逐臂 3-seed 一致性小结 ----------
    print()
    print("=" * 78)
    print("D. 逐臂 3-seed 预注册判据一致性")
    print("=" * 78)
    verdict = {}
    for arm in ARMS:
        gate = []
        for s in SEEDS:
            c, m = num(pick(arm, s, "corr_mask_amp")), num(pick(arm, s, "mask_at_peak"))
            g, f = num(pick(arm, s, "grad_top1pct_share")), num(pick(arm, s, "field_max_mm"))
            cv, tc = num(pick(arm, s, "scale_cv")), num(pick(arm, s, "scale_truth_corr"))
            if arm in ("A1", "A2"):
                gate.append(c is not None and c >= 0 and m is not None and m > 0.5)
            elif arm in ("A3a", "A3b"):
                gate.append(g is not None and g < 0.5 and f is not None and f <= 30)
            else:
                gate.append(cv is not None and cv >= 0.05 and tc is not None and tc >= 0.3
                            and pick(arm, s, "scale_is_constant") is False)
        verdict[arm] = gate
        print(f"  {arm:4s}: 逐 seed = {gate}  => {sum(gate)}/3 通过")
    print()
    print(f"  量通道非退化：A4 {sum(verdict['A4'])}/3 seed；A4b {sum(verdict['A4b'])}/3 seed")
    print(f"  ⇒「量通道三 seed 都没退化成常数」{'成立' if all(verdict['A4']) and all(verdict['A4b']) else '不成立（必须分臂表述）'}")
    print(f"  ⇒ 拓扑支：{'仍结掉（两臂均有 seed 不过 field_max<=30）' if not (all(verdict['A3a']) or all(verdict['A3b'])) else '需重新看'}")
    print(f"  ⇒ 门指向：{'两臂 3/3 过判据 ⇒ 结构上已解决' if all(verdict['A1']) and all(verdict['A2']) else '存在不过的 seed'}")

    print()
    print("=" * 78)
    print(f"VERDICT: {'PASS' if ok_all else 'FAIL'}  "
          f"（互校一致={ok_all}；S3 判负臂数={n_neg}/6）")
    print("=" * 78)


if __name__ == "__main__":
    main()
