"""E45 独立复核（第 12 轮）：逐臂逐 seed 重算机制读数。

不复用前几轮脚本的任何提取函数，直接从三份原始探针 JSON 重取，
并把每条读数与预注册判据逐条比对，最后给出 PASS/FAIL 判定表。

预注册判据（来自 HANDOVER_E45.md / E45 实验文件，冻结）：
  A1/A2 : corr(mask,amp) >= 0 且 mask_at_peak > 0.5
  A3a/A3b: grad_top1pct_share < 0.5 且 field_max_mm <= 30
  A4/A4b: scale_cv >= 0.05 且 scale_truth_corr >= 0.3
          且 scale_is_constant 必须为 False
          且 mean_out_equals_T 必须为 True（否则记实现缺陷，不作机制结论）
"""
import json
import math
from pathlib import Path

BASE = Path("tools/_evals/e45")
SEEDS = [0, 1, 2]
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]


def load(name):
    with open(BASE / name, encoding="utf-8") as fh:
        return json.load(fh)


def num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def main():
    probes = {s: load(f"diag_probe_s{s}.json") for s in SEEDS}
    rows = {}
    for s in SEEDS:
        arms = probes[s].get("arms", {})
        for arm in ARMS:
            a = arms.get(arm)
            if a is None:
                print(f"MISSING ARM: seed {s} arm {arm}")
                continue
            r = a.get("readouts", {}) or {}
            rows.setdefault(arm, {})[s] = r

    def pick(arm, seed, key):
        return rows.get(arm, {}).get(seed, {}).get(key)

    print("=" * 78)
    print("A. 门指向（A1/A2）：corr(mask,amp) >= 0 且 mask_at_peak > 0.5")
    print("=" * 78)
    for arm in ["A1", "A2"]:
        for s in SEEDS:
            c = num(pick(arm, s, "corr_mask_amp"))
            m = num(pick(arm, s, "mask_at_peak"))
            dry = num(pick(arm, s, "mask_dry_frac_lt_0p1"))
            res = num(pick(arm, s, "corr_mask_resid"))
            ok = (c is not None and c >= 0) and (m is not None and m > 0.5)
            print(
                f"  {arm} s{s}: corr(mask,amp)={c}  mask_at_peak={m}  "
                f"dry_frac={dry}  corr_mask_resid={res}  => {'PASS' if ok else 'FAIL'}"
            )

    print("=" * 78)
    print("B. 拓扑（A3a/A3b）：grad_top1pct_share < 0.5 且 field_max_mm <= 30")
    print("=" * 78)
    for arm in ["A3a", "A3b"]:
        for s in SEEDS:
            g = num(pick(arm, s, "grad_top1pct_share"))
            f = num(pick(arm, s, "field_max_mm"))
            t = num(pick(arm, s, "tail_max_over_p99"))
            ok = (g is not None and g < 0.5) and (f is not None and f <= 30)
            print(
                f"  {arm} s{s}: grad_top1pct_share={g}  field_max_mm={f}  "
                f"tail_max_over_p99={t}  => {'PASS' if ok else 'FAIL'}"
            )

    print("=" * 78)
    print("C. 量通道（A4/A4b）：scale_cv >= 0.05 且 scale_truth_corr >= 0.3；"
          "scale_is_constant 必须 False；mean_out_equals_T 必须 True")
    print("=" * 78)
    for arm in ["A4", "A4b"]:
        for s in SEEDS:
            cv = num(pick(arm, s, "scale_cv"))
            tc = num(pick(arm, s, "scale_truth_corr"))
            ic = pick(arm, s, "scale_is_constant")
            mt = pick(arm, s, "mean_out_equals_T")
            deg_ok = (cv is not None and cv >= 0.05) and (tc is not None and tc >= 0.3)
            nondeg = (ic is False)
            print(
                f"  {arm} s{s}: scale_cv={cv}  scale_truth_corr={tc}  "
                f"scale_is_constant={ic}  mean_out_equals_T={mt}  "
                f"=> 不退化={'PASS' if deg_ok else 'FAIL'}  非退化标记={'PASS' if nondeg else 'FAIL'}  "
                f"mean(out)=T={'PASS' if mt is True else 'FAIL/缺陷'}"
            )

    print("=" * 78)
    print("D. 门那组第三个读数 maxratio_as_trained（探针缺失检查）")
    print("=" * 78)
    for arm in ["A1", "A2"]:
        for s in SEEDS:
            v = pick(arm, s, "maxratio_as_trained")
            print(f"  {arm} s{s}: maxratio_as_trained={v}")

    print("=" * 78)
    print("E. 逐臂 3-seed 一致性小结")
    print("=" * 78)
    verdict = {}
    for arm in ARMS:
        gate = []
        for s in SEEDS:
            r = rows.get(arm, {}).get(s, {})
            if arm in ("A1", "A2"):
                c, m = num(r.get("corr_mask_amp")), num(r.get("mask_at_peak"))
                gate.append(c is not None and c >= 0 and m is not None and m > 0.5)
            elif arm in ("A3a", "A3b"):
                g, f = num(r.get("grad_top1pct_share")), num(r.get("field_max_mm"))
                gate.append(g is not None and g < 0.5 and f is not None and f <= 30)
            else:
                cv, tc = num(r.get("scale_cv")), num(r.get("scale_truth_corr"))
                gate.append(
                    cv is not None and cv >= 0.05 and tc is not None and tc >= 0.3
                    and r.get("scale_is_constant") is False
                )
        verdict[arm] = gate
        print(f"  {arm}: 预注册判据逐 seed = {gate}  => "
              f"{'3/3 通过' if all(gate) else f'{sum(gate)}/3 通过'}")

    # 量通道主张必须分臂陈述：只有 A4 三 seed 全过，A4b 只在 2/3 seed 成立
    print()
    a4 = verdict.get("A4", [])
    a4b = verdict.get("A4b", [])
    print(f"  量通道非退化：A4 {sum(a4)}/3 seed 成立；A4b {sum(a4b)}/3 seed 成立")
    print(f"  => 「量通道三 seed 都没退化成常数」这句话{'成立' if all(a4) and all(a4b) else '不成立（必须分臂表述）'}")


if __name__ == "__main__":
    main()
