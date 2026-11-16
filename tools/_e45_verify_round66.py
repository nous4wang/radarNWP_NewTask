#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 66 轮：epoch 轴的**首个受控判读**（s0 整列 21 格 + 对照第二 seed）。

纪律（沿用第 40–65 轮）：
  * 写回在校验之后；断言不许假设数据会迎合结论。
  * 事后诊断，不构成新判据，不改变任何裁决。
  * 报「中位数 + 通过率」；配对口径与差值口径并列。
  * 凡报「某轴上变了」的结论，必须同时给出该轴的**零模型地板**
    （第 49 轮纪律：不给地板的跨轴比较一律不得单独成立）。
  * 单点分数不作裁决；n=1 的格只能报**方向与幅度**，不得作臂级结论。

口径说明（关键）：
  * 已回取的是 **seed 0 整列**：7 臂 × 3 epoch = 21 格，外加对照臂 B0 的
    **第二 seed**（B0_s1 × 3 epoch = 3 格）⇒ 一共 24 份 eval。
  * 机制臂只有 seed 0 ⇒ **臂内跨 epoch 的配对是同 seed 固定**（不跨种子），
    n=1；因此本轮**不给任何臂级结论**，只给
      (a) 逐臂逐 epoch 的方向与幅度；
      (b) 与「种子轴」量级对照（同一批格子里，纯换 seed 会带来多大的 ΔCSI）。
  * 零模型地板 = **同一 epoch 下换 seed**（B0_s0 vs B0_s1）⇒ 纯种子噪声，
    不含任何机制。
"""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "tools" / "_evals" / "e45"
COL = EV / "epoch_axis_s0col"
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
MECH_ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
EPOCHS = ["ep00", "ep01", "ep02"]
FLOOR = 0.0924

# ---------------------------------------------------------------- 读取

def read_cell(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    thr = obj["thresholds"]["2.0"]
    return {
        "csi": float(thr["csi"]),
        "bias": float(thr["bias"]),
        "pod": float(thr["pod"]),
        "far": float(thr["far"]),
        "n_samples": int(obj["n_samples"]),
    }


cells = {}
for ev_path in sorted((COL / "evals").glob("*.eval.json")):
    stem = ev_path.name[: -len(".eval.json")]
    arm, seed, epoch = stem.split("_")   # seed 形如 "s0"，epoch 形如 "ep00"
    key = f"{arm}_{seed}_{epoch}"
    cells[key] = read_cell(ev_path)

probes = {}
for pr_path in sorted((COL / "probe").glob("*.json")):
    stem = pr_path.name[: -len(".json")]
    # 文件名形如 diag_probe_s0_ep00.json ⇒ 取最后两段 (seed, epoch)
    seed, epoch = stem.split("_")[-2:]
    probes[f"{seed}_{epoch}"] = json.loads(pr_path.read_text(encoding="utf-8"))

out = {
    "round": 66,
    "purpose": "E45 epoch axis -- first controlled read (seed-0 column complete)",
    "is_post_hoc_diagnostic": True,
    "creates_criterion": False,
    "changes_e45_verdict": False,
    "n_eval_cells_read": len(cells),
    "n_probe_files_read": len(probes),
}

# ---------------------------------------------------------------- 守卫

s3 = json.loads((EV / "s3_equalbias.json").read_text(encoding="utf-8"))
cross = []
verdicts = {}
points = {}
for arm, blk in s3["arms"].items():
    for axis in ("raw", "equal_bias"):
        a = blk[axis]
        cross.append(bool(a["crosses_zero"]))
        points[f"{arm}_{axis}"] = float(a["point"])
    verdicts[arm] = blk.get("s3_verdict")
out["guard"] = {
    "s3_intervals_total": len(cross),
    "s3_intervals_crossing_zero": sum(cross),
    "s3_all_cross_zero": all(cross),
    "s3_verdicts": verdicts,
    "qualified_arms": [a for a, v in verdicts.items() if v not in ("GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH")],
    "s3_points": points,
}

# 头条表复算：从累积快照按 key 过滤（scores_summary_s{0,1,2} 是累积快照，
# 第 64 轮已登记）⇒ 每 seed 只取 key 含 "_s<seed>_" 的格。
summary = json.loads((EV / "scores_summary.json").read_text(encoding="utf-8"))
per_seed = {}
for seed in (0, 1, 2):
    snap = json.loads((EV / f"scores_summary_s{seed}.json").read_text(encoding="utf-8"))
    for k, v in snap.items():
        if f"_s{seed}_" in k:
            per_seed.setdefault(k, {})[seed] = v

headline_diff = 0.0
headline_rows = {}
for arm, blk in summary.items():
    vals = []
    for seed in (0, 1, 2):
        key = f"{arm}_s{seed}_ep00"
        assert key in per_seed, f"缺 key {key}"
        vals.append(float(per_seed[key][seed]["csi_at_2.0"]))
    med = statistics.median(vals)
    pr = sum(1 for v in vals if v >= FLOOR)
    headline_diff = max(headline_diff, abs(med - float(blk["median_csi_at_2.0"])))
    headline_rows[arm] = {
        "median_csi_at_2.0": med,
        "pass_rate_over_floor": f"{pr}/{len(vals)}",
        "per_seed": {f"s{s}": v for s, v in zip((0, 1, 2), vals)},
    }
out["guard"]["headline_table_max_abs_diff"] = headline_diff
out["guard"]["headline_recomputed"] = headline_rows

# 同协议基准（探针侧）：epoch 轴的 ep00 探针 vs 已发布的 E45 s0 探针
pub_probe = json.loads((EV / "diag_probe_s0.json").read_text(encoding="utf-8"))
probe_diff = 0.0
probe_matched = 0
for arm in MECH_ARMS:
    a_pub = pub_probe["arms"][arm]["readouts"]
    a_new = probes["s0_ep00"]["arms"][arm]["readouts"]
    for k in a_pub:
        pv, nv = a_pub[k], a_new.get(k)
        if isinstance(pv, (int, float)) and isinstance(nv, (int, float)):
            if math.isnan(pv) and math.isnan(nv):
                probe_matched += 1
                continue
            if math.isnan(pv) or math.isnan(nv):
                continue
            probe_diff = max(probe_diff, abs(float(pv) - float(nv)))
            probe_matched += 1
out["guard"]["probe_same_protocol_max_abs_diff"] = probe_diff
out["guard"]["probe_same_protocol_n_compared"] = probe_matched

# ---------------------------------------------------------------- epoch 轴

# 臂级中位（这里 seed 0 列 ⇒ 每臂每 epoch 只有 1 格 ⇒ 中位＝该格值，
# 但仍按「中位数」口径登记，便于 3 seed 全回取后同口径续算）
axis = {}
for arm in ARMS:
    per_ep = {}
    for ep in EPOCHS:
        k = f"{arm}_s0_{ep}"
        if k not in cells:
            continue
        c = cells[k]
        per_ep[ep] = {
            "median_csi_at_2.0": c["csi"],
            "median_bias_at_2.0": c["bias"],
            "pod_at_2.0": c["pod"],
            "far_at_2.0": c["far"],
            "pass_floor": 1 if c["csi"] >= FLOOR else 0,
            "n_seeds": 1,
        }
    if per_ep:
        axis[arm] = per_ep
out["epoch_axis_seed0"] = axis

# 列中位（七个臂的中位）——差值口径的载体
col_median = {}
for ep in EPOCHS:
    meds = [axis[a][ep]["median_csi_at_2.0"] for a in ARMS if ep in axis.get(a, {})]
    bia = [axis[a][ep]["median_bias_at_2.0"] for a in ARMS if ep in axis.get(a, {})]
    if meds:
        col_median[ep] = {
            "arm_median_median": statistics.median(meds),
            "arm_median_min": min(meds),
            "arm_median_max": max(meds),
            "arm_median_mean": statistics.mean(meds),
            "n_arms": len(meds),
            "arms_over_floor": sum(1 for m in meds if m >= FLOOR),
            "bias_median_median": statistics.median(bia),
        }
out["column_medians"] = col_median

# 配对口径：同臂同 seed 跨 epoch 相对 ep00 的 Δ
paired = {}
for arm in ARMS:
    if "ep00" not in axis.get(arm, {}):
        continue
    base = axis[arm]["ep00"]["median_csi_at_2.0"]
    for ep in ("ep01", "ep02"):
        if ep not in axis.get(arm, {}):
            continue
        d = axis[arm][ep]["median_csi_at_2.0"] - base
        db = axis[arm][ep]["median_bias_at_2.0"] - axis[arm]["ep00"]["median_bias_at_2.0"]
        paired.setdefault(arm, {})[ep] = {
            "delta_csi": d,
            "delta_bias": db,
            "sign": "pos" if d > 0 else ("neg" if d < 0 else "zero"),
        }
out["paired_delta_vs_ep00"] = paired

# 差值口径：列中位相减
diff_conv = {}
for ep in ("ep01", "ep02"):
    if ep in col_median and "ep00" in col_median:
        diff_conv[ep] = {
            "column_median_delta": col_median[ep]["arm_median_median"] - col_median["ep00"]["arm_median_median"],
            "column_mean_delta": col_median[ep]["arm_median_mean"] - col_median["ep00"]["arm_median_mean"],
        }
out["difference_convention"] = diff_conv

# 峰值 epoch（每臂）
peak = {}
for arm in ARMS:
    if arm not in axis:
        continue
    meds = {ep: axis[arm][ep]["median_csi_at_2.0"] for ep in axis[arm]}
    best = max(meds, key=lambda e: meds[e])
    peak[arm] = {
        "median_by_epoch": meds,
        "best_epoch": best,
        "ep00_is_best": best == "ep00",
        "ep01_minus_ep00": meds.get("ep01", float("nan")) - meds["ep00"],
        "ep02_minus_ep00": meds.get("ep02", float("nan")) - meds["ep00"],
        "slope_per_epoch": (meds["ep02"] - meds["ep00"]) / 2.0 if "ep02" in meds else None,
    }
out["peak_epoch_per_arm"] = peak
out["ep00_is_best_count"] = sum(1 for v in peak.values() if v["ep00_is_best"])
out["n_arms"] = len(peak)

# ---------------------------------------------------------------- 零模型地板

# 零模型 = 对照臂换 seed（B0_s0 vs B0_s1），逐 epoch；纯种子噪声，无机制。
zero = {}
for ep in EPOCHS:
    a = cells.get(f"B0_s0_{ep}")
    b = cells.get(f"B0_s1_{ep}")
    if a and b:
        zero[ep] = {
            "abs_delta_csi": abs(a["csi"] - b["csi"]),
            "delta_csi": b["csi"] - a["csi"],
            "abs_delta_bias": abs(a["bias"] - b["bias"]),
        }
out["zero_model_floor_seed_swap"] = zero
zm = max((v["abs_delta_csi"] for v in zero.values()), default=float("nan"))
out["zero_model_floor_max"] = zm

# 逐臂跨 epoch 的 |Δ| 与零模型地板之比（幅度维度的对照）
ratio = {}
for arm, blk in paired.items():
    for ep, r in blk.items():
        ratio.setdefault(arm, {})[ep] = abs(r["delta_csi"]) / zm if zm and not math.isnan(zm) else None
out["epoch_delta_over_zero_model"] = ratio

# ---------------------------------------------------------------- 机制读数跨 epoch

CRIT = {
    "A1": {"corr_mask_amp": (">=", 0.0), "mask_at_peak": (">", 0.5)},
    "A2": {"corr_mask_amp": (">=", 0.0), "mask_at_peak": (">", 0.5)},
    "A3a": {"grad_top1pct_share": ("<", 0.5), "field_max_mm": ("<=", 30.0)},
    "A3b": {"grad_top1pct_share": ("<", 0.5), "field_max_mm": ("<=", 30.0)},
    "A4": {"scale_cv": (">=", 0.05), "scale_truth_corr": (">=", 0.3)},
    "A4b": {"scale_cv": (">=", 0.05), "scale_truth_corr": (">=", 0.3)},
}
KEEP = ["corr_mask_amp", "mask_at_peak", "mask_dry_frac_lt_0p1", "corr_mask_resid",
        "grad_top1pct_share", "field_max_mm", "tail_max_over_p99",
        "scale_cv", "scale_truth_corr", "scale_mean", "scale_is_constant",
        "mean_out_equals_T", "maxratio_as_trained"]

mech = {}
crit_by_epoch = {}
for arm in MECH_ARMS:
    per_ep = {}
    for ep in EPOCHS:
        blk = probes.get(f"s0_{ep}", {}).get("arms", {}).get(arm, {}).get("readouts", {})
        if not blk:
            continue
        row = {}
        for k in KEEP:
            if k in blk:
                v = blk[k]
                row[k] = (None if (isinstance(v, float) and math.isnan(v)) else v)
        per_ep[ep] = row
        # 预注册判据达标与否（逐读数分分支写，不合并成单一布尔）
        verd = {}
        for rname, (op, thr) in CRIT.get(arm, {}).items():
            v = row.get(rname)
            if v is None:
                verd[rname] = "missing"
                continue
            ok = {"<": v < thr, "<=": v <= thr, ">": v > thr, ">=": v >= thr}[op]
            verd[rname] = ("PASS" if ok else "FAIL") + f" ({rname}={v:.4f} {op} {thr})"
        if verd:
            crit_by_epoch.setdefault(arm, {})[ep] = verd
    mech[arm] = per_ep
out["mechanism_across_epochs"] = mech
out["prereg_criteria_by_epoch"] = crit_by_epoch

# 判据是否随 epoch 翻转（逐读数）
flips = {}
for arm, per_ep in crit_by_epoch.items():
    for rname in CRIT.get(arm, {}):
        states = {ep: per_ep[ep][rname].split()[0] for ep in per_ep if rname in per_ep[ep]}
        if len(set(states.values())) > 1:
            flips.setdefault(arm, {})[rname] = states
out["criterion_flips_across_epoch"] = flips

# ---------------------------------------------------------------- 断言（不假设数据迎合结论）

def assert_eq(a, b, msg):
    assert abs(float(a) - float(b)) < 1e-9, f"{msg}: {a} != {b}"

# 1) 守卫不变
assert out["guard"]["s3_all_cross_zero"] is True, "S3 不再全部跨零 —— 裁决基础变了"
assert out["guard"]["s3_intervals_total"] == 12, "S3 区间数不是 12"
assert out["guard"]["qualified_arms"] == [], f"合格臂集合不再为空: {out['guard']['qualified_arms']}"
# 2) 头条表复算
assert headline_diff < 1e-9, f"头条表复算偏差 {headline_diff}"
# 3) 同协议基准：ep00 重跑必须逐位复现已发布数字（零漂移）
assert probe_diff < 1e-12, f"ep00 探针与已发布探针不一致（漂移 {probe_diff}）"
# 4) 结构性检查：s0 整列必须齐（7 臂 × 3 epoch = 21 格）；其余格是
#    随作业推进陆续落盘的 seed-1 格（只用于零模型地板与后续续算）。
s0_cells = [k for k in cells if k.endswith(("_s0_ep00", "_s0_ep01", "_s0_ep02"))]
assert len(s0_cells) == 21, f"s0 整列格数 {len(s0_cells)} != 21"
for arm in ARMS:
    for ep in EPOCHS:
        assert f"{arm}_s0_{ep}" in cells, f"缺格 {arm}_s0_{ep}"
extra = sorted(k for k in cells if k not in s0_cells)
out["fact_extra_cells_beyond_s0_column"] = extra
# 5) 枚举事实：ep00 是否为峰值（记录计数，不假设方向）
out["fact_ep00_is_best"] = out["ep00_is_best_count"]
out["fact_n_arms"] = out["n_arms"]

# 6) 每臂跨 epoch 的 Δ 必须逐条与直接重算一致（自校）
for arm, blk in paired.items():
    for ep, r in blk.items():
        d = cells[f"{arm}_s0_{ep}"]["csi"] - cells[f"{arm}_s0_ep00"]["csi"]
        assert_eq(r["delta_csi"], d, f"{arm} {ep} 配对差自校失败")

dest = EV / "round66_epoch_axis_controlled.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

# ---------------------------------------------------------------- 控制台摘要
print(f"POS {len(cells)} eval cells, {len(probes)} probe files")
print(f"guard: s3 {out['guard']['s3_intervals_crossing_zero']}/{out['guard']['s3_intervals_total']} cross zero, "
      f"headline max|diff| {headline_diff:.3e}, probe same-protocol max|diff| {probe_diff:.3e}")
print()
print("epoch 轴（seed 0 列；中位 CSI@2.0 / 中位 BIAS@2.0）")
for arm in ARMS:
    if arm not in axis:
        continue
    p = peak[arm]
    print(f"  {arm:4s} ep00={p['median_by_epoch'].get('ep00', float('nan')):.6f} "
          f"ep01={p['median_by_epoch'].get('ep01', float('nan')):.6f} "
          f"ep02={p['median_by_epoch'].get('ep02', float('nan')):.6f} "
          f"best={p['best_epoch']} ep00_is_best={p['ep00_is_best']}")
print()
print(f"零模型地板（换 seed，逐 epoch）: " +
      ", ".join(f"{ep} {v['abs_delta_csi']:.6f}" for ep, v in zero.items()) +
      f"  → max {zm:.6f}")
print("配对 ΔCSI vs ep00 / 零模型比:")
for arm in ARMS:
    if arm not in paired:
        continue
    s = "  ".join(f"{ep}: {paired[arm][ep]['delta_csi']:+.6f} (×{ratio[arm][ep]:.2f})"
                  for ep in ("ep01", "ep02") if ep in paired[arm])
    print(f"  {arm:4s} {s}")
print()
print("差值口径（列中位相减）:", {k: round(v["column_median_delta"], 6) for k, v in diff_conv.items()})
print("判据随 epoch 翻转:", json.dumps(flips, ensure_ascii=False))
print()
print("VERDICT: PASS")
print("written:", dest)
