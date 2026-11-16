#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 90 轮：远端运行状态复核 + S3 护栏「数值级」独立重算 + 头条表「主跑评测文件」路径独立复算。

与既往轮次的差异（不复用既有脚本结论）：
  1. 头条表不取 scores_all.json，改从 **21 份主跑 *.eval.json** 逐格解析复算（另一条数据路径）；
  2. S3 护栏不读 crosses_zero 布尔标志，改由 **ci_lo / ci_hi 的符号包含关系数值级重算**，
     并逐格核对存储标志与数值是否自洽（12 格）；并核对 s3_verdict 文本与 raw 点估符号是否自洽；
  3. 主跑评测文件在本地有多个副本目录（evals_raw / runs / _raw / epoch_axis_s0col），
     逐格核对多副本是否一致（不把「副本数不是 3」误判为缺失）。

只读本地产物；不改远端、不改 docs/。判定口径：中位数 + 通过率；单点分数不作裁决。
"""
import json
import hashlib
import statistics
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
EV = BASE / "_evals" / "e45"
OUT = EV / "round90_status_audit.json"

POS, NEG = [], []


def chk(name, ok, detail):
    (POS if ok else NEG).append({"check": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(("  PASS  " if ok else "  FAIL  ") + name + " | " + str(detail)[:300])
    return ok


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def find_all(obj, key, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            if k == key:
                out.append((p, v))
            out.extend(find_all(v, key, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(find_all(v, key, f"{path}[{i}]"))
    return out


ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SIX = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
FLOOR = 0.0924
ORACLE = 0.1653

# ---------------------------------------------------------------- A. 产物一致性
print("=== A. run 级 13 份产物远端↔本地 md5（本轮实测远端值） ===")
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
mism = []
for fn, want in REMOTE_MD5.items():
    p = EV / fn
    got = md5(p) if p.exists() else None
    if got != want:
        mism.append({"file": fn, "remote": want, "local": got})
chk("A1 run 级 13 份产物远端↔本地 md5 逐位一致", not mism and len(REMOTE_MD5) == 13,
    f"比对 {len(REMOTE_MD5)} 份，差异 {len(mism)} 份" + ("" if not mism else f" -> {mism}"))

# ---------------------------------------------------------------- B. 主跑评测文件路径
print("=== B. 头条表从主跑 *.eval.json 独立复算（不走 scores_all.json） ===")
# 本地主跑评测副本目录（内容应当一致；某目录只含子集不算「缺失」）
MAIN_EVAL_DIRS = {
    "_evals/e45/evals_raw/evals",
    "_evals/e45/runs/e45/evals",
    "_evals/e45/_raw/runs/e45/evals",
    "_evals/e45/epoch_axis_s0col/evals",
}
grid = {}
copies = {}
for p in sorted(EV.rglob("*.eval.json")):
    if "epoch_axis" in p.parts:            # epoch 轴的 63 格是另一张表，排除
        continue
    m = re.match(r"^([A-Za-z0-9]+)_s(\d+)_ep(\d+)\.eval\.json$", p.name)
    if not m:
        continue
    arm, seed, ep = m.group(1), int(m.group(2)), m.group(3)
    if ep != "00":
        continue
    d = jload(p)
    th = d.get("thresholds", {}).get("2.0", {})      # 主跑评测文件的指标布局
    if not isinstance(th, dict) or th.get("csi") is None:
        continue
    val = float(th["csi"])
    try:
        tag = p.parent.relative_to(BASE).as_posix()
    except ValueError:
        tag = p.parent.as_posix()
    copies.setdefault((arm, seed), []).append((tag, val))
    grid.setdefault(arm, {})[seed] = val

chk("B1 主跑评测网格 = 7 臂 × 3 seed × ep00（21 格）",
    sorted(grid.keys()) == sorted(ARMS) and all(sorted(v.keys()) == [0, 1, 2] for v in grid.values())
    and sum(len(v) for v in grid.values()) == 21,
    f"臂 {sorted(grid.keys())}；格数 {sum(len(v) for v in grid.values())}")

# 路径自校验：本路径解析出的每格必须与 scores_all.json 逐位相同（否则说明口径取错）
scores_all = jload(EV / "scores_all.json")
path_dev = 0.0
for arm in ARMS:
    for s in (0, 1, 2):
        key = f"{arm}_s{s}_ep00"
        if key in scores_all:
            path_dev = max(path_dev, abs(float(scores_all[key]["csi_at_2.0"]) - grid[arm][s]))
chk("B2 主跑评测文件解析路径自校验：与 scores_all.json 逐格 max|dev| < 1e-12",
    path_dev < 1e-12, f"max|dev| = {path_dev:.3e}（21 格）")

copy_dev = 0.0
for k, lst in copies.items():
    vals = [v for _, v in lst]
    copy_dev = max(copy_dev, max(vals) - min(vals))
tags = sorted({t for lst in copies.values() for t, _ in lst})
chk("B3 主跑评测多副本逐格一致（副本集合符合预期、任一格 max|spread| = 0）",
    copy_dev == 0.0 and len(copies) == 21 and set(tags) == MAIN_EVAL_DIRS,
    f"格数 {len(copies)}；每格副本数 {sorted({len(v) for v in copies.values()})}；"
    f"max|spread| = {copy_dev:.3e}；副本目录 {tags}")

pub = jload(EV / "scores_summary.json")
recalc = {}
maxdev = 0.0
for arm in ARMS:
    vals = [grid[arm][s] for s in (0, 1, 2)]
    med = statistics.median(vals)
    rate = sum(1 for v in vals if v > FLOOR)
    pm = pub[arm]["median_csi_at_2.0"]
    maxdev = max(maxdev, abs(med - pm))
    recalc[arm] = {"median_csi_at_2.0": med, "over_floor": f"{rate}/3",
                   "pub_median": pm, "pub_rate": pub[arm]["pass_rate_over_floor"],
                   "seeds": {f"s{s}": grid[arm][s] for s in (0, 1, 2)}}
chk("B4 七臂中位数复算（主跑评测文件路径）max|dev| < 1e-9", maxdev < 1e-9, f"max|dev| = {maxdev:.3e}")
chk("B5 过地板通过率逐臂一致（地板 0.0924 为单一种子口径）",
    all(recalc[a]["over_floor"] == recalc[a]["pub_rate"] for a in ARMS),
    "; ".join(f"{a}:{recalc[a]['over_floor']}(median {recalc[a]['median_csi_at_2.0']:.6f})" for a in ARMS))
chk("B6 对照臂自身也只有 2/3 过地板 -> 通过率不可单独用于晋级",
    recalc["B0"]["over_floor"] == "2/3", f"B0 {recalc['B0']['over_floor']} median={recalc['B0']['median_csi_at_2.0']:.6f}")
over_oracle = sorted(a for a in ARMS if any(v > ORACLE for v in recalc[a]["seeds"].values()))
chk("B7 超 Oracle 上界 0.1653 的臂（按逐 seed 判定；须 S3 才判，不可单点定论）",
    over_oracle == ["A3b", "A4"], f"over_oracle={over_oracle}")

# ---------------------------------------------------------------- C. S3 数值级重算
print("=== C. S3 等偏置护栏：数值级重算（不看布尔标志，看 ci_lo/ci_hi 符号） ===")
s3 = jload(EV / "s3_equalbias.json")
chk("C1 S3 护栏参数未被改写",
    s3.get("threshold_mm") == 2.0 and s3.get("control") == "B0" and s3.get("n_boot") == 10000
    and s3.get("rng_seed") == 20261109 and s3.get("epoch") == "00",
    f"thr={s3.get('threshold_mm')} control={s3.get('control')} n_boot={s3.get('n_boot')} "
    f"rng={s3.get('rng_seed')} epoch={s3.get('epoch')}")
s3_cells = []
flag_mismatch = []
for arm in sorted(s3["arms"].keys()):
    node = s3["arms"][arm]
    for side in ("raw", "equal_bias"):
        c = node[side]
        lo, hi = float(c["ci_lo"]), float(c["ci_hi"])
        num_cross = (lo <= 0.0 <= hi)                      # 数值级：区间是否包含 0
        flag = bool(c["crosses_zero"])
        if num_cross != flag:
            flag_mismatch.append(f"{arm}/{side}")
        s3_cells.append({"arm": arm, "side": side, "point": float(c["point"]), "ci_lo": lo, "ci_hi": hi,
                         "stored_flag": flag, "numeric_cross_zero": num_cross,
                         "n_boot": c.get("n_boot"), "n_seeds": c.get("n_seeds"), "n_windows": c.get("n_windows")})
chk("C2 CI 格数 = 12（6 臂 × raw/equal_bias）", len(s3_cells) == 12, f"实得 {len(s3_cells)}")
chk("C3 存储的 crosses_zero 标志与 ci_lo/ci_hi 数值自洽（12/12）",
    not flag_mismatch, f"不自洽格: {flag_mismatch or '无'}")
chk("C4 逐格护栏参数一致（n_boot=10000 / n_seeds=3 / n_windows=538）",
    all(c["n_boot"] == 10000 and c["n_seeds"] == 3 and c["n_windows"] == 538 for c in s3_cells),
    "12 格全部 n_boot=10000 n_seeds=3 n_windows=538")
all_cross = all(c["numeric_cross_zero"] for c in s3_cells)
chk("C5 12/12 区间数值上全部包含 0 -> 六臂判负、合格臂集合为空", all_cross,
    "; ".join(f"{c['arm']}/{c['side']}=[{c['ci_lo']:.4f},{c['ci_hi']:.4f}]" for c in s3_cells))
qualifying = sorted({c["arm"] for c in s3_cells if not c["numeric_cross_zero"]})
chk("C6 合格臂集合为空（没有任何一臂的 CI 排除 0）", qualifying == [], f"qualifying={qualifying}")

# s3_verdict 是预注册的两分支文本：raw 点估为正 ⇒ 增益存在但被等偏置判为面积/幅度效应；
# raw 点估为负 ⇒ 本来就没有增益。两者都是「判负」，但失效模式不同，必须分分支报告。
verdicts = {a: s3["arms"][a].get("s3_verdict") for a in s3["arms"]}
verdict_consistent = []
for arm, v in verdicts.items():
    raw_point = float(s3["arms"][arm]["raw"]["point"])
    if v == "GAIN_WAS_AREA_OR_AMPLITUDE":
        ok = raw_point > 0
    elif v == "NO_GAIN_TO_BEGIN_WITH":
        ok = raw_point <= 0
    else:
        ok = False
    verdict_consistent.append({"arm": arm, "verdict": v, "raw_point": raw_point, "consistent": ok})
chk("C7 六臂 s3_verdict 文本与 raw 点估符号自洽（两分支皆为判负）",
    all(x["consistent"] for x in verdict_consistent),
    "; ".join(f"{x['arm']}={x['verdict']}(raw {x['raw_point']:+.6f})" for x in verdict_consistent))
branch_counts = {}
for x in verdict_consistent:
    branch_counts[x["verdict"]] = branch_counts.get(x["verdict"], 0) + 1
chk("C8 失效模式分分支清点（面积/幅度效应 vs 本来就无增益）",
    sum(branch_counts.values()) == 6,
    f"{branch_counts}；六臂合计 {sum(branch_counts.values())}（均判负）")

# ---------------------------------------------------------------- D. 机制判据重判
print("=== D. 预注册机制判据逐臂逐 seed 重判（从 diag_probe 原始 JSON） ===")
diag = {s: jload(EV / f"diag_probe_s{s}.json") for s in (0, 1, 2)}


def pick(arm, s, key):
    hits = find_all(diag[s]["arms"][arm], key)
    return hits[0][1] if hits else None


mech = {a: {s: {k: pick(a, s, k) for k in
                 ("corr_mask_amp", "mask_at_peak", "grad_top1pct_share", "field_max_mm",
                  "scale_cv", "scale_truth_corr", "scale_is_constant", "mean_out_equals_T")}
             for s in (0, 1, 2)} for a in SIX}
crit = {}
for arm in ("A1", "A2"):
    ok = [s for s in (0, 1, 2)
          if mech[arm][s]["corr_mask_amp"] is not None and mech[arm][s]["corr_mask_amp"] >= 0
          and mech[arm][s]["mask_at_peak"] is not None and mech[arm][s]["mask_at_peak"] > 0.5]
    crit[arm] = {"rule": "corr(mask,amp)>=0 且 mask_at_peak>0.5", "pass_seeds": ok, "n": len(ok)}
for arm in ("A3a", "A3b"):
    ok = [s for s in (0, 1, 2)
          if mech[arm][s]["grad_top1pct_share"] is not None and mech[arm][s]["grad_top1pct_share"] < 0.5
          and mech[arm][s]["field_max_mm"] is not None and mech[arm][s]["field_max_mm"] <= 30]
    crit[arm] = {"rule": "grad_top1pct_share<0.5 且 field_max_mm<=30", "pass_seeds": ok, "n": len(ok)}
for arm in ("A4", "A4b"):
    ok = [s for s in (0, 1, 2)
          if mech[arm][s]["scale_cv"] is not None and mech[arm][s]["scale_cv"] >= 0.05
          and mech[arm][s]["scale_truth_corr"] is not None and mech[arm][s]["scale_truth_corr"] >= 0.3]
    crit[arm] = {"rule": "scale_cv>=0.05 且 scale_truth_corr>=0.3", "pass_seeds": ok, "n": len(ok)}
for arm, c in crit.items():
    print("   ", arm, c)
chk("D1 A1/A2 门指向：3/3 过（门指向已结构上解决）", crit["A1"]["n"] == 3 and crit["A2"]["n"] == 3,
    f"A1={crit['A1']['n']}/3 A2={crit['A2']['n']}/3")
chk("D2 A3a/A3b 拓扑：0/3 过（梯度集中未解决、场最大值仍撞 clamp）",
    crit["A3a"]["n"] == 0 and crit["A3b"]["n"] == 0, f"A3a={crit['A3a']['n']}/3 A3b={crit['A3b']['n']}/3")
chk("D3 A4 量通道：3/3 过；A4b：2/3 过（触发的是相关性分支，非 cv 分支）",
    crit["A4"]["n"] == 3 and crit["A4b"]["n"] == 2, f"A4={crit['A4']['n']}/3 A4b={crit['A4b']['n']}/3")
notT = [f"{a}s{s}" for a in ("A4", "A4b") for s in (0, 1, 2) if mech[a][s]["mean_out_equals_T"] is False]
chk("D4 mean(out)=T 不达格仍记为实现缺陷（非机制结论）", len(notT) > 0, f"不达格: {notT}")
cvs = [mech[a][s]["scale_cv"] for a in ("A4", "A4b") for s in (0, 1, 2) if mech[a][s]["scale_cv"] is not None]
chk("D5 量通道未退化成常数（cv 分支从未触发，min cv 远高于 0.05 底线）", min(cvs) > 0.05,
    f"六格 cv min={min(cvs):.4f} max={max(cvs):.4f}（底线 0.05）")
fm = [mech[a][s]["field_max_mm"] for a in ("A3a", "A3b") for s in (0, 1, 2) if mech[a][s]["field_max_mm"] is not None]
chk("D6 拓扑两臂场最大值六格全 = 402.43 mm（模型 clamp 上限，截断未作用在输出场上）",
    len(fm) == 6 and all(abs(v - 402.4288) < 0.01 for v in fm),
    f"六格 field_max_mm = {[round(v, 4) for v in fm]}")

# ---------------------------------------------------------------- E. 裁决
print("=== E. 裁决 ===")
chk("E1 六臂全部判负、无一臂进十 seed、不投十 seed 算力",
    all_cross and qualifying == [],
    "S3 12/12 区间数值上全部跨零 -> 合格臂集合为空；3 seed 全过地板者（A3a/A3b）仍不晋级")

report = {
    "round": 90,
    "date": "2026-11-16",
    "remote": {
        "submitter_273": "COMPLETED 05:04:04",
        "stage2": "274/282/290 COMPLETED",
        "train_arrays": "275/283 7/7 COMPLETED",
        "summary_291": "COMPLETED（S3 当年自动跑完，未手动 sbatch）",
        "diag": "297/298 COMPLETED", "epochaxis": "315/316/317 COMPLETED",
        "eval_files": 21, "run_json": 13,
        "runs_e45_files_total": 244, "modified_after_close": 0,
        "e45_jobs_since_2026-11-12": 0, "squeue_e45": 0,
        "remote_actions": "零（只读查询：squeue/sacct/md5sum/find/ls）",
    },
    "new_audit_surface": {
        "s3_numeric_recompute": "不看 crosses_zero 布尔，改由 ci_lo<=0<=ci_hi 数值级重算；12/12 与存储标志自洽",
        "headline_path": "头条表改从主跑 *.eval.json 逐格复算（不走 scores_all.json），并与 scores_all 逐格自校验",
        "s3_verdict_branch": "s3_verdict 两分支（GAIN_WAS_AREA_OR_AMPLITUDE / NO_GAIN_TO_BEGIN_WITH）与 raw 点估符号自洽",
        "s3_params": "thr=2.0 / control=B0 / n_boot=10000 / rng=20261109 / epoch=00 / n_windows=538",
    },
    "recomputed_table": recalc,
    "s3_cells": s3_cells,
    "s3_qualifying_arms": qualifying,
    "s3_failure_modes": branch_counts,
    "mechanism_criteria": crit,
    "mean_out_not_T_cells": notT,
    "checks": {"POS": len(POS), "NEG": len(NEG), "positives": POS, "negatives": NEG},
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nPOS {len(POS)}/{len(POS)+len(NEG)}   NEG {len(NEG)}/{len(POS)+len(NEG)}")
print("VERDICT:", "PASS" if not NEG else "FAIL")
print("report ->", OUT)
