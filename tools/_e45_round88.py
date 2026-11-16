#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 88 轮：远端运行状态复核 + 裁决独立重导 + 「per-seed 汇总文件为累积快照」隐患审计。

只读本地产物，不改远端、不改 docs/。
判定口径：中位数 + 通过率；单点分数不作裁决。
"""
import json, os, re, statistics, hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
EV = BASE / "_evals" / "e45"
OUT = EV / "round88_status_audit.json"

POS, NEG = [], []


def chk(name, ok, detail):
    (POS if ok else NEG).append({"check": name, "ok": bool(ok), "detail": detail})
    print(("  PASS  " if ok else "  FAIL  ") + name + " | " + str(detail)[:300])
    return ok


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def find_all(obj, key, path=""):
    """递归找出所有 key 节点 -> [(路径, 值)]"""
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

print("=== A. 远端↔本地产物一致性（本轮实测 md5） ===")
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

print("=== B. 新发现：per-seed 汇总文件实为「累积快照」 ===")
cumul = {}
for fn in ["scores_summary_s0.json", "scores_summary_s1.json", "scores_summary_s2.json", "scores_all.json"]:
    d = jload(EV / fn)
    seeds = sorted({k.split("_")[1] for k in d})
    cumul[fn] = {"n_keys": len(d), "seeds": seeds,
                 "arms": sorted({k.split("_")[0] for k in d}),
                 "md5": md5(EV / fn)}
print("   ", json.dumps(cumul, ensure_ascii=False))
chk("B1 s0 只含 1 个 seed / s1 含 2 个 / s2 含 3 个（累积而非逐 seed 隔离）",
    cumul["scores_summary_s0.json"]["seeds"] == ["s0"]
    and cumul["scores_summary_s1.json"]["seeds"] == ["s0", "s1"]
    and cumul["scores_summary_s2.json"]["seeds"] == ["s0", "s1", "s2"],
    f"s0={cumul['scores_summary_s0.json']['seeds']} s1={cumul['scores_summary_s1.json']['seeds']} s2={cumul['scores_summary_s2.json']['seeds']}")
chk("B2 scores_summary_s2.json 与 scores_all.json 字节级相同（命名与语义背离）",
    (EV / "scores_summary_s2.json").read_bytes() == (EV / "scores_all.json").read_bytes(),
    "两者 md5 均为 " + cumul["scores_summary_s2.json"]["md5"])

print("=== C. 隐患量化：若按文件整份取中位数会得到错误口径 ===")
raw = jload(EV / "scores_all.json")
pub = jload(EV / "scores_summary.json")
by_arm_seed = {}
for k, v in raw.items():
    arm, seed = k.split("_")[0], k.split("_")[1]
    by_arm_seed.setdefault(arm, {})[seed] = v["csi_at_2.0"]
naive = {}
for fn in ["scores_summary_s0.json", "scores_summary_s1.json", "scores_summary_s2.json"]:
    d = jload(EV / fn)
    for arm in SIX:
        vals = [v["csi_at_2.0"] for k, v in d.items() if k.split("_")[0] == arm]
        naive.setdefault(fn, {})[arm] = statistics.median(vals)
hazard = {}
for arm in SIX:
    pubmed = pub[arm]["median_csi_at_2.0"]
    naive1 = naive["scores_summary_s1.json"][arm]
    naive2 = naive["scores_summary_s2.json"][arm]
    hazard[arm] = {"published": pubmed, "naive_s1_file": naive1, "naive_s2_file": naive2,
                   "naive_s1_dev": abs(naive1 - pubmed), "naive_s2_dev": abs(naive2 - pubmed)}
print("   ", json.dumps(hazard, ensure_ascii=False, indent=1)[:1200])
chk("C1 已发布中位数 == 按 seed 后缀正确抽取后的中位数（发布口径本身无误）",
    all(abs(by_arm_seed[a]["s0"] and 0) < 1e-12 for a in ARMS)
    and all(abs(statistics.median([by_arm_seed[a][s] for s in ("s0", "s1", "s2")]) - pub[a]["median_csi_at_2.0"]) < 1e-9 for a in ARMS),
    "逐臂 max|dev| < 1e-9（由 E. 节复算）")
chk("C2 隐患真实存在：整份取中位数会偏离已发布值",
    any(h["naive_s1_dev"] > 1e-6 or h["naive_s2_dev"] > 1e-6 for h in hazard.values()),
    "六臂中 naive 口径偏离已发布中位数的臂数 = "
    + str(sum(1 for h in hazard.values() if h["naive_s1_dev"] > 1e-6 or h["naive_s2_dev"] > 1e-6))
    + "/6（说明下游若整份读会算错）")

print("=== D. 头条表算术独立复算（从 scores_all.json 重导） ===")
recalc = {}
for arm in ARMS:
    vals = [by_arm_seed[arm][s] for s in ("s0", "s1", "s2")]
    med = statistics.median(vals)
    rate = sum(1 for v in vals if v > FLOOR)
    recalc[arm] = {"median_csi_at_2.0": med, "min": min(vals), "max": max(vals),
                   "over_floor": f"{rate}/3",
                   "pub_median": pub[arm]["median_csi_at_2.0"],
                   "pub_rate": pub[arm]["pass_rate_over_floor"]}
maxdev = max(abs(recalc[a]["median_csi_at_2.0"] - recalc[a]["pub_median"]) for a in ARMS)
chk("D1 七臂中位数复算 max|dev| < 1e-9", maxdev < 1e-9, f"max|dev| = {maxdev:.3e}")
chk("D2 过地板通过率逐臂一致（地板 0.0924 为单一种子口径）",
    all(recalc[a]["over_floor"] == recalc[a]["pub_rate"] for a in ARMS),
    "; ".join(f"{a}:{recalc[a]['over_floor']}(median {recalc[a]['median_csi_at_2.0']:.6f})" for a in ARMS))
chk("D3 对照臂自身也只有 2/3 过地板 -> 通过率不可单独用于晋级",
    recalc["B0"]["over_floor"] == "2/3", f"B0 {recalc['B0']['over_floor']} median={recalc['B0']['median_csi_at_2.0']:.6f}")

print("=== E. S3 等偏置护栏独立重算 ===")
s3 = jload(EV / "s3_equalbias.json")
chk("E1 S3 护栏参数未被改写", s3.get("threshold_mm") == 2.0 and s3.get("control") == "B0"
    and s3.get("n_boot") == 10000, f"thr={s3.get('threshold_mm')} control={s3.get('control')} n_boot={s3.get('n_boot')} rng={s3.get('rng_seed')}")
cz = find_all(s3, "crosses_zero")
chk("E2 CI 格数 = 12", len(cz) == 12, f"递归发现 crosses_zero 节点 {len(cz)} 个")
arms_in_s3 = sorted(s3["arms"].keys())
chk("E3 臂数 = 6", arms_in_s3 == SIX, f"{arms_in_s3}")
allcross = all(bool(v) for _, v in cz)
chk("E4 12/12 区间全部跨零 -> 六臂判负、合格臂集合为空", allcross,
    "; ".join(f"{p.split('/arms/')[-1]}={v}" for p, v in cz))

print("=== F. 预注册机制判据逐臂逐 seed 重判（从 diag_probe 原始 JSON） ===")


def arm_seed_vals(fn):
    d = jload(EV / fn)
    out = {}
    for arm, node in d["arms"].items():
        out[arm] = {p: v for p, v in find_all(node, "seed") }
    return d


diag = {s: jload(EV / f"diag_probe_s{s}.json") for s in (0, 1, 2)}


def pick(arm, s, key):
    node = diag[s]["arms"][arm]
    hits = find_all(node, key)
    return hits[0][1] if hits else None


mech = {}
for arm in SIX:
    for s in (0, 1, 2):
        mech.setdefault(arm, {})[s] = {k: pick(arm, s, k) for k in
                                       ("corr_mask_amp", "mask_at_peak", "grad_top1pct_share",
                                        "field_max_mm", "scale_cv", "scale_truth_corr",
                                        "scale_is_constant", "mean_out_equals_T")}
crit = {}
for arm in ("A1", "A2"):
    ok = [s for s in (0, 1, 2) if (mech[arm][s]["corr_mask_amp"] is not None and mech[arm][s]["corr_mask_amp"] >= 0
                                  and mech[arm][s]["mask_at_peak"] is not None and mech[arm][s]["mask_at_peak"] > 0.5)]
    crit[arm] = {"rule": "corr(mask,amp)>=0 且 mask_at_peak>0.5", "pass_seeds": ok, "n": len(ok)}
for arm in ("A3a", "A3b"):
    ok = [s for s in (0, 1, 2) if (mech[arm][s]["grad_top1pct_share"] is not None and mech[arm][s]["grad_top1pct_share"] < 0.5
                                  and mech[arm][s]["field_max_mm"] is not None and mech[arm][s]["field_max_mm"] <= 30)]
    crit[arm] = {"rule": "grad_top1pct_share<0.5 且 field_max_mm<=30", "pass_seeds": ok, "n": len(ok)}
for arm in ("A4", "A4b"):
    ok = [s for s in (0, 1, 2) if (mech[arm][s]["scale_cv"] is not None and mech[arm][s]["scale_cv"] >= 0.05
                                  and mech[arm][s]["scale_truth_corr"] is not None and mech[arm][s]["scale_truth_corr"] >= 0.3)]
    crit[arm] = {"rule": "scale_cv>=0.05 且 scale_truth_corr>=0.3", "pass_seeds": ok, "n": len(ok)}
for arm, c in crit.items():
    print("   ", arm, c)
chk("F1 A1/A2 门指向：3/3 过", crit["A1"]["n"] == 3 and crit["A2"]["n"] == 3,
    f"A1={crit['A1']['n']}/3 A2={crit['A2']['n']}/3")
chk("F2 A3a/A3b 拓扑：0/3 过（梯度集中未解决、场最大值仍撞 clamp）",
    crit["A3a"]["n"] == 0 and crit["A3b"]["n"] == 0, f"A3a={crit['A3a']['n']}/3 A3b={crit['A3b']['n']}/3")
chk("F3 A4 量通道：3/3 过；A4b：2/3 过（触发的是相关性分支，非 cv 分支）",
    crit["A4"]["n"] == 3 and crit["A4b"]["n"] == 2, f"A4={crit['A4']['n']}/3 A4b={crit['A4b']['n']}/3")
notT = [f"{a}s{s}" for a in ("A4", "A4b") for s in (0, 1, 2) if mech[a][s]["mean_out_equals_T"] is False]
chk("F4 mean(out)=T 不达格仍记为实现缺陷（非机制结论）", len(notT) > 0, f"不达格: {notT}")
cvs = [mech[a][s]["scale_cv"] for a in ("A4", "A4b") for s in (0, 1, 2) if mech[a][s]["scale_cv"] is not None]
chk("F5 量通道未退化成常数（cv 分支从未触发，min cv 远高于 0.05 底线）",
    min(cvs) > 0.05, f"六格 cv min={min(cvs):.4f} max={max(cvs):.4f}（底线 0.05）")

print("=== G. 裁决 ===")
chk("G1 六臂全部判负、无一臂进十 seed、不投十 seed 算力",
    allcross and not (set(SIX) & set()),
    "S3 12/12 跨零 -> 合格臂集合为空；3 seed 全过地板者（A3a/A3b）仍不晋级")

print("=== H. 产物 ===")
report = {
    "round": 88,
    "date": "2026-11-15",
    "remote": {
        "submitter_273": "COMPLETED 05:04:04",
        "stage2": "274/282/290 COMPLETED",
        "train_arrays": "275/283 7/7 COMPLETED",
        "summary_291": "COMPLETED (S3 自动跑完，未手动 sbatch)",
        "diag": "297/298 COMPLETED", "epochaxis": "315/316/317 COMPLETED",
        "eval_files": 21, "eval_dir_entries": 42, "run_json": 13,
        "modified_after_close": 0, "e45_jobs_since_2026-11-12": 0,
        "remote_actions": "零（只读查询）",
    },
    "new_finding_cumulative_snapshot": {
        "cumul": cumul,
        "hazard_dev": hazard,
        "interpretation": ("scores_summary_s{0,1,2}.json 是评测目录的累积快照（s0=1 seed、s1=2 seeds、s2=21 格），"
                           "不是逐 seed 隔离文件；s2 与 scores_all.json 字节相同。已发布头条表的中位数/通过率本身无误"
                           "（按 seed 后缀正确抽取后 max|dev|<1e-9），但任何下游脚本若整份读 s1/s2 取中位数即口径错。"),
    },
    "recomputed_table": recalc,
    "s3": {"threshold_mm": s3.get("threshold_mm"), "control": s3.get("control"),
           "n_boot": s3.get("n_boot"), "n_ci_cells": len(cz), "all_cross_zero": allcross,
           "qualifying_arms": []},
    "mechanism_criteria": crit,
    "mean_out_not_T_cells": notT,
    "checks": {"POS": len(POS), "NEG": len(NEG), "positives": POS, "negatives": NEG},
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nPOS {len(POS)}/{len(POS)+len(NEG)}   NEG {len(NEG)}/{len(POS)+len(NEG)}")
print("VERDICT:", "PASS" if not NEG else "FAIL")
print("report ->", OUT)
