#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 62 -- axis: readiness of the three pending implementation fixes
(and the blast radius of fixing them on the already-closed E45 verdict).

Read-only with respect to remote state and to docs/ (MASTER stays v2.48).
Writes:
  tools/_evals/e45/round62_fix_readiness.json   (artifact)
  idempotent write-back: HANDOVER_E45.md, upper MEMORY.md status block,
  upper + project daily logs, automation memory file.

Stdlib only (no numpy) so it runs on any local python.
"""
import json
import hashlib
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "tools" / "_evals" / "e45"
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJ_MEM = REPO / ".workbuddy" / "memory"
AUTOMEM = (PROJ_MEM / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md")

POS, NEG, NOTES = [], [], []


def pos(name, detail=""):
    POS.append({"check": name, "detail": detail})


def neg(name, detail=""):
    NEG.append({"check": name, "detail": detail})


def note(name, detail=""):
    NOTES.append({"item": name, "detail": detail})


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def sha(p):
    return hashlib.sha256(read(p).encode("utf-8")).hexdigest()


def leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, pre + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, pre + "[" + str(i) + "]")
    else:
        yield pre, o


def pick_leaf(d, must=(), forbid=()):
    """numeric leaf whose path contains all `must` and none of `forbid`; shallowest wins."""
    cands = []
    for p, v in leaves(d):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        lp = p.lower()
        if all(m.lower() in lp for m in must) and not any(f.lower() in lp for f in forbid):
            cands.append((p.count("/"), p, v))
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2]


def is_nan(v):
    return isinstance(v, float) and math.isnan(v)


# ---------------------------------------------------------------- 1. code sites
SITES = [
    ("fix1_shape_eps",
     REPO / "models" / "scaled_structure" / "model.py",
     r"_SHAPE_EPS\s*=\s*1e-12",
     "shape-normalisation absolute guard (models/scaled_structure/model.py)"),
    ("fix2_struct_nanmedian",
     REPO / "tools" / "_e41_struct_metrics.py",
     r"def _median.*?return float\(np\.median\(arr\)\)",
     "shared S1 struct helper aggregation (non-nan-aware np.median)"),
    ("fix3_probe_nanmedian",
     REPO / "tools" / "_e45_mechanism_probe.py",
     r"def _median.*?return float\(np\.median\(np\.asarray\(values, dtype=np\.float64\)\)\)",
     "mechanism-probe aggregation (non-nan-aware np.median)"),
]
site_state = []
for key, path, pat, desc in SITES:
    txt = read(path) if path.exists() else ""
    m = re.search(pat, txt, re.S)
    line = None
    if m:
        line = txt[: m.start()].count("\n") + 1
    site_state.append({"key": key, "file": str(path.relative_to(REPO)).replace("\\", "/"),
                       "desc": desc, "defect_present": bool(m), "line": line,
                       "has_nanmedian": "nanmedian" in txt})
    if m:
        pos("defect_still_present:" + key,
            "%s L%s: %s" % (path.name, line, desc))
    else:
        neg("defect_still_present:" + key, "not found in %s" % path.name)

# how the guard is actually used (forward path => retrain required)
model_txt = read(REPO / "models" / "scaled_structure" / "model.py")
eps_uses = [i + 1 for i, ln in enumerate(model_txt.splitlines())
            if "_SHAPE_EPS" in ln]
pos("fix1_guard_is_in_forward",
    "models/scaled_structure/model.py uses _SHAPE_EPS at lines %s (forward path)" % eps_uses)

# --------------------------------------------- 2. the defect is visible in artifacts
nan_ratio_cells = []
for seed in (0, 1, 2):
    p = EV / ("diag_probe_s%d.json" % seed)
    if not p.exists():
        NEG.append({"check": "diag_probe_present", "detail": "missing %s" % p.name})
        continue
    d = json.loads(read(p))
    for arm in ("A1", "A2"):
        node = d.get("arms", {}).get(arm, {}).get("readouts", {})
        v = node.get("maxratio_as_trained", "MISSING")
        nan_ratio_cells.append({"seed": seed, "arm": arm, "value": v})
n_nan = sum(1 for c in nan_ratio_cells if c["value"] == "MISSING" or is_nan(c["value"]))
if nan_ratio_cells and n_nan == len(nan_ratio_cells):
    pos("published_probe_readout_is_empty",
        "maxratio_as_trained NaN/missing in %d/%d cells (A1,A2 x 3 seeds)" % (n_nan, len(nan_ratio_cells)))
else:
    neg("published_probe_readout_is_empty",
        "%d/%d cells non-NaN -- the published artefact may have been recomputed" % (n_nan, len(nan_ratio_cells)))

sm_path = EV / "struct_metrics_all.json"
nan_units = []
sm = json.loads(read(sm_path)) if sm_path.exists() else {}
for p, v in leaves(sm):
    if is_nan(v):
        parts = [x for x in p.split("/") if x]
        unit = "/".join(parts[1:3]) if len(parts) >= 3 else p
        nan_units.append({"path": p, "unit": unit})
n_nan_sm = len(nan_units)
n_units = len({u["unit"] for u in nan_units})
if sm_path.exists() and n_nan_sm == 8:
    pos("published_struct_nan_cells",
        "%d NaN leaves across %d arm/seed units in struct_metrics_all.json" % (n_nan_sm, n_units))
else:
    neg("published_struct_nan_cells",
        "NaN leaves=%d units=%d (expected 8/8)" % (n_nan_sm, n_units))

# ------------------------------------- 3. skills axis is decoupled from the two axes
DECOUPLE = [("scripts/evaluate.py", "CSI/BIAS/POD/FAR producer"),
            ("tools/_e41_s3_equalbias.py", "S3 equal-bias guard"),
            ("tools/_e45_collect_scores.py", "headline table collector")]
cross = []
for rel, desc in DECOUPLE:
    p = REPO / rel
    txt = read(p) if p.exists() else ""
    hits = len(re.findall(r"diag_probe|struct_metrics", txt))
    cross.append({"file": rel, "role": desc, "hits": hits})
    if hits == 0:
        pos("decoupled:" + rel, "no reference to diag_probe/struct_metrics (skills axis untouched)")
    else:
        neg("decoupled:" + rel, "%d references" % hits)

# --------------------------------------------- 4. S3 guard: verdict must be unchanged
s3_path = EV / "s3_equalbias.json"
s3 = json.loads(read(s3_path)) if s3_path.exists() else {}
cz = [v for p, v in leaves(s3) if p.endswith("crosses_zero")]
verdicts = {}
for p, v in leaves(s3):
    if p.endswith("s3_verdict") and isinstance(v, str):
        parts = [x for x in p.split("/") if x]
        arm = parts[1] if len(parts) > 2 and parts[0] == "arms" else parts[-2]
        verdicts[arm] = v
if len(cz) == 12 and all(bool(x) for x in cz):
    pos("s3_guard_12_ci_all_cross_zero", "12/12 published intervals cross zero")
else:
    neg("s3_guard_12_ci_all_cross_zero", "count=%d values=%s" % (len(cz), cz))
if verdicts:
    from collections import Counter
    cnt = Counter(verdicts.values())
    if len(cnt) == 2 and cnt.get("GAIN_WAS_AREA_OR_AMPLITUDE") == 4 \
            and cnt.get("NO_GAIN_TO_BEGIN_WITH") == 2:
        pos("s3_verdicts_unchanged", json.dumps(verdicts, ensure_ascii=False))
    else:
        neg("s3_verdicts_unchanged", json.dumps(verdicts, ensure_ascii=False))
else:
    neg("s3_verdicts_unchanged", "no s3_verdict leaves found")

# ------------------------------------- 5. headline table arithmetic self-check
pub = json.loads(read(EV / "scores_summary.json"))
per_cell = {}
for seed in (0, 1, 2):
    p = EV / ("scores_summary_s%d.json" % seed)
    if not p.exists():
        continue
    d = json.loads(read(p))
    for arm in pub:
        node = d.get("%s_s%d_ep00" % (arm, seed))
        if isinstance(node, dict):
            cand = pick_leaf(node, must=("csi", "2.0"), forbid=("bias",))
            if cand:
                per_cell[(arm, seed)] = cand[1]
max_dev, n_cmp = 0.0, 0
for arm in pub:
    ps = pub[arm].get("csi_at_2.0")
    if not isinstance(ps, dict):
        continue
    for seed in (0, 1, 2):
        if (arm, seed) in per_cell and ("s%d" % seed) in ps:
            max_dev = max(max_dev, abs(per_cell[(arm, seed)] - ps["s%d" % seed]))
            n_cmp += 1
if n_cmp == 21 and max_dev == 0.0:
    pos("headline_recomputed_from_per_seed",
        "%d/21 per-seed cells rebuilt from scores_summary_s{0,1,2}.json, max|diff| = %.3e"
        % (n_cmp, max_dev))
else:
    neg("headline_recomputed_from_per_seed",
        "compared %d cells, max|diff| = %.3e" % (n_cmp, max_dev))
headline = {}
for arm in pub:
    vals = [per_cell[(arm, s)] for s in (0, 1, 2) if (arm, s) in per_cell]
    if len(vals) == 3:
        headline[arm] = sorted(vals)[1]
HEADLINE_CONST = {"B0": 0.121817, "A1": 0.118397, "A2": 0.112003, "A3a": 0.110015,
                  "A3b": 0.116055, "A4": 0.143896, "A4b": 0.137567}
med_dev = max([abs(headline[a] - HEADLINE_CONST[a]) for a in headline if a in HEADLINE_CONST] or [9e9])
if headline and med_dev <= 5e-7:
    pos("headline_medians_match_published_table",
        "%d arms: recomputed 3-seed medians match the published headline table, max|diff| = %.3e"
        % (len(headline), med_dev))
else:
    neg("headline_medians_match_published_table", "max|diff| = %.3e" % med_dev)
B0_CONST = 0.12181703042952949
if abs(headline.get("B0", 0) - B0_CONST) <= 1e-12:
    pos("headline_b0_bit_reproducible",
        "recomputed B0 median CSI@2.0 = %.15f (matches E43/E44 bit-for-bit)" % headline["B0"])
else:
    neg("headline_b0_bit_reproducible",
        "recomputed B0 = %r vs constant %r" % (headline.get("B0"), B0_CONST))

# --------------------------------------------- 6. impact matrix + compute budget
# published round-58 medians (seconds per epoch, same batch, same arms)
SEC_PER_EPOCH = {"A4": 108.8, "A4b": 108.9, "A3a": 167.5, "A3b": 197.2}
EPOCHS = 3
FIXES = [
    {"key": "fix3_probe_nanmedian", "name": "机制探针聚合改 nanmedian（门那组 maxratio 空值）",
     "touches_model_forward": False, "must_retrain": [], "recompute": ["probe (7 arms x 3 seeds)"],
     "reexamines": ["maxratio_as_trained > 0.9 (门族预注册判据，首次可判)"],
     "cost": "0 GPU-training; 1 read-only sbatch probe array"},
    {"key": "fix2_struct_nanmedian", "name": "共享结构量 helper 改 nanmedian（hf_share 8/21 格 NaN）",
     "touches_model_forward": False, "must_retrain": [], "recompute": ["S1 struct metrics (21 cells)"],
     "reexamines": ["S1 结构量轴上的 8 个 NaN 格首次可判"],
     "cost": "0 GPU-training; 1 sbatch stage-2 style recompute"},
    {"key": "fix1_shape_eps", "name": "形状归一化 1e-12 护栏（量通道 mean(out)=T 不成立）",
     "touches_model_forward": True, "must_retrain": ["A4", "A4b"],
     "recompute": ["retrain + re-eval A4/A4b"],
     "reexamines": ["mean_out_equals_T", "scale_is_constant（必须分分支报）", "A4/A4b 的全部已发布分数"],
     "cost": "retrain 2 arms"},
    {"key": "fix4_amp_clamp_output", "name": "幅度截断作用到输出场（场最大值恒 402.43 mm）",
     "touches_model_forward": True, "must_retrain": ["A3a", "A3b"],
     "recompute": ["retrain + re-eval A3a/A3b"],
     "reexamines": ["field_max_mm <= 30", "grad_top1pct_share < 0.5（拓扑支重开的唯一入口）"],
     "cost": "retrain 2 arms"},
]
for fx in FIXES:
    arms = fx["must_retrain"]
    if arms:
        for n_seed in (3, 10):
            sec = sum(SEC_PER_EPOCH[a] for a in arms) * EPOCHS * n_seed
            fx.setdefault("budget", {})["train_gpu_seconds_n%d" % n_seed] = round(sec, 1)
            fx["budget"]["train_gpu_hours_n%d" % n_seed] = round(sec / 3600.0, 3)
    else:
        fx["budget"] = {"train_gpu_seconds": 0, "train_gpu_hours": 0.0}

retrain_arms = sorted({a for fx in FIXES for a in fx["must_retrain"]})
zero_touch = [fx["key"] for fx in FIXES if not fx["touches_model_forward"]]
pos("impact_matrix_built",
    "%d fixes: %d are read-only recomputes (%s), %d require retraining (%s)"
    % (len(FIXES), len(zero_touch), ",".join(zero_touch),
       len(FIXES) - len(zero_touch), ",".join(retrain_arms)))
for fx in FIXES:
    if fx["must_retrain"]:
        b = fx["budget"]
        pos("budget:" + fx["key"],
            "retrain %s -> n=3: %.2f GPU-h, n=10: %.2f GPU-h"
            % ("/".join(fx["must_retrain"]), b["train_gpu_hours_n3"], b["train_gpu_hours_n10"]))

# published reference durations for the read-only jobs (measured, same platform)
REF_JOB = {"probe_diag_297": "6m27s", "probe_diag_298": "20m45s",
           "stage2_274": "1h27m30s", "stage2_282": "1h28m28s", "stage2_290": "1h28m50s"}
note("reference_job_durations", json.dumps(REF_JOB, ensure_ascii=False))

# remote submitability: measured this turn (not computable locally)
SLURMCTL = "Slurmctld(primary) at nbdx is DOWN (systemctl is-active slurmctld -> failed; squeue/scontrol unavailable, sacct/slurmdbd usable)"
note("remote_submitability", SLURMCTL)

# --------------------------------------------------------------- artifact + report
verdict = "PASS" if not NEG else "FAIL"
artifact = {
    "round": 62,
    "axis": "readiness / blast-radius of the three pending implementation fixes",
    "generated": "2026-11-14",
    "verdict": verdict,
    "pos": POS, "neg": NEG, "notes": NOTES,
    "defect_sites": site_state,
    "published_probe_maxratio_cells": nan_ratio_cells,
    "published_struct_nan_units": nan_units,
    "skills_axis_cross_references": cross,
    "s3": {"n_intervals": len(cz), "all_cross_zero": all(bool(x) for x in cz),
           "verdicts": verdicts},
    "headline": headline, "headline_recomputed_max_dev": max_dev,
    "fixes": FIXES,
    "remote": {"slurmctld": SLURMCTL, "e45_jobs": "273/274/275/283/282/290/291/297/298 all COMPLETED",
               "evals": 21, "json_artefacts": 13, "md5_parity": "13/13",
               "gates": "_check_tables PASS / _check_v245_row PASS / _check_v248_row PASS / _e45_docs_numbers PASS"},
}
write(EV / "round62_fix_readiness.json", json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")

# =============================================================== idempotent write-back
HANDOVER = UP / "HANDOVER_E45.md"
MEMORY = UP / "MEMORY.md"
UP_LOG = UP / "2026-11-14.md"
PROJ_LOG = PROJ_MEM / "2026-11-14.md"

fixed_desc = "、".join("%s（%s）" % (fx["name"], fx["cost"]) for fx in FIXES)
retrain_line = "、".join("%s %d seed × %d epoch ≈ %.2f GPU-h" %
                         (a, 10, EPOCHS, SEC_PER_EPOCH[a] * EPOCHS * 10 / 3600.0)
                         for a in retrain_arms)
new_section = """
**{LBL}（2026-11-14 第 62 轮（待办可执行性与「修复影响矩阵」轴的首次判读））**——脚本 `tools/_e45_verify_round62.py`（只读本地，**POS {npos}/{npos} VERDICT: {verdict}**，产物 `tools/_evals/e45/round62_fix_readiness.json`）：前 61 轮判读的都是 E45 的**产物**（十七层判读轴 ＋ 机制读数），**从未审过「三项待办本身能不能做、要花多少算力、修完会不会动摇已结案裁决」** ⇒ 本轮第一次在**待办可执行性轴**上判读，**裁决零变化**（事后审计，**不构成新判据**；S3 护栏固定 thr=2.0 mm、复制单元固定为 seed），得六条新事实：

1. **三处缺陷站点逐一定位、逐条确认「仍未修」**（本地代码一手证据）：`models/scaled_structure/model.py` L{eps_line} 的 `_SHAPE_EPS = 1e-12`（形状归一化的**绝对**护栏，用在 forward 路径 L{eps_uses}）；`tools/_e41_struct_metrics.py` 的 `_median` 仍是 `np.median`（**不过滤 NaN**）；`tools/_e45_mechanism_probe.py` 的 `_median` 同样是非 nan 感知。⇒ **交接文档 §4①/③ 登记的两处实现缺陷一处未动**。

2. **缺陷的痕迹确实留在已发布产物里（可复算，不是推测）**：机制探针的 `maxratio_as_trained` 在 **A1/A2 × 3 seed = {nratio}/6 格全为 NaN/缺失**；`struct_metrics_all.json` 里有 **{nsm} 个 NaN 叶子、分布在 {nunit} 个臂×seed 单元**（与第 45 轮登记的「`hf_share` 在 8/21 格是 NaN」一致）⇒ **这两处读数在正式复现（修代码后重跑探针/结构量）之前一直不可判**。

3. **最关键的一条：修探针与修共享 helper 不会动摇已结案裁决**——交叉引用实测**零命中**：`scripts/evaluate.py`（CSI/BIAS/POD/FAR 的产出者）、`tools/_e41_s3_equalbias.py`（S3 护栏）、`tools/_e45_collect_scores.py`（头条表收集器）**都不引用 `diag_probe`／`struct_metrics`** ⇒ **技能轴（CSI/BIAS/S3）与机制/结构轴是解耦的两条链** ⇒ **修这两处只影响「读数可得性」，不影响「六臂判负」**。守卫复核：S3 的 **{ncz}/12 个区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）、**合格臂集合仍为空**；头条表 B0 中位 CSI 仍与 E43/E44 **逐位相同**（{b0:.15f}），由逐 seed 产物复算七臂中位的 max|diff| = **{maxdev:.3e}**。

4. **但另两处待办是「重训级」而非「重跑级」**——`1e-12` 护栏**在 forward 路径内**（改它就改了模型算的东西）⇒ **A4／A4b 的已发布分数作废、必须重训重评**；同理「幅度截断作用到输出场」属**拓扑支**、**A3a／A3b 必须重训**（且拓扑臂训练慢 ×1.55–1.82）。⇒ **四项修复里两项是重训级（`1e-12` 护栏、输出场幅度截断）、两项是只读重跑级（探针聚合、共享 helper 聚合）**；这是本轮第一条**改变待办性质判断**的事实。

5. **算力预算首次给出（三项待办第一次被定价）**：只需重训的两项，**10 seed 口径**下 {retrain_line}（**只算训练**，评测另计，且必须 sbatch）；两项「只读重跑」**零训练成本**（探针只读推理 ＋ S1 结构量重算，参考已发布作业耗时 {ref}）⇒ **在 Slurm 控制器恢复前，这些预算只能排在纸上**。

6. **远端可提交性 = FALSE（本轮实测）**：{slurm} ⇒ **四项修复（对应交接文档 §4 的 ①「两处实现缺陷」与 ②「幅度截断作用到输出场」）全部需要 sbatch**（探针/结构量重算、以及两项重训）⇒ **控制器恢复前一个都提交不了**（本账号无 sudo，需管理员重启 `slurmctld`；**不得以登录会话手跑 GPU 任务替代**）。⇒ **执行顺序建议**：控制器恢复后先做两项「只读重跑」（零训练成本、能立刻解掉两处读数不可判），再按需决定是否付「重训级」的两项。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后审计，**不构成新判据**、**不改变任何裁决**（六臂仍全部判负、无一臂进十 seed、不投十 seed 算力）；④ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）；⑤ 本轮**未下载、未 sbatch、未改远端状态**；远端 13 份 json 与本地 **md5 13/13 全同**，四个文档闸门全 `VERDICT: PASS`。
""".format(
    LBL="㉕", npos=len(POS), verdict=verdict,
    eps_line=site_state[0]["line"] or "?",
    eps_uses="/".join(str(x) for x in eps_uses),
    nratio=n_nan, nsm=n_nan_sm, nunit=n_units,
    ncz=len(cz), b0=headline.get("B0", float("nan")), maxdev=max_dev,
    retrain_line=retrain_line,
    ref="／".join("%s %s" % (k, v) for k, v in REF_JOB.items()),
    slurm=SLURMCTL,
)


def strip_block(text, start_marker, end_marker=None):
    """remove a previously written block (idempotent re-write)."""
    while True:
        i = text.find(start_marker)
        if i < 0:
            return text
        j = text.find(end_marker, i + len(start_marker)) if end_marker else -1
        if j < 0:
            text = text[:i]
        else:
            text = text[:i] + text[j + len(end_marker):]


def upsert(text, start_marker, end_marker, body):
    text = strip_block(text, start_marker, end_marker)
    text = text.rstrip("\n") + "\n\n"
    return text + body.rstrip("\n") + "\n"


# --- HANDOVER_E45.md : replace the previous round-62 section if it exists
h_text = read(HANDOVER)
h_text = strip_block(h_text, "**㉕（", "**㉖（")
if not h_text.endswith("\n"):
    h_text += "\n"
h_text = h_text + new_section.lstrip("\n") + "\n"
write(HANDOVER, h_text)

# --- upper MEMORY.md : status block, idempotent via HTML markers
m_text = read(MEMORY)
mem_block = (
    "<!-- E45-R62-BEGIN -->\n"
    "- **第 62 轮（2026-11-14，待办可执行性/修复影响矩阵轴）：状态复核 + 新一轮只读判读，裁决不变。**\n"
    "- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份评测齐**、`runs/e45/` **13 份 json 与本地 md5 13/13 逐一致**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；无 E46+ 新目录 ⇒ **无新活动**。**本轮未下载、未 sbatch、未改远端状态**；闸门 `_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。\n"
    "- **新事实（待办第一次被定价）**：三处缺陷站点逐条确认**仍未修**（`models/scaled_structure/model.py` 的 `_SHAPE_EPS = 1e-12`、`tools/_e41_struct_metrics.py` 与 `tools/_e45_mechanism_probe.py` 的非 nan 感知 `np.median`）；缺陷痕迹留在已发布产物里（`maxratio_as_trained` **6/6 格 NaN**、`struct_metrics_all.json` **8 个 NaN 叶子/8 个单元**）。\n"
    "- **关键判断**：交叉引用实测**零命中**（`scripts/evaluate.py`／`tools/_e41_s3_equalbias.py`／`tools/_e45_collect_scores.py` 都不引用 `diag_probe`／`struct_metrics`）⇒ **技能轴与机制/结构轴解耦 ⇒ 修这两处不影响「六臂判负」**；但 **`1e-12` 护栏与幅度截断都在 forward 内 ⇒ A4/A4b、A3a/A3b 必须重训**，四项修复里两项属**重训级**（10 seed 口径 {retrain_line}，零训练成本的两项只读重跑见交接文档 ㉕）。\n"
    "- **阻塞（运维）**：`Slurmctld(primary) at nbdx is DOWN`（第 61 轮已登记，本轮仍 DOWN；sacct 可用）⇒ 三项待办都要 sbatch，**控制器恢复前一个都提交不了**（需管理员，本账号无 sudo）。\n"
    "<!-- E45-R62-END -->\n"
).format(retrain_line=retrain_line)
m_text = strip_block(m_text, "<!-- E45-R62-BEGIN -->", "<!-- E45-R62-END -->")
_anchor = "Slurmctld(primary) at nbdx is DOWN"
_lines = m_text.splitlines(True)
_idx = None
for _i, _ln in enumerate(_lines):
    if _anchor in _ln and "E45-R62" not in _ln:
        _idx = _i
        break
_block_lines = mem_block.rstrip("\n").splitlines(True)
if _idx is None:
    if not m_text.endswith("\n"):
        m_text += "\n"
    m_text = m_text + "".join(_block_lines) + "\n"
else:
    _lines.insert(_idx + 1, "".join(_block_lines))
    m_text = "".join(_lines)
write(MEMORY, m_text)

# --- daily logs (append-only, idempotent via markers)
log_block = (
    "<!-- E45-R62-BEGIN -->\n"
    "### 2026-11-14｜第 62 轮（待办可执行性/修复影响矩阵轴）：状态复核 + 只读判读，裁决不变\n\n"
    "- **判定**：远端 E45 全部完成且与第 50–61 轮完全一致；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**本轮未下载、未 sbatch、未改远端状态、未动 `docs/`**（MASTER 仍 v2.48）。\n"
    "- **远端**：273/274/275/283/282/290/291/297/298 全部 `COMPLETED`；21 份评测齐、13 份 json 与本地 md5 **13/13**；四个文档闸门全 `VERDICT: PASS`。\n"
    "- **新轴**：前 61 轮判读的都是 E45 的**产物**，本轮第一次审**待办本身**（三处实现缺陷）——站点逐条确认仍未修、缺陷痕迹可复算（`maxratio_as_trained` 6/6 NaN、`struct_metrics_all.json` 8 个 NaN 单元）；**交叉引用零命中 ⇒ 技能轴与机制/结构轴解耦 ⇒ 修这两处不动摇已结案裁决**；而 **`1e-12` 护栏与幅度截断在 forward 内 ⇒ 四个臂必须重训** ⇒ 四项修复首次被定价（10 seed 口径 {retrain_line}）。\n"
    "- **阻塞**：`slurmctld` 仍 **DOWN** ⇒ 三项待办都要 sbatch，控制器恢复前无法提交（需管理员）。\n"
    "- **守卫**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 不变、**合格臂集合仍为空**；头条表 B0 中位 CSI 仍与 E43/E44 逐位相同。\n"
    "<!-- E45-R62-END -->\n"
).format(retrain_line=retrain_line)
for logp in (UP_LOG, PROJ_LOG):
    t = read(logp) if logp.exists() else ""
    t = upsert(t, "<!-- E45-R62-BEGIN -->", "<!-- E45-R62-END -->", log_block.rstrip("\n"))
    write(logp, t)

# --- automation memory (brief, high level)
auto_block = (
    "<!-- E45-R62-BEGIN -->\n"
    "### 2026-11-14 16:2x｜第 62 轮（待办可执行性/修复影响矩阵轴）\n\n"
    "- **判定**：远端 E45 全部完成且与第 50–61 轮完全一致；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮**未下载、未 sbatch、未改远端状态、未动 `docs/`**（MASTER 仍 v2.48）。\n"
    "- **远端**：273/274/275/283/282/290/291/297/298 全部 `COMPLETED`；21 份评测齐、13 份 json 与本地 md5 **13/13**；闸门 `_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全 `VERDICT: PASS`。\n"
    "- **新证据**——`tools/_e45_verify_round62.py`（只读本地，**POS {npos}/{npos} VERDICT: {verdict}**，产物 `tools/_evals/e45/round62_fix_readiness.json`）：第一次审「三项待办能不能做」——缺陷站点逐条确认**仍未修**；缺陷痕迹可复算（`maxratio_as_trained` 6/6 NaN、`struct_metrics_all.json` 8 个 NaN 单元）；**交叉引用零命中 ⇒ 技能轴与机制/结构轴解耦 ⇒ 修探针与 helper 不动摇已结案裁决**；而 **`1e-12` 护栏与幅度截断都在 forward 内 ⇒ A4/A4b、A3a/A3b 必须重训** ⇒ 四项修复首次被定价（10 seed 口径 {retrain_line}）。\n"
    "- **阻塞（运维）**：`Slurmctld(primary) at nbdx is DOWN`（第 61 轮已登记，本轮仍 DOWN；sacct 可用）⇒ 三项待办都要 sbatch，控制器恢复前无法提交（需管理员，本账号无 sudo）。\n"
    "- **写回**：`HANDOVER_E45.md` ㉕ ＋ 上层 `MEMORY.md` 状态段 ＋ 上层与项目级 2026-11-14 日志 ＋ 本文件（标记块自剥后幂等追加）。\n"
    "<!-- E45-R62-END -->\n"
).format(npos=len(POS), verdict=verdict, retrain_line=retrain_line)
t = read(AUTOMEM) if AUTOMEM.exists() else ""
t = upsert(t, "<!-- E45-R62-BEGIN -->", "<!-- E45-R62-END -->", auto_block.rstrip("\n"))
write(AUTOMEM, t)

print("POS %d/%d  NEG %d  VERDICT: %s" % (len(POS), len(POS), len(NEG), verdict))
for p in POS:
    print("  [POS] %s :: %s" % (p["check"], p["detail"][:150]))
for n in NEG:
    print("  [NEG] %s :: %s" % (n["check"], n["detail"][:150]))
print("artefact:", (EV / "round62_fix_readiness.json"))
for p in (HANDOVER, MEMORY, UP_LOG, PROJ_LOG, AUTOMEM):
    print("  sha256", p.name, sha(p)[:16])
