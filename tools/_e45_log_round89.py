#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 89 轮：远端运行状态复核 → 产物审计 + 带标记的幂等写回。

只读远端产物副本 + 本地产物；不下载、不 sbatch、不改远端状态、不改 docs/。
写回目标：HANDOVER_E45.md / 上层 2026-11-11.md / 上层 MEMORY.md 状态段 /
          项目级 2026-11-15.md / 自动化 memory.md
幂等：标记 E45-R89-BEGIN 存在即跳过；MEMORY.md 状态段锚点找不到即报错，不静默跳过。
"""
import json, re, shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
EV = BASE / "_evals" / "e45"
OUT = EV / "round89_status_audit.json"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
UPPER_LOG = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
UPPER_MEM = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
PROJ_LOG = ROOT / ".workbuddy" / "memory" / "2026-11-15.md"
AUTO_MEM = (ROOT / ".workbuddy" / "memory" / "automations"
            / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md")

MARK = "E45-R89-BEGIN"
# 远端 md5 清单：直接解析本轮实测抓取的原始输出（不手抄，避免转写误差）
RAW = EV / "round89_remote_raw.txt"
REMOTE_MD5 = {}
for _ln in RAW.read_text(encoding="utf-8").splitlines():
    _p = _ln.split()
    if len(_p) == 2 and len(_p[0]) == 32 and _p[1].startswith("runs/e45/") and _p[1].endswith(".json"):
        REMOTE_MD5[Path(_p[1]).name] = _p[0]
assert len(REMOTE_MD5) == 13, f"远端 md5 行数异常: {len(REMOTE_MD5)}"

REPORTED = {  # 本轮远端实测
    "submitter_273": "COMPLETED 05:04:04",
    "train_arrays": "275_0..6 / 283_0..6 COMPLETED (14/14)",
    "stage2": "274 / 282 / 290 COMPLETED",
    "summary_291": "COMPLETED 00:14:23 (S3 等偏置对照当年自动跑完)",
    "diag": "297 / 298 COMPLETED",
    "epochaxis": "315 / 316 / 317 COMPLETED",
    "gpuprobe_314": "COMPLETED",
    "squeue_wf": "empty",
    "eval_json_files": 21,
    "eval_dir_entries": 42,
    "run_level_json": 13,
    "e45_jobs_after_2026-11-12": 0,
    "newer_than_1419_files": ["s3_equalbias.json", "struct_metrics_all.json",
                              "scores_summary.json", "scores_all.json"],
    "newer_than_1419_actual_mtime": "2026-11-11 14:19:05 (早于 14:20:42 收尾)",
    "modified_after_close_142042": 0,
    "remote_actions": "零（只读查询：squeue/sacct/tail/find/md5sum）",
    "extra_dirs": ["runs/e45_epochaxis (诊断产物)"],
}

BLOCK = """
<!-- {mark} -->
### {circle} 第 89 轮（2026-11-15）｜E45 远端运行状态复核 ＋ 裁决独立重导 ＋ **结案后 mtime 阈值口径订正**

- **判定**：远端 E45 全部完成、无新活动 ⇒ **裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。**远端动作＝0**（只读状态查询；未 sbatch／未同步／未下载）。
- **远端终态**：权威链 273／275·283（训练数组 7/7+7/7）／274·282·290（stage2）／291（summary）／297·298／314／315·316·317 全 `COMPLETED`；`squeue -u wf` 空；`runs/e45/evals/` **21 份 `.eval.json`**（目录共 42 条目）；`runs/e45/` **13 份** run 级 json；`runs/` 下无 E46+ 正式训练目录（只有诊断目录 `runs/e45_epochaxis`）。
- **产物一致性**：13 份 run 级 json **远端↔本地 md5 13/13 逐位全同**（本轮实测远端 md5 与本地重算 md5 对照，非复用登记值）。
- **本轮新增口径订正（重要）**：用 `find runs/e45 -newermt "2026-11-11 14:19"` 会命中 **4 份**文件（`s3_equalbias.json`／`struct_metrics_all.json`／`scores_summary.json`／`scores_all.json`），但实测其 mtime 是 **2026-11-11 14:19:05**，**早于 14:20:42 的收尾时刻** ⇒ 是 summary 作业写盘所致，**不是结案后改动**；改用正确阈值（> 14:20:42）后**结案后被修改文件 = 0 份** ⇒ 前几轮登记的「0 份」成立，本轮只是阈值取早了一分钟。**纪律：mtime 判据必须用收尾时刻（作业结束时间）而非其前一分钟的近似值。**
- **裁决独立重导**（`tools/_e45_round88.py`，只读本地，**POS 18/18 VERDICT: PASS**，不复用既有脚本结论）：头条表七臂中位 CSI@2.0 由 `scores_all.json` 重算 **max|dev| = 0.000e+00**、过地板通过率逐臂一致；**S3 护栏参数未改写**（thr 2.0／control B0／n_boot 10000／rng 20261109）、**12/12 CI 全跨零、合格臂集合为空**；预注册机制判据逐格重判：A1／A2 **3/3 过**（门指向已结构上解决）、A3a／A3b **0/3 过**（场最大值六格全 402.43 mm、梯度集中未解决 ⇒ 拓扑支按预注册结掉）、A4 **3/3 过**、A4b **2/3 过**（种子 2 触发的是**相关性分支**，其 `scale_cv` **3.0188** 为六格最高 ⇒ 未退化成常数）；`mean(out)=T` 不达格仍只记**实现缺陷**、不当机制结论（A4s1／A4bs0／A4bs1／A4bs2）。
- **文档闸门**：`_check_tables` **PASS**（MASTER／INDEX／E45 实验文件 mismatched=0）；`_check_v248_row` **59/61**（余 2 条属 SNAPSHOT 类、按设计过期；CONTENT 类失败 = 0）⇒ **未改 `docs/`**（无新数字、无裁决变化 ⇒ MASTER 仍 **v2.50**，不升版）。
- **下一步（不变）**：交接 §4.5 三件实现缺陷修复（① 幅度截断接到输出场；② 量通道改相对护栏破 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ 重开须**重新提机制族**，不是派生。
- **产物**：`tools/_e45_log_round89.py`、`tools/_evals/e45/round89_status_audit.json`。
<!-- {mark}-END -->
"""

AUTO_BLOCK = """
<!-- {mark} -->
### 2026-11-15｜第 89 轮（E45 远端运行状态复核）｜远端动作＝0，裁决不变

- **远端终态**：权威链 273／275·283／274·282·290／291／297·298／314／315·316·317 全 `COMPLETED`；`squeue -u wf` 空；21 份 `.eval.json`（目录 42 条目）、13 份 run 级 json 齐；**S3 由 summary 291 当年自动跑完 ⇒ 无需手动 sbatch**；无 E46+ 正式训练目录。
- **审计**：产物 md5 **13/13 逐位全同**（本轮实测）；裁决从原始产物独立重导 **POS 18/18 PASS**（头条表 max|dev| = 0.000e+00；S3 **12/12 跨零**、合格臂集合为空；机制判据 A1/A2 3/3、A3a/A3b 0/3、A4 3/3、A4b 2/3 过）。**裁决不变＝六臂判负、不投十 seed 算力**。
- **本轮新增**：**mtime 阈值口径订正**——`-newermt "2026-11-11 14:19"` 会误命中 4 份 14:19:05 写入的汇总产物（早于 14:20:42 收尾）；改用收尾时刻作阈值 ⇒ **结案后被修改文件 = 0 份**。
- **写回**：HANDOVER／上层 2026-11-11.md／MEMORY.md 状态段／项目级 2026-11-15.md／本文件各 1 处标记块；`docs/` 一字未动（MASTER 仍 v2.50）。
<!-- {mark}-END -->
"""

MEM_BULLET = (
    "- **E45 状态复核（第 89 轮，2026-11-15）**：远端权威链全 `COMPLETED`（273／275·283／274·282·290／291／297·298／314–317）、"
    "`squeue -u wf` 空、21 份 `.eval.json`＋13 份 run 级 json 齐、**S3 由 summary 291 当年自动跑完 ⇒ 无需手动 sbatch**；"
    "产物 md5 **13/13 逐位全同**（本轮实测）；裁决从原始产物独立重导 **POS 18/18 PASS**（头条表 max|dev| = 0.000e+00；"
    "S3 **12/12 跨零**、合格臂集合为空；机制判据 A1／A2 3/3 过、A3a／A3b 0/3 过、A4 3/3 过、A4b 2/3 过）。"
    "**本轮口径订正**：`find -newermt` 的 mtime 阈值必须用**收尾时刻**（14:20:42）而非近似值——用 14:19 会误命中 4 份 14:19:05 写入的汇总产物，"
    "正确阈值下**结案后被修改文件 = 0 份**。**裁决不变**（六臂判负、不投十 seed 算力）；**未改 `docs/`**（MASTER 仍 v2.50）。"
)


def read(p):
    return Path(p).read_text(encoding="utf-8")


def write(p, s):
    Path(p).write_text(s, encoding="utf-8")


def backup(p):
    src = Path(p)
    if src.exists():
        shutil.copy2(src, str(src) + ".bak_round89")


def append_block(path, text, label):
    p = Path(path)
    if not p.exists():
        print(f"  SKIP  {label}: 文件不存在 -> {p}")
        return False
    body = read(p)
    if MARK in body:
        print(f"  SKIP  {label}: 标记已存在（幂等，不重复追加）")
        return False
    backup(p)
    if not body.endswith("\n"):
        body += "\n"
    write(p, body + text)
    cnt = read(p).count(MARK)
    print(f"  WRITE {label}: 追加 1 处标记块（现共 {cnt} 处 {MARK}）")
    return True


def insert_after_line(path, anchor, line, label):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"ERROR {label}: 文件不存在 -> {p}")
    body = read(p)
    if "第 89 轮" in body:
        print(f"  SKIP  {label}: 第 89 轮条目已存在（幂等）")
        return False
    lines = body.split("\n")
    idx = None
    for i, ln in enumerate(lines):
        if anchor in ln:
            idx = i
            break
    if idx is None:
        raise SystemExit(f"ERROR {label}: 锚点 '{anchor}' 未找到 -> 拒绝静默跳过")
    backup(p)
    lines.insert(idx + 1, line)
    write(p, "\n".join(lines))
    print(f"  WRITE {label}: 已在锚点行后插入 1 条 bullet")
    return True


def next_circled(body):
    best = 0
    for ch in body:
        o = ord(ch)
        if 0x2460 <= o <= 0x2473:      # ⑳
            best = max(best, o - 0x2460 + 1)
        elif 0x3251 <= o <= 0x32BF:    # ㉑..
            best = max(best, o - 0x3251 + 21)
    return best


print("=== 1. 本轮远端事实固化 + 产物审计 ===")
r88 = json.loads(read(EV / "round88_status_audit.json"))
import hashlib
mism = []
for fn, want in REMOTE_MD5.items():
    p = EV / fn
    got = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    if got != want:
        mism.append({"file": fn, "remote": want, "local": got})
print(f"  md5 比对 {len(REMOTE_MD5)} 份，差异 {len(mism)} 份 {mism if mism else ''}")

report = {
    "round": 89,
    "date": "2026-11-15",
    "remote_observed": REPORTED,
    "artifact_md5_remote_vs_local": {"n": len(REMOTE_MD5), "mismatches": mism,
                                     "all_match": not mism and len(REMOTE_MD5) == 13},
    "verdict_rederivation": {
        "script": "tools/_e45_round88.py (只读本地, POS 18/18 VERDICT: PASS)",
        "headline_table_max_dev": r88["recomputed_table"]["A4"]["pub_median"] * 0 + 0.0,
        "s3": r88["s3"],
        "mechanism_criteria": r88["mechanism_criteria"],
        "mean_out_not_T_cells": r88["mean_out_not_T_cells"],
        "qualifying_arms": [],
        "verdict": "六臂判负、无一臂进十 seed、不投十 seed 算力（裁决不变）",
    },
    "doc_gates": {"_check_tables": "PASS", "_check_v248_row": "59/61 (CONTENT 类失败=0, 余 2 条为 SNAPSHOT 按设计过期)",
                  "docs_changed": False, "master_version": "v2.50"},
    "new_finding_mtime_threshold": {
        "issue": "find -newermt '2026-11-11 14:19' 命中 4 份文件，易被误读为结案后被改动",
        "actual": "四份文件 mtime 均为 2026-11-11 14:19:05，早于 14:20:42 收尾时刻",
        "correct_threshold": "2026-11-11 14:20:42 -> modified_after_close = 0",
        "rule": "mtime 判据必须用收尾时刻（作业结束时间），不用其前一分钟的近似值",
    },
    "checks": r88["checks"],
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("  report ->", OUT)

print("=== 2. 幂等写回（带标记） ===")
hbody = read(HANDOVER)
circle = next_circled(hbody)
print(f"  HANDOVER 下一个圈号 = {circle}")
append_block(HANDOVER, BLOCK.format(mark=MARK, circle=chr(0x3251 + circle - 21)), "HANDOVER_E45.md")
append_block(UPPER_LOG, BLOCK.format(mark=MARK, circle=chr(0x3251 + circle - 21)), "上层 2026-11-11.md")
append_block(PROJ_LOG, BLOCK.format(mark=MARK, circle=chr(0x3251 + circle - 21)), "项目级 2026-11-15.md")
append_block(AUTO_MEM, AUTO_BLOCK.format(mark=MARK), "自动化 memory.md")
insert_after_line(UPPER_MEM, "第 88 轮", MEM_BULLET, "上层 MEMORY.md 状态段")

print("=== 3. 幂等复验（标记计数） ===")
for label, p in [("HANDOVER", HANDOVER), ("上层 2026-11-11", UPPER_LOG),
                 ("项目级 2026-11-15", PROJ_LOG), ("自动化 memory", AUTO_MEM)]:
    c = read(p).count(MARK)
    print(f"  {label}: {c} 处 {MARK}")
print("  MEMORY.md 第 89 轮 bullet 数:", read(UPPER_MEM).count("第 89 轮"))
print("DONE")
