#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round-82 write-back (idempotent, marker-based).

Touches ONLY the memory/handover files. `docs/` is intentionally NOT touched:
no new numbers, no verdict change -> MASTER stays v2.50 (v2.50 belongs to E46).

Idempotency contract (same as round 81):
  * marker blocks (E45-R82-BEGIN/END) are stripped before re-append;
  * MEMORY.md bullets are matched by regex, not by a hard-coded round number;
  * table rows are built from cell lists; the separator is inserted by code.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

UPPER = r"R:/1radarNWPprediction/.workbuddy/memory"
PROJ = r"R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory"

HANDOVER = os.path.join(UPPER, "HANDOVER_E45.md")
UPPER_LOG = os.path.join(UPPER, "2026-11-11.md")
UPPER_MEM = os.path.join(UPPER, "MEMORY.md")
PROJ_LOG = os.path.join(PROJ, "2026-11-15.md")
AUTO_MEM = os.path.join(PROJ, "automations",
                        "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

BEGIN = "<!-- E45-R82-BEGIN -->"
END = "<!-- E45-R82-END -->"

# headline table: arm -> (median CSI@2.0, pass-rate over floor)
HEAD = [
    ("B0", "0.121817", "2/3"),
    ("A1", "0.118397", "2/3"),
    ("A2", "0.112003", "2/3"),
    ("A3a", "0.110015", "3/3"),
    ("A3b", "0.116055", "3/3"),
    ("A4", "0.143896", "2/3"),
    ("A4b", "0.137567", "2/3"),
]


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def strip_marker(text: str) -> str:
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\r?\n?", re.S)
    return pat.sub("", text)


def row(cells: list) -> str:
    """Build a markdown table row from a cell list; the separator is added here."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(header: list, rows: list) -> str:
    out = [row(header), row(["---"] * len(header))]
    out.extend(row(r) for r in rows)
    return "\n".join(out)


HEAD_TABLE = table(
    ["臂", "中位 CSI@2.0", "过地板通过率"],
    [[a, m, p] for a, m, p in HEAD],
)

HANDOVER_BLOCK = f"""{BEGIN}
### 2026-11-15｜第 82 轮（E45 远端运行状态复核：权威链终态 ＋ **运行树 mtime 冻结证明** ＋ **意外写回副作用的定位与修复**）

- **判定**：远端 E45 **全部完成、无新活动、裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。**远端动作：零**（只读状态查询；未 sbatch、未同步、未改远端状态）。
- **远端**：权威链 **273（提交器，COMPLETED 05:04:04）／274·282·290（stage2）／275·283（训练数组）／291（summary）／297·298（诊断）／314（gpuprobe）／315·316·317（epoch 轴）全部 `COMPLETED`**；提交器 273 日志尾部写明 `summary job id : 291` ＋ `the summary (job 291) finished: COMPLETED` ＋ `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **S3 等偏置对照当年即由汇总作业自动跑完（未手动 sbatch）**；`runs/e45/evals/` **21 份 `.eval.json`**（目录共 **42 份**）、`runs/e45/` **13 份** run 级 json 齐；`s3_equalbias.json` 在位（7325 B，Nov 11 14:19）；`squeue -u wf` 里 8 个 **`qlfn2-f37`（job 327–334）** 属同一 `wf` 账号下的**另一条项目线**（与 E45 无关，不干预）；Slurm 控制器 `active`；`runs/` 下**无 E46+ 目录**。**易误读点（沿用第 73 轮）**：`sacct` 里的 102 `CANCELLED+`／103／118／265 是 v2.42/v2.43 修传参 bug 前的**废作业**，**不属权威链**；本轮把这 4 个作业**单独断言**为「已知的废作业集合且集合未变」，避免新失败藏在旧噪声里。
- **审计一（产物一致性）**：13 份 run 级 json 的远端 md5 与本地 `tools/_evals/e45/` **13/13 逐位相同、0 处漂移**；显式断言期望格数 **21 份评测 / 42 份 evals 目录 / 13 份 run 级 json** ⇒ 已发布头条表与 S3 裁决的数据源自 2026-11-11 结案以来**未被改动**。
- **审计二（已修实现缺陷的同步漂移）**：远端 4 个已修文件 md5 与第 71–81 轮登记值**逐位相同**（`scaled_structure/model.py` `77f18b6c…`、`topo_simvp/model.py` `f65554ca…`、`tools/_e41_struct_metrics.py` `742503c8…`、`tools/_e45_mechanism_probe.py` `e7d3666a…`），**对照组 `amp_monotone_gate/model.py` 双方同为 `e90857d3…`** ⇒ **待同步最小集仍是这 4 个文件** ⇒ **未同步即未重训重测 ⇒ 裁决不变**。
- **审计三（文档闸门）**：`_check_tables`／`_check_v245_row`／`_e45_docs_numbers` **PASS**；`_check_v246/247/248_row` 的失败项**按 CONTENT／SNAPSHOT 分类后 CONTENT 类 = 0**（余 4 项全是「版本指针须停在旧版」类快照断言，与实验数字无关）。
- **审计四（本轮新增：运行树 mtime 冻结证明）**：前 81 轮核的是 **13 份 json 的 md5**——那只证明**已发布产物**没漂移，**不能证明运行树的其余部分**（checkpoint、逐窗表、探针文件）在结案后没被动过。本轮改用**全树 mtime** 断言：`runs/e45/` 共 **244 份**文件，**在 2026-11-11 14:19 结案之后被修改的文件数 = 0**；全部 139 份「晚于 2026-11-12」的文件**只存在于 `runs/e45_epochaxis/`**（139/139，诊断性 epoch 轴产物，不是主跑输出）；且 `sacct -S 2026-11-12` 中**不存在任何 `e45-train`／`e45-stage2`／`e45-submit` 作业** ⇒ **主链从未被重启，产物树自结案起冻结**。
- **裁决复验**：S3 护栏 **12/12 区间全部跨零**、6 条 `s3_verdict` 齐、**合格臂集合为空**；头条表由逐 seed 快照**精确 key 匹配**重算，中位 CSI@2.0 max|dev| = 4.53e−07、过地板通过率逐臂一致、**超 Oracle 0.1653 的种子只有 A3b、A4 各 1/3（＝seed 0）**；`tools/_e45_verify_round71.py` **27/27 PASS**、`_e45_verify_round78.py` **18/18 PASS**、`_e45_verify_round81.py` **27/27 PASS**、本轮 `tools/_e45_verify_round82.py` **POS 15/15 VERDICT: PASS** ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**。

{HEAD_TABLE}

- **本轮事故与修复（本轮唯一真实收获）**：本轮为跑文档闸门，用一个 `for f in $(grep -ln …)` 的**批量循环**扫出了历史脚本并**全部执行**——其中 `_e45_log_round16.py`（写回脚本）与 `_e45_verify_round53.py`（写回脚本）**并非只读**，它们把**已存在的历史块又追加了一份**：`_e45_log_round16.py` 用**当前运行时间戳**在 2026-11-12.md 与自动化 memory 里追加了第二个「第 16 次独立复核」块，`_e45_verify_round53.py` 在 2026-11-14.md 与自动化 memory 里追加了第二个「第 53 轮」块。⇒ **修复**：新脚本 `tools/_e45_repair_round82.py`，按**位置**（不是文本相等——两份措辞不同，按字节去重会漏掉）判定哪一份是多余的：2026-11-12.md 里**带当天时间戳**的那份是多余的（保留 `2026-11-12 07:07` 的正本）、2026-11-14.md 里**紧贴 `<!-- E45-R62-BEGIN -->` 之前**的那份是多余的（保留含「补记 54/55」的正本）、自动化 memory 里 **`E45-R81-END` 与 `E45-R62-BEGIN` 之间**的整段是多余的；每个文件先落 `.bak` 再改，改完逐项复验唯一性。⇒ **四份权威文件（`HANDOVER_E45.md`／`MEMORY.md`／`2026-11-11.md`／项目级 `2026-11-15.md`）的 md5 与第 81 轮基线逐位相同，未受任何影响**；`docs/` 全程未动。⇒ **新纪律：跑闸门只能逐个点名执行，禁止用 `grep -ln` 批量扫脚本全量执行——「名字里带 check/verify」不等于只读，历史脚本里混着写回脚本**。
- **写回**：**未改 `docs/`**（无新数字、无裁决变化 ⇒ **不升版**，MASTER 仍 **v2.50**，v2.50 属 E46）；仅按惯例把本轮登记进 HANDOVER／逐日日志／MEMORY 状态段。
- **本轮产物**：`tools/_e45_verify_round82.py`、`tools/_e45_repair_round82.py`、`tools/_evals/e45/round82_status_audit.json`、备份目录 `.workbuddy/memory/_repair_r82_backup/`。
- **三项待用户确认仍挂着**（未擅自执行）：① E46 第二轮探针结果是否写回 MASTER（＝升 v2.51）；② 4 个修复文件按最小集同步远端；③ dx=3px／20–400mm 扫描须先预注册。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后复核、**不构成新判据、不改变任何裁决**；③ **MASTER 不升版**（仍 v2.50）。
{END}
"""

PROJ_BLOCK = f"""{BEGIN}
### E45 第 82 轮（2026-11-15）｜远端运行状态复核 ＋ 运行树 mtime 冻结证明 ＋ 意外写回副作用的定位与修复

- 远端权威链 273/274·282·290/275·283/291/297·298/314/315·316·317 全 `COMPLETED`；21 份 `.eval.json` 齐；13 份 run 级 json 远端↔本地 md5 13/13 全同；S3 由 summary 291 当年自动跑完；无 E46+ 目录；队列里 8 个 `qlfn2-f37`（job 327–334）属另一项目线。**远端动作：零**。
- 三重审计：产物一致 13/13（显式断言 21/42/13 格数）；同步漂移仍恰为 4 个已修文件（对照组 md5 相同）⇒ 裁决不变；文档闸门按 CONTENT／SNAPSHOT 分类后 **CONTENT 类失败 = 0**。
- **第四重审计（本轮新增）**：运行树 **mtime 冻结证明** —— `runs/e45/` **244 份文件中，结案后（2026-11-11 14:19 之后）被修改的 = 0 份**；139 份「晚于 2026-11-12」的文件**全部**在 `runs/e45_epochaxis/`（诊断产物）；`sacct -S 2026-11-12` **无任何 e45-train／e45-stage2／e45-submit** ⇒ **主链从未重启**。
- 裁决复验：S3 **12/12 区间跨零**、合格臂集合为空；头条表中位数 max|dev| = 4.53e−07、通过率逐臂一致；`_e45_verify_round71` 27/27、`_e45_verify_round78` 18/18、`_e45_verify_round81` 27/27、`_e45_verify_round82` **15/15** 均 PASS ⇒ **六臂判负、不投十 seed 算力**。
- **本轮事故与修复**：批量扫脚本执行闸门时**误跑了两个历史写回脚本**（`_e45_log_round16.py`、`_e45_verify_round53.py`），它们把已存在的历史块**又追加了一份**（2026-11-12.md 的「第 16 次独立复核」、2026-11-14.md 与自动化 memory 的「第 53 轮」）。已用 `tools/_e45_repair_round82.py` **按位置**判定并删除多余副本（先落 `.bak` 再改，逐项复验唯一性）；**四份权威文件 md5 与第 81 轮基线逐位相同、未受影响**，`docs/` 全程未动。**新纪律：跑闸门只能逐个点名执行，禁止批量扫脚本全量执行。**
- **未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。产物：`tools/_e45_verify_round82.py`、`tools/_e45_repair_round82.py`、`tools/_evals/e45/round82_status_audit.json`。
{END}
"""

MEM_CONCL = (
    "- **E45 状态复核（第 82 轮，2026-11-15）**：远端权威链全 `COMPLETED`、21 份评测齐、"
    "13 份 run 级 json 与本地 md5 **13/13 逐位相同**、E45 队列空（他项目线 8 个 `qlfn2-f37` 在跑）、"
    "无 E46+ 目录 ⇒ **E45 无新活动**。文档闸门按 CONTENT／SNAPSHOT 分类后 **CONTENT 类失败 = 0**。"
    "**本轮新增第四重审计**：**运行树 mtime 冻结证明**——`runs/e45/` **244 份文件中结案后被修改的 = 0 份**，"
    "139 份「晚于 2026-11-12」的文件全部在 `runs/e45_epochaxis/`（诊断产物），"
    "`sacct -S 2026-11-12` 无任何 e45-train／e45-stage2／e45-submit ⇒ **主链从未重启**。"
    "**裁决不变**（六臂判负、不投十 seed 算力）。审计脚本 `tools/_e45_verify_round82.py`（POS 15/15 VERDICT: PASS）。"
    "**本轮事故已修复**：批量扫脚本跑闸门时误跑了两个历史写回脚本，重复追加了历史块；"
    "已由 `tools/_e45_repair_round82.py` 按位置删除多余副本（先落 `.bak`），四份权威文件 md5 未变。"
)

MEM_STATUS = (
    "- **E45 状态复核（第 82 轮，2026-11-15，远端终态）**：远端权威链全 `COMPLETED`"
    "（273／274·282·290／275·283／291／297·298／314／315·316·317）、`runs/e45/evals/` 42 份"
    "（21 `.eval.json` ＋ 21 `.windows.json`）、13 份 run 级 json 远端↔本地 md5 **13/13 全同**；"
    "S3 由 summary 291 当年自动跑完（未手动 sbatch）；队列里 8 个 `qlfn2-f37`（job 327–334）属**另一条项目线**、"
    "不干预；控制器 `active`；无 E46+ 目录 ⇒ **E45 无新活动**。文档闸门按 CONTENT／SNAPSHOT 分类后 "
    "**CONTENT 类失败 = 0**；裁决复验 `_e45_verify_round71`（27/27）、`_e45_verify_round78`（18/18）、"
    "`_e45_verify_round81`（27/27）、`_e45_verify_round82`（**15/15**）均 PASS ⇒ "
    "**裁决不变**（六臂判负、不投十 seed 算力）。**本轮新增第四重审计**：**运行树 mtime 冻结证明**——"
    "`runs/e45/` **244 份文件中结案（2026-11-11 14:19）后被修改的 = 0 份**，139 份「晚于 2026-11-12」的文件"
    "**全部**在 `runs/e45_epochaxis/`（诊断产物），且 `sacct -S 2026-11-12` **无任何 e45-train／e45-stage2／"
    "e45-submit** ⇒ **主链从未重启**。**未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。"
    "**本轮事故已修复**：批量扫脚本跑闸门时误跑了 `_e45_log_round16.py`／`_e45_verify_round53.py` 两个"
    "**历史写回脚本**，重复追加了历史块（2026-11-12.md 的「第 16 次独立复核」、2026-11-14.md 与自动化 memory 的"
    "「第 53 轮」）⇒ 已由 `tools/_e45_repair_round82.py` **按位置**删除多余副本（先落 `.bak`、改后复验唯一性）；"
    "**四份权威文件 md5 与第 81 轮基线逐位相同、未受影响**，`docs/` 全程未动。"
    "新登记纪律：**跑闸门只能逐个点名执行，禁止用 `grep -ln` 批量扫脚本全量执行**——「名字里带 check/verify」"
    "不等于只读，历史脚本里混着写回脚本。"
)


def append_block(path: str, block: str) -> None:
    text = strip_marker(read(path))
    text = text.rstrip() + "\n\n" + block.rstrip() + "\n"
    write(path, text)


def replace_mem_bullets() -> None:
    text = read(UPPER_MEM)
    pat = re.compile(r"^- \*\*E45 状态复核（第 \d+ 轮[^\n]*\n", re.M)
    matches = list(pat.finditer(text))
    if len(matches) < 2:
        raise SystemExit(
            f"EXPECTED >=2 E45 status bullets in MEMORY.md, found {len(matches)} "
            "- refusing to write silently"
        )
    for idx, repl in ((1, MEM_STATUS), (0, MEM_CONCL)):
        m = matches[idx]
        text = text[:m.start()] + repl + "\n" + text[m.end():]
    write(UPPER_MEM, text)


def main() -> int:
    targets = [HANDOVER, UPPER_LOG, PROJ_LOG, UPPER_MEM, AUTO_MEM]
    for t in targets:
        if not os.path.isfile(t):
            raise SystemExit(f"missing write-back target: {t}")

    append_block(HANDOVER, HANDOVER_BLOCK)
    append_block(UPPER_LOG, HANDOVER_BLOCK)
    append_block(PROJ_LOG, PROJ_BLOCK)
    append_block(AUTO_MEM, PROJ_BLOCK)
    replace_mem_bullets()

    for t in (HANDOVER, UPPER_LOG, PROJ_LOG, AUTO_MEM):
        txt = read(t)
        n = txt.count(BEGIN)
        if n != 1:
            raise SystemExit(f"{t}: marker count {n} != 1")
        print(f"{os.path.basename(t):22s} md5={md5(t)}  markers={n}")
    mem_txt = read(UPPER_MEM)
    n82 = len(re.findall(r"E45 状态复核（第 82 轮", mem_txt))
    n_old = len(re.findall(r"E45 状态复核（第 \d+ 轮", mem_txt))
    if n82 != 2 or n_old != 2:
        raise SystemExit(
            f"MEMORY.md: expected 2 round-82 status bullets, got {n82} "
            f"(total status bullets {n_old})"
        )
    print(f"{'MEMORY.md':22s} md5={md5(UPPER_MEM)}  bullets={n82}")
    print("WRITEBACK: OK (docs/ intentionally untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
