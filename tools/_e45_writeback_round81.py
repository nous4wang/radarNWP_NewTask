#!/usr/bin/env python3
"""Round-81 write-back (idempotent, marker-based).

Touches ONLY the memory/handover files. `docs/` is intentionally NOT touched:
no new numbers, no verdict change -> MASTER stays v2.50.

Idempotency contract:
  * marker blocks (E45-R81-BEGIN/END) are stripped before re-append
  * MEMORY.md bullets are matched by regex, not by hard-coded round number
  * table rows are built from cell lists; the separator is inserted by code
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

BEGIN = "<!-- E45-R81-BEGIN -->"
END = "<!-- E45-R81-END -->"

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
    """Remove any existing E45-R81 marker block (idempotency)."""
    pat = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END) + r"\r?\n?", re.S
    )
    return pat.sub("", text)


def row(cells: list[str]) -> str:
    """Build a markdown table row from a cell list; the separator is added here."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = [row(header), row(["---"] * len(header))]
    out.extend(row(r) for r in rows)
    return "\n".join(out)


HEAD_TABLE = table(
    ["臂", "中位 CSI@2.0", "过地板通过率"],
    [[a, m, p] for a, m, p in HEAD],
)

HANDOVER_BLOCK = f"""{BEGIN}
### 2026-11-15｜第 81 轮（E45 远端运行状态复核：权威链终态 ＋ 四重审计 ＋ **epoch 轴与主跑评测器级一致性首次核验**）

- **判定**：远端 E45 **全部完成、无新活动、裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。**远端动作：零**（只读状态查询；未 sbatch、未下载、未同步、未改远端状态）。
- **远端**：权威链 **273（提交器，COMPLETED 05:04:04）／274·282·290（stage2）／275·283（训练数组）／291（summary）／297·298（诊断）／314（gpuprobe）／315·316·317（epoch 轴）全部 `COMPLETED`**；提交器 273 日志尾部写明 `summary job id : 291` ＋ `the summary (job 291) finished: COMPLETED` ＋ `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **S3 等偏置对照当年即由汇总作业自动跑完（未手动 sbatch）**；`runs/e45/evals/` **21 份 `.eval.json`**（目录共 **42 份**＝21 评测＋21 窗表）、`runs/e45/` **13 份** run 级 json 齐；`s3_equalbias.json` 在位（7325 B，Nov 11 14:19）；`squeue -u wf` 里 8 个 **`qlfn2-f37`（job 327–334）** 属同一 `wf` 账号下的**另一条项目线**（与 E45 无关，不干预）；Slurm 控制器 `active`；`runs/` 下**无 E46+ 目录**。**易误读点（沿用第 73 轮）**：`sacct` 里的 102 `CANCELLED+`／103／118／265 是 v2.42/v2.43 修传参 bug 前的**废作业**，**不属权威链**。
- **审计一（产物一致性）**：13 份 run 级 json 的远端 md5 与本地 `tools/_evals/e45/` **13/13 逐位相同、0 处漂移** ⇒ 已发布头条表与 S3 裁决的数据源自 2026-11-11 结案以来**未被改动**，本轮**无需重新下载**。
- **审计二（已修实现缺陷的同步漂移）**：远端 4 个已修文件 md5 与第 71–80 轮登记值**逐位相同**（`scaled_structure/model.py` `77f18b6c…`、`topo_simvp/model.py` `f65554ca…`、`tools/_e41_struct_metrics.py` `742503c8…`、`tools/_e45_mechanism_probe.py` `e7d3666a…`），**对照组 `amp_monotone_gate/model.py` 双方同为 `e90857d3…`** ⇒ **待同步最小集仍是这 4 个文件**、其余 E45 运行时代码无漂移 ⇒ **未同步即未重训重测 ⇒ 裁决不变**。
- **审计三（文档闸门）**：`_check_tables` **PASS**、`_check_v245_row` **PASS**、`_e45_docs_numbers` **PASS**；`_check_v248_row` **POS 59/61 / NEG 2/61**（＝第 70、72–80 轮登记的**按设计过期**基线，余两项为「版本指针须停在 v2.48」「v2.48 行后须空行」类快照断言，**与实验数字无关**）；**CONTENT 类失败数 = 0**（五份权威文档里的 E45 承重数字仍在）。
- **审计四（本轮新增：epoch 轴与主跑的评测器级一致性）**：前 80 轮分别核过「主跑产物 md5」与「epoch 轴产物完备性」，**却从未把两者的 ep00 网格放在同一台仪器上比对**。本轮按 `(臂, seed, epoch)` **完整键**索引两份评测网格后：主跑 **21 格**（7 臂×3 seed，仅 ep00）、epoch 轴 **63 格**（7 臂×3 seed×ep00/01/02）；**ep00 的 21 格逐格 CSI@2.0 差异 max|diff| = 0.000e+00（逐位相同）**，且 **ep01 与 ep00 在全部 21 格都不同（最小差 0.00587）** ⇒ **epoch 轴不是主跑的副本，两条链在同一批 checkpoint 上给出完全一致的峰值端读数，而 epoch 轴确实换了 epoch**。这条把第 65–68、77 轮的 epoch 轴判读从「同一台仪器自证」升级为**两台独立作业的交叉验证**。
- **裁决复验**：S3 护栏 **12/12 区间全部跨零**（本轮按 `arms/<臂>/<raw|equal_bias>/{{point,ci_lo,ci_hi,crosses_zero}}` **正确键**独立重算）、6 条 `s3_verdict` 齐、**合格臂集合为空**；头条表由逐 seed 快照**精确 key 匹配**重算：中位 CSI@2.0 max|dev| = 4.53e−07、过地板通过率逐臂一致、**超 Oracle 0.1653 的种子只有 A3b、A4 各 1/3（＝seed 0）**；累积快照守卫通过（无任一臂三列全等）；`tools/_e45_verify_round71.py` **POS 27/27 VERDICT: PASS**、`tools/_e45_verify_round78.py` **POS 18/18 VERDICT: PASS**、`tools/_e45_verify_round81.py` **POS 27/27 VERDICT: PASS** ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**。

{HEAD_TABLE}

- **写回**：**未改 `docs/`**（无新数字、无裁决变化 ⇒ **不升版**，MASTER 仍 **v2.50**，v2.50 属 E46）；仅按惯例把本轮登记进 HANDOVER／逐日日志／MEMORY 状态段。
- **本轮产物**：`tools/_e45_verify_round81.py`、`tools/_evals/e45/round81_status_audit.json`。
- **本轮踩坑（可复用规则，两处同一族）**：① **S3 区间不是「长度为 2 的列表」**——真实结构是 `arms/<臂>/<raw|equal_bias>/{{point,ci_lo,ci_hi,crosses_zero}}`，按列表假设去找会命中 **0 个区间**（本轮首版即栽在这里，报成「12 个区间一个都没找到」）；② **索引键必须包含全部区分维度**——epoch 轴目录里 `ep00/ep01/ep02` 共用同一 `(臂, seed)`，只用 `(臂, seed)` 做键会把三个 epoch 塌成一行，于是拿**主跑的 ep00**去比**epoch 轴的 ep02**，报出一个 0.134 的**假差异**（看起来像结论冲突，实为键设计错误）。⇒ **新纪律：凡按多个维度索引一份产物，键必须含全部维度；且必须断言「预期格数」（本轮 21 / 63），而不是只断言「脚本跑通」**（与第 66、74、77 轮同族，但本轮是首次由**同一台仪器内部的键塌陷**造成，已升级为可复用规则）。
- **三项待用户确认仍挂着**（未擅自执行）：① E46 第二轮探针结果是否写回 MASTER（＝升 v2.51）；② 4 个修复文件按最小集同步远端；③ dx=3px／20–400mm 扫描须先预注册。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后复核、**不构成新判据、不改变任何裁决**；③ **MASTER 不升版**（仍 v2.50）。
{END}
"""

PROJ_BLOCK = f"""{BEGIN}
### E45 第 81 轮（2026-11-15）｜远端运行状态复核 ＋ 四重审计 ＋ epoch 轴与主跑评测器级一致性首次核验

- 远端权威链 273/274·282·290/275·283/291/297·298/314/315·316·317 全 `COMPLETED`；21 份 `.eval.json` 齐；13 份 run 级 json 远端↔本地 md5 13/13 全同；S3 由 summary 291 当年自动跑完；无 E46+ 目录；队列里 8 个 `qlfn2-f37`（job 327–334）属另一项目线。**远端动作：零**。
- 三重审计：产物一致 13/13；同步漂移仍恰为 4 个已修文件（对照组 md5 相同）⇒ 裁决不变；文档闸门 `_check_tables`／`_check_v245_row`／`_e45_docs_numbers` PASS，`_check_v248_row` 59/61（按设计过期基线，CONTENT 类失败 = 0）。
- **第四重审计（本轮新增）**：epoch 轴与主跑的**评测器级交叉验证** ⇒ ep00 的 21 格 CSI@2.0 **逐位相同（max|diff| = 0）**、ep01 与 ep00 全 21 格都不同 ⇒ epoch 轴不是副本，两条独立作业链在峰值端读数完全一致。
- 裁决复验：S3 **12/12 区间跨零**（本轮按正确键独立重算）、合格臂集合为空；头条表中位数 max|dev| = 4.53e−07、通过率逐臂一致；`_e45_verify_round71` 27/27、`_e45_verify_round78` 18/18、`_e45_verify_round81` 27/27 均 PASS ⇒ **六臂判负、不投十 seed 算力**。
- **未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。产物：`tools/_e45_verify_round81.py`、`tools/_evals/e45/round81_status_audit.json`。
- 新登记纪律：S3 区间不是「长度为 2 的列表」而是 `arms/<臂>/<变体>/{{point,ci_lo,ci_hi,crosses_zero}}`；**索引键必须含全部区分维度并断言预期格数**（本轮 `(臂,seed)` 键把 ep00/01/02 塌成一行，制造了 0.134 的假差异）。
{END}
"""

MEM_CONCL = (
    "- **E45 状态复核（第 81 轮，2026-11-15）**：远端权威链全 `COMPLETED`、21 份评测齐、"
    "13 份 run 级 json 与本地 md5 **13/13 逐位相同**、E45 队列空（他项目线 8 个 `qlfn2-f37` 在跑）、"
    "无 E46+ 目录 ⇒ **E45 无新活动**。文档闸门 `_check_tables`／`_check_v245_row`／`_e45_docs_numbers` PASS；"
    "`_check_v248_row` **59/61**（余两项为按设计过期的版本指针快照断言，与实验数字无关；CONTENT 类失败 = 0）。"
    "**本轮新增第四重审计**：epoch 轴与主跑的**评测器级交叉验证**——ep00 的 21 格 CSI@2.0 **逐位相同"
    "（max|diff| = 0）**、ep01 与 ep00 全 21 格都不同 ⇒ epoch 轴不是副本，两条独立作业链峰值端读数一致。"
    "**裁决不变**（六臂判负、不投十 seed 算力）。审计脚本 `tools/_e45_verify_round81.py`（POS 27/27 VERDICT: PASS）。"
)

MEM_STATUS = (
    "- **E45 状态复核（第 81 轮，2026-11-15，远端终态）**：远端权威链全 `COMPLETED`"
    "（273／274·282·290／275·283／291／297·298／314／315·316·317）、`runs/e45/evals/` 42 份"
    "（21 `.eval.json` ＋ 21 `.windows.json`）、13 份 run 级 json 远端↔本地 md5 **13/13 全同**；"
    "S3 由 summary 291 当年自动跑完（未手动 sbatch）；队列里 8 个 `qlfn2-f37`（job 327–334）属**另一条项目线**、"
    "不干预；控制器 `active`；无 E46+ 目录 ⇒ **E45 无新活动**。文档闸门 `_check_tables`／`_check_v245_row`／"
    "`_e45_docs_numbers` PASS、`_check_v248_row` **59/61**（按设计过期的版本指针快照断言）；裁决复验 "
    "`_e45_verify_round71`（27/27）、`_e45_verify_round78`（18/18）、`_e45_verify_round81`（27/27）均 PASS ⇒ "
    "**裁决不变**（六臂判负、不投十 seed 算力）。**本轮新增第四重审计**：epoch 轴与主跑的**评测器级交叉验证**——"
    "ep00 的 21 格 CSI@2.0 **逐位相同（max|diff| = 0）**、ep01 与 ep00 全 21 格都不同 ⇒ epoch 轴不是副本。"
    "**未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。新登记纪律：**索引键必须含全部区分维度"
    "并断言预期格数**（本轮 `(臂,seed)` 键把 ep00/01/02 塌成一行，制造 0.134 的假差异）；**S3 区间不是长度为 2 的列表**。"
)


def append_block(path: str, block: str) -> None:
    text = strip_marker(read(path))
    # Normalize the tail so repeated runs are byte-identical: strip trailing
    # whitespace, then append exactly one blank line + block + newline.
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
    # replace from the back so earlier offsets stay valid
    for idx, repl in ((1, MEM_STATUS), (0, MEM_CONCL)):
        m = matches[idx]
        text = text[:m.start()] + repl + "\n" + text[m.end():]
    write(UPPER_MEM, text)


def main() -> int:
    targets = [HANDOVER, UPPER_LOG, PROJ_LOG, UPPER_MEM]
    for t in targets:
        if not os.path.isfile(t):
            raise SystemExit(f"missing write-back target: {t}")

    append_block(HANDOVER, HANDOVER_BLOCK)
    append_block(UPPER_LOG, HANDOVER_BLOCK)
    append_block(PROJ_LOG, PROJ_BLOCK)
    replace_mem_bullets()

    for t in (HANDOVER, UPPER_LOG, PROJ_LOG):
        txt = read(t)
        n = txt.count(BEGIN)
        if n != 1:
            raise SystemExit(f"{t}: marker count {n} != 1")
        print(f"{os.path.basename(t):22s} md5={md5(t)}  markers={n}")
    mem_txt = read(UPPER_MEM)
    n81 = len(re.findall(r"E45 状态复核（第 81 轮", mem_txt))
    n_old = len(re.findall(r"E45 状态复核（第 \d+ 轮", mem_txt))
    if n81 != 2 or n_old != 2:
        raise SystemExit(
            f"MEMORY.md: expected 2 round-81 status bullets, got {n81} "
            f"(total status bullets {n_old})"
        )
    print(f"{'MEMORY.md':22s} md5={md5(UPPER_MEM)}  bullets={n81}")
    print("WRITEBACK: OK (docs/ intentionally untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
