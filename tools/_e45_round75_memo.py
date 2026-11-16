#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 第 75 轮写回：把本轮结论追加到 5 处记忆文件。

纪律（沿用本项目既有做法）：按标记定位、幂等；锚点找不到就报错，不静默跳过。
只写记忆文件，不改 docs/（无新数字、无裁决变化 ⇒ MASTER 不升版）。
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(r"R:/1radarNWPprediction/radar-nwp-bench")
UP = pathlib.Path(r"R:/1radarNWPprediction/.workbuddy/memory")

AUTO = REPO / ".workbuddy/memory/automations/e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md"
DAILY = UP / "2026-11-15.md"
HANDOVER = UP / "HANDOVER_E45.md"
MEMORY = UP / "MEMORY.md"
WSLOG = REPO / ".workbuddy/memory/2026-11-15.md"

BEGIN, END = "<!-- E45-R75-BEGIN -->", "<!-- E45-R75-END -->"

BLOCK = """
### 2026-11-15｜第 75 轮（状态复核 ＋ 远端 21 份逐评测原始 JSON 的最细粒度独立复核）

- **判定**：远端 E45 **全部完成、无新活动、裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。权威链 273/274/275/282/283/290/291/297/298/314/315/316/317 全 `COMPLETED`；提交器 273 日志确认 summary 291 当年即自动跑完 S3 等偏置对照（**无需手动 sbatch**）；21 份评测齐、13 份 run 级 json 齐、无 E46+ 目录、`squeue -u wf` 空。**易误读点（沿用第 73 轮）**：`sacct` 里的 102/103/118/265 是 v2.42/v2.43 修传参 bug 前的废作业，不属权威链路。
- **本轮真做的事**：前 74 轮核过状态、聚合产物 md5、文档闸门、已发布裁决，但**从未校验过 21 份 `*.eval.json`（评测器原始输出）本身**——整条证据链的最细一环。本轮首次把它们逐份取回（用**远端 md5** 锁定字节），并从**列联计数**独立重算指标。脚本 `tools/_e45_verify_round75.py`（只读本地，**POS 65/65 VERDICT: PASS**，产物 `tools/_evals/e45/round75_raw_eval_audit.json`）：
  1. **评测器自身算术自洽**：由 hits/misses/false_alarms 重算 CSI/BIAS/POD/FAR，21 格 × 4 阈值**最大偏差 0.000e+00**（此前所有层都直接信任评测器写出的字段，从未验证过算术本身）；
  2. **聚合 = 原始**：8 个字段 × 21 格与 `scores_all.json` **逐位相同（max|dev| = 0.000e+00）**；21 格样本数一律 538；
  3. **HANDOVER §3 公布数全部复现**：seed-0 CSI@2.0 七臂（最大偏差 4.526e−07 ＝ 六位小数舍入）、seed-0 BIAS@2.0 七臂（最大偏差 4.519e−05 ＝ 四位小数舍入）；
  4. **头条表从原始层独立重算**：过地板通过率与已发布一致（A3a 3/3、A3b 3/3、B0 2/3、A4b 2/3）；累积快照检测器未触发（无任一臂三种子数值全等）。
- **原始层新增读数（中位数 + 通过率，不作裁决）**：median CSI@2.0 / 中位 BIAS / 过地板 = B0 0.121817 / 0.509542 / 2/3；A1 0.118397 / 0.443021 / 2/3；A2 0.112003 / 0.449028 / 2/3；A3a 0.110015 / 0.639378 / 3/3；A3b 0.116055 / 0.618842 / 3/3；A4 0.143896 / 0.836171 / 2/3；A4b 0.137567 / 2.821505 / 2/3。⇒ **差值口径（median 差）A4 +0.0221、A4b +0.0157，与第 44 轮独立算出的数逐位吻合**（本轮是在原始层复现它）；**超 Oracle 上界 0.1653 的种子只有 A4 与 A3b 各 1/3**（都是 seed 0）⇒ 与「过预报换分」的既有叙事一致，这是**第一次在原始评测层**看到这一结构；A4b 的中位 BIAS 2.8215 ≈ 对照的 5.5 倍。
- **写回**：未改 `docs/`（无新数字、无裁决变化）⇒ **MASTER 不升版（仍 v2.49）**。第 75 轮记录已写入逐日日志、HANDOVER_E45.md、MEMORY.md 状态段、工作区记忆与自动化记忆。
- **产物**：`tools/_e45_verify_round75.py`、`tools/_evals/e45/round75_raw_eval_audit.json`、`tools/_evals/e45/e45_evals.tar.gz`（21 份原始评测 + 21 份 windows 明细，远端 md5 `2d859447fce615230d96b0c55536640f`）。
- **三项待用户确认仍挂着**：E46 三段 verdict 写回 MASTER（＝升 v2.50）／4 个修复文件按最小集同步远端／dx=3px、20–400mm 扫描须先预注册。
"""

MEM_BULLET = (
    "- **E45 第 75 轮（2026-11-15）**：远端状态不变（权威链全 `COMPLETED`、提交器 273 日志确认 summary 291 自动跑完 S3、"
    "21 份评测齐、13 份 json 远端↔本地 md5 13/13 全同、`squeue -u wf` 空、无 E46+ 目录）；**裁决不变**"
    "（六臂判负、无一臂进十 seed、不投十 seed 算力）；**本轮首次校验最细一环**——把 21 份 `*.eval.json`（评测器原始输出）"
    "逐份取回并用远端 md5 锁定字节，再由**列联计数**独立重算指标（脚本 `tools/_e45_verify_round75.py`，只读本地，"
    "**POS 65/65 VERDICT: PASS**）：**评测器算术自洽 max|dev| = 0.000e+00**、**聚合与原始逐位相同（21 格 × 8 字段）**、"
    "**HANDOVER §3 公布的 seed-0 CSI/BIAS 七臂全部复现**（偏差 4.5e−07 / 4.5e−05 ＝ 小数位舍入）、"
    "**头条表从原始层独立重算与已发布一致**（过地板 A3a 3/3、A3b 3/3、B0 2/3、A4b 2/3）；"
    "**原始层新增读数**：差值口径 A4 +0.0221、A4b +0.0157 **与第 44 轮独立算出的数逐位吻合**；"
    "**超 Oracle 0.1653 的种子只有 A4 与 A3b 各 1/3（都是 seed 0）**，A4b 中位 BIAS 2.8215 ≈ 对照 5.5 倍 "
    "⇒ 「过预报换分」的结构第一次在原始层被看到；**未改 `docs/`、MASTER 不升版（仍 v2.49）**；"
    "**新登记一条**：**复核要从最细一环往上推**——聚合产物自洽不等于底层评测器自洽，"
    "本轮才第一次验证了评测器自身的算术（列联计数 → 指标）。\n"
)

WS_NOTE = (
    "\n### 2026-11-15｜E45 第 75 轮：远端 21 份逐评测原始 JSON 的最细粒度独立复核\n"
    "- 远端 E45 权威链全 `COMPLETED`、无新活动、**裁决不变**（六臂判负、不投十 seed 算力）；MASTER 仍 v2.49（未改 docs/）。\n"
    "- 新增核验：21 份 `*.eval.json` 逐份取回（远端 md5 锁定）＋ 由列联计数独立重算 CSI/BIAS/POD/FAR ⇒ "
    "**评测器算术 max|dev| = 0.000e+00**、聚合与原始逐位相同、§3 公布数全部复现、头条表独立重算一致。"
    "脚本 `tools/_e45_verify_round75.py`（POS 65/65 PASS），产物 `tools/_evals/e45/round75_raw_eval_audit.json`。\n"
)


def upsert_block(path: pathlib.Path, body: str) -> str:
    """把 body 放进 BEGIN/END 标记之间（幂等：已存在则替换）。"""
    txt = path.read_text(encoding="utf-8")
    if BEGIN in txt and END in txt:
        i = txt.index(BEGIN)
        j = txt.index(END) + len(END)
        new = txt[:i] + BEGIN + body + END + txt[j:]
        act = "replaced"
    else:
        new = txt.rstrip("\n") + "\n" + BEGIN + body + END + "\n"
        act = "appended"
    path.write_text(new, encoding="utf-8")
    return act


def main() -> int:
    errs = []

    # 1) 自动化记忆（必需）
    if not AUTO.exists():
        errs.append(f"missing {AUTO}")
    else:
        act = upsert_block(AUTO, BLOCK)
        print(f"OK  automation memory: {act}")

    # 2) 逐日日志（上层记忆目录）
    if not DAILY.exists():
        errs.append(f"missing {DAILY}")
    else:
        act = upsert_block(DAILY, "\n### E45 第 75 轮（2026-11-15 07:4x）｜远端逐评测原始 JSON 的最细粒度独立复核\n"
                                  "- 远端权威链全 `COMPLETED`、无新活动、**裁决不变**（六臂判负、不投十 seed 算力）；MASTER 仍 v2.49。\n"
                                  "- 21 份 `*.eval.json` 逐份取回（远端 md5 锁定）＋列联计数独立重算 ⇒ 评测器算术 max|dev| = 0.000e+00、"
                                  "聚合与原始逐位相同、§3 公布数全部复现、头条表独立重算一致（A3a/A3b 3/3 过地板、B0/A4b 2/3）。\n"
                                  "- 差值口径 A4 +0.0221 / A4b +0.0157 与第 44 轮逐位吻合；超 Oracle 0.1653 的种子只有 A4、A3b 各 1/3（seed 0）。\n"
                                  "- 产物 `tools/_e45_verify_round75.py`、`tools/_evals/e45/round75_raw_eval_audit.json`。\n")
        print(f"OK  daily log: {act}")

    # 2b) 逐日日志 2026-11-11.md（E45 结案当天的日志；本轮结论按任务要求同步追加）
    DAILY11 = UP / "2026-11-11.md"
    if not DAILY11.exists():
        errs.append(f"missing {DAILY11}")
    else:
        act = upsert_block(DAILY11, BLOCK)
        print(f"OK  daily log 2026-11-11: {act}")

    # 3) HANDOVER_E45.md（追加一节，带标记）
    if not HANDOVER.exists():
        errs.append(f"missing {HANDOVER}")
    else:
        act = upsert_block(HANDOVER, "\n**⑪（2026-11-15 第 75 轮新增证据）远端 21 份逐评测原始 JSON 的最细粒度独立复核**——"
                                     "脚本 `tools/_e45_verify_round75.py`（只读本地，**POS 65/65 VERDICT: PASS**，产物 "
                                     "`tools/_evals/e45/round75_raw_eval_audit.json`）：前 74 轮核过状态、聚合产物 md5、文档闸门与已发布裁决，"
                                     "**但从未校验过 21 份 `*.eval.json`（评测器原始输出）本身**。本轮把它们逐份取回并用**远端 md5** 锁定字节，"
                                     "再由**列联计数**独立重算：① 评测器算术自洽（CSI/BIAS/POD/FAR 由 hits/misses/false_alarms 重算，"
                                     "21 格 × 4 阈值 **max|dev| = 0.000e+00**）；② 聚合产物与原始**逐位相同**（8 字段 × 21 格，max|dev| = 0.000e+00，"
                                     "21 格样本数一律 538）；③ HANDOVER §3 已公布的 seed-0 CSI@2.0 / BIAS@2.0 七臂**全部复现**"
                                     "（最大偏差 4.526e−07 / 4.519e−05 ＝ 小数位舍入）；④ 头条表从原始层**独立重算**与已发布一致"
                                     "（过地板 A3a 3/3、A3b 3/3、B0 2/3、A4b 2/3；累积快照检测器未触发）。**原始层新增读数**："
                                     "median CSI@2.0 / 中位 BIAS / 过地板 = B0 0.121817 / 0.509542 / 2/3；A1 0.118397 / 0.443021 / 2/3；"
                                     "A2 0.112003 / 0.449028 / 2/3；A3a 0.110015 / 0.639378 / 3/3；A3b 0.116055 / 0.618842 / 3/3；"
                                     "A4 0.143896 / 0.836171 / 2/3；A4b 0.137567 / 2.821505 / 2/3 ⇒ **差值口径 A4 +0.0221、A4b +0.0157 "
                                     "与第 44 轮独立算出的数逐位吻合**（本轮是在原始层复现它）；**超 Oracle 上界 0.1653 的种子只有 A4 与 A3b 各 1/3**"
                                     "（都是 seed 0）⇒ 「过预报换分」的结构第一次在原始层被看到，A4b 中位 BIAS 2.8215 ≈ 对照的 5.5 倍。"
                                     "**裁决零变化**（六臂仍全部判负、无一臂进十 seed），**MASTER 不升版**（仍 v2.49，无新数字入表）。\n")
        print(f"OK  handover: {act}")

    # 4) MEMORY.md 状态段：在「第 74 轮」那一条之后插入第 75 轮（锚点缺失即报错）
    if not MEMORY.exists():
        errs.append(f"missing {MEMORY}")
    else:
        txt = MEMORY.read_text(encoding="utf-8")
        if "- **E45 第 75 轮（2026-11-15）**" in txt:
            print("OK  memory.md: already present (skipped)")
        else:
            anchor = "- **E45 第 74 轮（2026-11-15）**"
            idx = txt.find(anchor)
            if idx < 0:
                errs.append("MEMORY.md: anchor for 第74轮 bullet NOT FOUND")
            else:
                end = txt.find("\n", idx)
                end = len(txt) if end < 0 else end + 1
                txt = txt[:end] + MEM_BULLET + txt[end:]
                MEMORY.write_text(txt, encoding="utf-8")
                print("OK  memory.md: inserted 第75轮 bullet")

    # 5) 工作区逐日日志
    if not WSLOG.exists():
        errs.append(f"missing {WSLOG}")
    else:
        txt = WSLOG.read_text(encoding="utf-8")
        if "E45 第 75 轮" in txt:
            print("OK  workspace log: already present (skipped)")
        else:
            WSLOG.write_text(txt.rstrip("\n") + "\n" + WS_NOTE, encoding="utf-8")
            print("OK  workspace log: appended")

    if errs:
        print("\nERRORS (anchors/files missing -> NOT silently skipped):")
        for e in errs:
            print("  -", e)
        return 1
    print("\nALL WRITEBACKS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
