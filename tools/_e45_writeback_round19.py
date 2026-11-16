#!/usr/bin/env python
"""Write round-19 verification record into the four memory locations.

Idempotent: if the round-19 marker is already present, skip that file.
Line endings of each target are preserved (detected from the file itself).
"""
import io
import os
import sys

MARK = "第 19 次独立复核"

MEM_LINE = (
    "**（2026-11-12 10:43 第 19 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、"
    "提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、"
    "远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**；"
    "新脚本 `tools/_e45_verify_round19.py` 不 import 旧脚本、从原始产物重算："
    "S3 六臂原始增益 CI 与配平后 CI **全跨零**（A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，"
    "A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）；"
    "3-seed 中位数/过地板通过率：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、"
    "A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ "
    "中位数高于对照者仅 A4/A4b 且 BIAS 同时爆表（0.8362/2.8215）、"
    "**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳；"
    "机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致（门 3/3 PASS；A3a/A3b 0/3 ⇒ 拓扑支结掉；"
    "A4 3/3、A4b 2/3；`maxratio_as_trained` 三 seed 全 `nan`）；"
    "写回脚本连跑 **0 改动**、两个校验器 `VERDICT: PASS` ⇒ 结论无误，"
    "**未改 docs、未下载、未 sbatch**。本轮新脚本新增三项一致性断言："
    "S3 各块的 `n_windows`/`n_seeds` 与 `struct_metrics_all.json` 的窗口数一致、"
    "`scores_summary` 的 `seeds` 列表与逐 seed 分数覆盖一致、中位数为排序后的中值。"
    "前十八轮复核结论完全一致，无新增偏离。）**"
)

BLOCK = """
---

## 2026-11-12 10:43（本次运行 · 第 19 次独立复核）

**判定：远端 E45 全部完成且与前十八轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 **273** `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、
  汇总 **291** `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**（squeue 无作业）；
  提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**：远端 `runs/e45/` 13 个 json 与本地 `tools/_evals/e45/` 13 个 json 逐一相同
  （本地目录 14 项 = 13 json + `_s2_scores_raw.txt`）⇒ 无需重新下载。
- **S3 独立重算**（新脚本 `tools/_e45_verify_round19.py`，不 import 旧脚本、直接从原始产物重取）：
  阈值 2.0 mm、epoch 00、control=B0、n_boot 10000；六臂**原始增益 CI 与配平后 CI 全部跨零**，
  12 组 `crosses_zero` 标志与重算逐项一致；
  A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`
  ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率**（地板 0.0924、Oracle 0.1653）：
  B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、
  A4 0.1439/2-3、A4b 0.1376/2-3（A4b 离散 9.23×）
  ⇒ 中位数高于对照的只有 A4/A4b，**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；
  **对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳，不构成晋级依据。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：
  门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），
  `mask_at_peak` 0.7224～1.0（全 >0.5）⇒ **两臂 3/3 PASS，门指向结构上已解决**；
  A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，
  `field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支结掉**；
  A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、
  `scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；
  `mean_out_equals_T` A4 [True,False,True]、A4b 三 seed 全 False ⇒ 仍属实现缺陷；
  **`maxratio_as_trained` 三 seed 全 `nan`（读数缺失）**。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（五个目标文件均 `unchanged`）、
  `_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**
  （v2.44/v2.45 行均 4 cells，与 §12 表头一致）；
  MASTER 头部 **v2.45**、§4.47 卡片、§9 E45 行、§12 v2.44/v2.45 两行、INDEX E45 行、
  上层 `MEMORY.md` 指针 v2.45 全部在位。
- 记忆：上层 `MEMORY.md` 状态段追加本轮复核注记；项目级与上层 `2026-11-12.md` 追加本轮复核段。

### 本轮新增可复用经验

- **复核脚本要连「跨产物的元数据一致性」一起断言**：本轮新增三条断言——S3 各统计块的 `n_windows`/`n_seeds`
  必须与 `struct_metrics_all.json` 的窗口数（538）和 3-seed 覆盖一致、`scores_summary` 的 `seeds`
  列表长度必须与逐 seed 分数覆盖一致、中位数必须是排序后的中间值（而非某种偶然的聚合）。
  这些断言能在「产物被截断/写半/错配」时立刻暴露，而不必等到数字对不上才发觉。
- **`scores_all.json` 与 `scores_summary_s2.json` 的 md5 完全相同**（累积语义：`_s2` 已含 s0+s1+s2）
  ⇒ 两者互校是天然的一致性检查点，不必重复下载。

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈
（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷
（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）
＋ 判据工具区分「臂不稳」与「地板口径不稳」。
"""

AUTO_BLOCK = """
---

## 2026-11-12 10:43（本次运行 · 第 19 次独立复核）

**判定：远端 E45 全部完成且与前十八轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、
  汇总 291 `COMPLETED`（00:14:23）、21 份评测齐、队列空；提交器日志尾
  `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- md5 13/13 全同：远端 `runs/e45/` 13 个 json 与本地 `tools/_evals/e45/` 逐一相同 ⇒ 无需重新下载。
- S3 独立重算（`tools/_e45_verify_round19.py`，不 import 旧脚本）：六臂原始增益 CI 与配平后 CI 全跨零；
  A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ 六臂判负成立。
- 3-seed 中位数/过地板通过率（地板 0.0924、Oracle 0.1653）：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、
  A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 高于对照者仅 A4/A4b 且 BIAS 爆表；
  对照 B0 自己只 2-3 过地板 ⇒ 地板口径不稳。
- 机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致；写回脚本 0 改动、两个校验器 VERDICT: PASS。
- 本轮新增三项跨产物元数据一致性断言（S3 n_windows/n_seeds、summary.seeds 覆盖、中位数排序校验）。
"""


def read_text(path):
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path, text):
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def append_block(path, block):
    if not os.path.exists(path):
        print("SKIP (missing): %s" % path)
        return
    txt = read_text(path)
    if MARK in txt:
        print("SKIP (already has marker): %s" % path)
        return
    eol = "\r\n" if "\r\n" in txt else "\n"
    blk = block.replace("\n", eol)
    if not txt.endswith(eol):
        txt += eol
    write_text(path, txt + blk)
    print("APPENDED: %s" % path)


def insert_memory_line(path):
    """Insert the round-19 note right after the round-18 note line."""
    if not os.path.exists(path):
        print("SKIP (missing): %s" % path)
        return
    txt = read_text(path)
    if MARK in txt:
        print("SKIP (already has marker): %s" % path)
        return
    eol = "\r\n" if "\r\n" in txt else "\n"
    lines = txt.split(eol)
    anchor = None
    for i, l in enumerate(lines):
        if "第 18 次独立复核" in l and l.lstrip().startswith("**（2026-11-12 09:29"):
            anchor = i
    if anchor is None:
        # fall back: last line that mentions the 18th round
        for i, l in enumerate(lines):
            if "第 18 次独立复核" in l:
                anchor = i
    if anchor is None:
        print("ERROR (anchor for round-18 note not found): %s" % path)
        sys.exit(2)
    lines.insert(anchor + 1, MEM_LINE)
    write_text(path, eol.join(lines))
    print("INSERTED after line %d: %s" % (anchor + 1, path))


if __name__ == "__main__":
    upper = "R:/1radarNWPprediction/.workbuddy/memory"
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    projmem = os.path.join(proj, ".workbuddy", "memory")
    auto = os.path.join(projmem, "automations",
                        "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

    append_block(os.path.join(projmem, "2026-11-12.md"), BLOCK)
    append_block(os.path.join(upper, "2026-11-12.md"), BLOCK)
    insert_memory_line(os.path.join(upper, "MEMORY.md"))
    append_block(auto, AUTO_BLOCK)
    print("DONE")
