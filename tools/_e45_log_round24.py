# -*- coding: utf-8 -*-
"""把 E45 第 24 次独立复核的记录写入四处记忆（幂等：已有同轮次记录则跳过）。

用法：python tools/_e45_log_round24.py
"""
import io
import os

TS = "2026-11-12 16:15"
MARK = "第 24 次独立复核"

AUTO = r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\automations\e3cc6636-3e60-4a58-8f7e-67d4c896b0b6\memory.md"
PROJ = r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\2026-11-12.md"
UP_DAILY = r"R:\1radarNWPprediction\.workbuddy\memory\2026-11-12.md"
UP_MEM = r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md"

auto_note = """
## %s（本次运行 · 第 24 次独立复核）

**判定：远端 E45 全部完成且与前二十三轮完全一致；裁决与写回均无需改动；本轮未下载、未 sbatch、未改 docs（仅新增复核脚本 `tools/_e45_verify_round24.py`）。**

- 远端：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、汇总 291 `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**（`tools/_e45_md5_check.py` 实时抓取比对）⇒ 无需重新下载。
- **S3 独立重算**（`tools/_e45_verify_round24.py`，不 import 旧脚本、直接从原始产物重取）：六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ **六臂判负、不投十 seed 算力**成立。**VERDICT: PASS（POS 92/92、NEG 18/18、PAT 9/9）**。
- **3-seed 中位数 / 过地板通过率 / 超 Oracle**：B0 0.1218/2-3/0-3、A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、A3a 0.1100/3-3/0-3、A3b 0.1161/3-3/1-3、A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3 ⇒ 中位数高于对照者仅 A4/A4b、**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳，不构成晋级依据。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），`mask_at_peak` 0.72～1.0（全 >0.5）⇒ **两臂 3/3 PASS，门指向结构上已解决**；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支结掉**；A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`mean_out_equals_T` A4 [True,False,True]、A4b 三 seed 全 False ⇒ 仍属实现缺陷；**`maxratio_as_trained` 三 seed 全 `nan`（读数缺失）**。
- **本轮新增两项此前未做的检查（均 PASS）**：① **文档口径漂移**——地板 0.0924 / Oracle 0.1653 / 场最大值 402.43 在 E45 语境（MASTER 头部版本行、§4.47 卡片、§9 E45 行、§12 v2.44/v2.45 行；INDEX E45 行；E45 判据文件；HANDOVER）内不得出现异值；② **交接文档时效性**——`HANDOVER_E45.md` §0 现状段须仍写「已全部跑完」、§4「真正的下一步」段不得残留待跑指引。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（五个目标文件均 `unchanged`）、`_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**（v2.44/v2.45 均 4 cells）；MASTER 头部 **v2.45**、§4.47 卡片、§9 E45 行、§12 v2.44+v2.45 两行、INDEX E45 行全部在位。
- **本轮修复新脚本自身 4 处口径 bug**（均为脚本错、项目结论无误）：S3 裁决键是 `s3_verdict`（非 `verdict`）；`scores_summary.json` **顶层即臂键**（不必再找嵌套臂表）；**拓扑支是预注册负结果，必须走 NEG 通道**（写成 POS 会把「预期不过」刷红）；**漂移扫描必须限定 E45 语境**（首版扫 MASTER 全文，把 E17/§4.21 等历史数值误报为漂移）。

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
""" % TS

proj_note = """
---

## %s（本次运行 · 第 24 次独立复核 · 工作区级）

**判定：远端 E45 全部完成且与前二十三轮完全一致；裁决与写回均无需改动；本轮未下载、未 sbatch、未改 docs（仅新增复核脚本 `tools/_e45_verify_round24.py`）。**

- 远端：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、汇总 291 `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**（`tools/_e45_md5_check.py`）⇒ 无需重新下载。
- **S3 独立重算**（`tools/_e45_verify_round24.py`，不 import 旧脚本）：六臂原始增益 CI 与配平后 CI **全跨零**；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ 六臂判负成立。**VERDICT: PASS（POS 92/92、NEG 18/18、PAT 9/9）**。
- 3-seed 中位数/过地板通过率/超 Oracle：B0 0.1218/2-3/0-3、A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、A3a 0.1100/3-3/0-3、A3b 0.1161/3-3/1-3、A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3 ⇒ 高于对照者仅 A4/A4b 且 BIAS 爆表；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳。
- 机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致（门 3/3 PASS；A3a/A3b 0/3 ⇒ 拓扑支结掉；A4 3/3、A4b 2/3；`maxratio_as_trained` 三 seed 全 `nan`）。
- **新增两项检查（PASS）**：文档口径漂移（地板/Oracle/场最大值在 E45 语境内无异值）、交接文档时效性（§0 写「已全部跑完」、§4「真正的下一步」无待跑指引）。
- 写回复核：`_sync_v245.py` **0 改动**、`_check_tables.py` 与 `_check_v245_row.py` 均 **VERDICT: PASS**。
- **修新脚本自身 4 处口径 bug**：S3 裁决键 `s3_verdict`；`scores_summary.json` 顶层即臂键；拓扑支属预注册负结果须走 NEG 通道；漂移扫描须限定 E45 语境。

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
""" % TS

mem_note = (
    "**（%s 第 24 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、"
    "提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、"
    "远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**；新脚本 `tools/_e45_verify_round24.py` "
    "不 import 旧脚本、从原始产物重算：**POS 92/92、NEG 18/18、PAT 9/9、VERDICT: PASS**；"
    "S3 六臂原始增益 CI 与配平后 CI **全跨零**（A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，"
    "A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）；3-seed 中位数/过地板通过率/超 Oracle：B0 0.1218/2-3/0-3、"
    "A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、A3a 0.1100/3-3/0-3、A3b 0.1161/3-3/1-3、"
    "A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3 ⇒ 中位数高于对照者仅 A4/A4b 且 BIAS 爆表、"
    "**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳；机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致"
    "（门 3/3 PASS；A3a/A3b 0/3 ⇒ 拓扑支结掉；A4 3/3、A4b 2/3；`maxratio_as_trained` 三 seed 全 `nan`）；"
    "写回脚本连跑 **0 改动**、两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未下载、未 sbatch**。"
    "本轮新增两项检查：文档口径漂移（三个关键常量在 E45 语境内无异值）与交接文档时效性"
    "（现状段/「真正的下一步」段）。前二十三轮复核结论完全一致，无新增偏离。）**") % TS


def append(path, text):
    if not os.path.exists(path):
        print("SKIP (missing): %s" % path)
        return
    with io.open(path, "r", encoding="utf-8") as f:
        cur = f.read()
    if MARK in cur:
        print("IDEMPOTENT (already present): %s" % path)
        return
    with io.open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("APPENDED: %s" % path)


def insert_after(path, anchor, text):
    if not os.path.exists(path):
        print("SKIP (missing): %s" % path)
        return
    with io.open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if any(MARK in l for l in lines):
        print("IDEMPOTENT (already present): %s" % path)
        return
    idx = None
    for i, l in enumerate(lines):
        if anchor in l:
            idx = i
    if idx is None:
        print("ANCHOR NOT FOUND in %s -> %s" % (path, anchor))
        return
    lines.insert(idx + 1, text)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print("INSERTED after line %d: %s" % (idx + 1, path))


append(AUTO, auto_note)
append(PROJ, proj_note)
append(UP_DAILY, proj_note)
insert_after(UP_MEM, "15:01 第 23 次独立复核", mem_note)
