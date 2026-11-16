#!/usr/bin/env python
"""Idempotent round-18 log writer for E45 monitoring.

Anchor rule learned in round 15: never regex-replace a reusable ordinal like
"第 N 次" (it collides with history). Instead anchor on the UNIQUE timestamped
heading / on the unique round-N bullet, and refuse to write twice.
"""
import io
import os
import sys

MARK = "第 18 次独立复核"
STAMP = "2026-11-12 09:29"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ_LOG = os.path.join(REPO, ".workbuddy", "memory", "2026-11-12.md")
UP_LOG = r"R:\1radarNWPprediction\.workbuddy\memory\2026-11-12.md"
MEM = r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md"
AUTO_MEM = os.path.join(REPO, ".workbuddy", "memory", "automations",
                        "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

SECTION = """
## {stamp}（本次运行 · {mark}）

**判定：远端 E45 全部完成且与前十七轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 **273** `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、汇总 **291** `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**（squeue 无作业）；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**：远端 `runs/e45/` 13 个 json 与本地 `tools/_evals/e45/` 13 个 json 逐一相同 ⇒ 无需重新下载。
- **S3 独立重算**（新脚本 `tools/_e45_verify_round18.py`，不 import 旧脚本、直接从原始产物重取）：阈值 2.0 mm、epoch 00、control=B0、n_boot 10000；六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率**（地板 0.0924、Oracle 0.1653）：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 中位数高于对照的只有 A4/A4b，**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳，不构成晋级依据。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），`mask_at_peak` 0.7224～1.0（全 >0.5）⇒ **两臂 3/3 PASS，门指向结构上已解决**；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支结掉**；A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`mean_out_equals_T` A4 [True,False,True]、A4b 三 seed 全 False ⇒ 仍属实现缺陷；**`maxratio_as_trained` 三 seed 全 `nan`（读数缺失）**。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（幂等，五个目标文件均 `unchanged`）、`_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**（v2.44/v2.45 行均 4 cells，与 §12 表头一致）；MASTER 头部 **v2.45**、§4.47、§9 E45 行、§12 v2.44/v2.45 两行、INDEX E45 行、`MEMORY.md` 指针 v2.45 全部在位。
- 记忆：上层 `MEMORY.md` 状态段追加本轮复核注记；项目级与上层 `2026-11-12.md` 追加本轮复核段。

### 本轮新增可复用经验（复核脚本自身的坑）

- **`scores_summary.json` 的 `seed_spread_csi_at_2.0` 是 max/min 比值，不是极差**：本轮首版按「max−min」重算 ⇒ 7 臂全部误报 FAIL（A4b 得 0.1248，产物为 **9.226194**＝0.1399/0.0152）。**比率型统计量必须先看数值量级再猜公式**（A4b 的 9.23× 是本项目「离散度」的既定口径）。
- **负向断言的参数极性要写对**：`neg_check(name, cond)` 的语义是「cond 为真才判失败」；首版写成 `not any(pass_list)` ⇒ 拓扑支明明三 seed 全不过，却反而被判成「应有的负结果没出现」而刷红。**负向断言的参数应该是「违规事件发生了吗」，而不是「违规事件没发生吗」。**
- **「逐 seed 期望模式」要按读数分别写**：`scale_cv` 在 A4b 三 seed 全过（2.11/1.89/3.02），只有 `scale_truth_corr` 才是 [过、过、不过]；首版把 [T,T,F] 模式套到 `scale_cv` 上 ⇒ 1 项误报 FAIL。**模式必须绑定到具体读数，不能按臂套用。**

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
"""

MEM_NOTE = ("**（2026-11-12 09:29 第 18 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、"
            "远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**；新脚本 `tools/_e45_verify_round18.py` "
            "不 import 旧脚本、从原始产物重算：S3 六臂原始增益 CI 与配平后 CI **全跨零**（A1/A3a/A3b/A4 = "
            "`GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）；3-seed 中位数/过地板通过率："
            "B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、"
            "A4b 0.1376/2-3 ⇒ 中位数高于对照者仅 A4/A4b 且 BIAS 同时爆表（0.8362/2.8215）、**对照 B0 自己只 "
            "2-3 过地板** ⇒ 地板口径不稳；机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致（门 3/3 PASS；"
            "A3a/A3b 0/3 ⇒ 拓扑支结掉；A4 3/3、A4b 2/3；`maxratio_as_trained` 三 seed 全 `nan`）；"
            "写回脚本连跑 **0 改动**、两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未下载、未 sbatch**。"
            "本轮新脚本自身修了三个断言口径 bug（`seed_spread` 是比值不是极差／负向断言参数极性写反／"
            "[T,T,F] 模式误套到 `scale_cv`），详见项目日志。前十七轮复核结论完全一致，无新增偏离。）**")


def append_once(path, text, marker):
    if not os.path.exists(path):
        print("MISSING: %s" % path)
        return False
    with io.open(path, "r", encoding="utf-8") as f:
        body = f.read()
    if marker in body:
        print("unchanged (marker present): %s" % path)
        return False
    body = body.rstrip("\n") + "\n" + text
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print("appended: %s" % path)
    return True


def insert_note_after(path, anchor_prefix, note):
    if not os.path.exists(path):
        print("MISSING: %s" % path)
        return False
    with io.open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if any(MARK in l for l in lines):
        print("unchanged (marker present): %s" % path)
        return False
    idx = None
    for i, l in enumerate(lines):
        if l.startswith(anchor_prefix):
            idx = i
    if idx is None:
        print("ERROR: anchor not found in %s -> refuse to write silently" % path)
        return False
    lines.insert(idx + 1, note)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print("inserted note: %s" % path)
    return True


changed = 0
changed += 1 if append_once(PROJ_LOG, SECTION.format(stamp=STAMP, mark=MARK), MARK) else 0
changed += 1 if append_once(UP_LOG, SECTION.format(stamp=STAMP, mark=MARK), MARK) else 0
changed += 1 if insert_note_after(MEM, "**（2026-11-12 08:15 第 17 次独立复核", MEM_NOTE) else 0
changed += 1 if append_once(AUTO_MEM, SECTION.format(stamp=STAMP, mark=MARK), MARK) else 0

print("\nchanged files: %d" % changed)
