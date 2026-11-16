# -*- coding: utf-8 -*-
"""Append the round-21 E45 re-verification record to the memory files.

Idempotent-ish: refuses to write twice (assertion on the round marker).
All replacements use unique anchors; a missing anchor raises instead of
silently skipping.
"""
import io

TS = "2026-11-12 12:52"
ROUND = 21

UPPER_MEM = ".workbuddy/memory/MEMORY.md"
PROJ_LOG = "radar-nwp-bench/.workbuddy/memory/2026-11-12.md"
UPPER_LOG = ".workbuddy/memory/2026-11-12.md"
AUTO_MEM = ("radar-nwp-bench/.workbuddy/memory/automations/"
            "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

# ---------------------------------------------------------------- 1) upper MEMORY.md
note = (
    " **（%s 第 %d 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、"
    "提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、"
    "远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**；新脚本 "
    "`tools/_e45_verify_round21.py` 不 import 旧脚本、从原始产物重算："
    "**POS 182/0、NEG 16/0、VERDICT: PASS**；"
    "S3 六臂原始增益 CI 与配平后 CI **全跨零**（A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，"
    "A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）；3-seed 中位数/过地板通过率/超 Oracle 次数："
    "B0 0.1218/2-3/0-3、A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、A3a 0.1100/3-3/0-3、"
    "A3b 0.1161/3-3/1-3、A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3 ⇒ 中位数高于对照者仅 A4/A4b 且 BIAS 同时爆表"
    "（0.8362/2.8215）、**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳；机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致"
    "（门 3/3 PASS；A3a/A3b 0/3 ⇒ 拓扑支结掉；A4 3/3、A4b 2/3；`maxratio_as_trained` 三 seed 全 `nan`）；"
    "写回脚本连跑 **0 改动**、两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未下载、未 sbatch**。"
    "本轮新增三项检查：① **远端评测清单 ↔ 本地聚合产物覆盖比对**"
    "（21 份 `*.eval.json` 文件名解析出的 (臂,seed) 集合与 `scores_all.json` 的 21 条记录**完全相同** "
    "⇒ 聚合无静默丢臂）；② **晋级资格合取判定**"
    "（S3 配平后不跨零 AND 3/3 过地板 AND 臂级读数全过 ⇒ **合格臂集合为空**，并逐臂给出阻断原因）；"
    "③ **文档引用数字与产物一致性**＋**阶梯规则与拒绝记录并存**（规则仍在、拒绝也在 ⇒ 无自相矛盾）。"
    "前二十轮复核结论完全一致，无新增偏离。）**") % (TS, ROUND)

with io.open(UPPER_MEM, "r", encoding="utf-8") as f:
    txt = f.read()
anchor = "前十九轮复核结论完全一致，无新增偏离。）**"
if ("第 %d 次独立复核" % ROUND) in txt:
    print("upper MEMORY.md: round-%d note already present -> skip" % ROUND)
else:
    assert txt.count(anchor) == 1, "anchor count=%d (expected exactly 1)" % txt.count(anchor)
    txt = txt.replace(anchor, anchor + note)
    with io.open(UPPER_MEM, "w", encoding="utf-8") as f:
        f.write(txt)
    print("upper MEMORY.md: appended round-%d note" % ROUND)

# ---------------------------------------------------------------- 2) daily logs
para = """
---

## __TS__（本次运行 · 第 __ROUND__ 次独立复核）

**判定：远端 E45 全部完成且与前二十轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 **273** `COMPLETED`（05:04:04）、训练数组 **275/283**（各 7 子任务全 `COMPLETED`）、评测 **274/282/290** 全 `COMPLETED`、汇总 **291** `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 实时比对 13/13 全同**（远端 `runs/e45/` 13 个 json vs 本地 `tools/_evals/e45/`，脚本内直接吃 `remote.py` stdout，不落临时文件）⇒ 无需重新下载。
- **S3 独立重算**（新脚本 `tools/_e45_verify_round21.py`，不 import 旧脚本、直接从原始产物重取）：阈值 2.0 mm、epoch 00、control=B0、n_boot 10000；六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率 / 超 Oracle 次数**（地板 0.0924、Oracle 0.1653，由 `scores_all.json` 重算并与 `scores_summary.json` 互校）：B0 0.1218/2-3/0-3、A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、A3a 0.1100/3-3/0-3、A3b 0.1161/3-3/1-3、A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3（A4b 离散 9.2262×）⇒ 中位数高于对照的只有 A4/A4b，**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳，不构成晋级依据。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），`mask_at_peak` 0.7224～1.0（全 >0.5）⇒ **两臂 3/3 PASS，门指向结构上已解决**；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支结掉**；A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`mean_out_equals_T` A4 [True,False,True]、A4b 三 seed 全 False ⇒ 仍属实现缺陷；**`maxratio_as_trained` 三 seed 全 `nan`（读数缺失）**。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（幂等，五个目标文件均 `unchanged`）、`_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**（v2.44/v2.45 行均 4 cells，与 §12 表头一致）；MASTER 头部 **v2.45**、§4.47 卡片、§9 E45 行、§12 v2.44/v2.45 两行、INDEX E45 行、上层 `MEMORY.md` 指针 v2.45 全部在位。

### 本轮新增三项检查（此前各轮未做）

1. **远端评测清单 ↔ 本地聚合产物覆盖比对**：远端 21 份 `runs/e45/evals/*.eval.json` 的文件名解析成 (臂, seed) 集合，与本地 `scores_all.json` 的 21 条记录**完全相同**（两向差集均为空）⇒ 聚合产物没有静默丢掉任何臂-种子。
2. **晋级资格合取判定**（阶梯规则的逻辑核心）：`合格臂 = (S3 配平后 CI 不跨零) AND (3/3 过地板) AND (臂级读数全过)`。实测**合格臂集合为空**，逐臂阻断原因：A1/A2/A4 = S3 跨零 + 过地板 2/3；A3a/A3b = S3 跨零 + 臂级读数不过（`field_max_mm` 402.43 > 30）；A4b = 三项全中。**A3a/A3b 虽 3/3 过地板但中位数低于对照 ⇒ 仍不晋级**，与 MASTER v2.45 的「无一臂进阶十 seed」一致。
3. **文档引用数字一致性 + 阶梯规则与拒绝记录并存**：文档引用值 402.43 / 0.0924 / 0.2789 / 0.858320 与产物重算值一致；判据文档**仍保留晋级规则原文**（阶梯阶 2、不得写 SOTA）**同时**记录了「没有任何一条臂够格进入阶 2」与「不投十 seed 算力」⇒ 规则在、拒绝也在，无自相矛盾。

### 本轮新增可复用经验（复核脚本自身的坑）

- **`tr -d '[:space:]'` 不能用在多行 `sacct` 输出上**：首版为绕开「状态列带空格」而对整段输出做 `tr -d`，结果 136 行被粘成 **1 行**，7 条作业状态断言全部误报 FAIL（273/291/275/283 明明都是 `COMPLETED`）。正确做法：`sacct -n -P --format=JobID,JobName,State,Elapsed` 用 **管道符分列**，逐行 split 即可，**根本不需要 `tr`**（`tr` 只适合单行字段内去空格）。
- **squeue 的 `%.2t` 给的是缩写状态**（`R`/`PD`/`CG`），拿它和 `RUNNING`/`PENDING` 全称比对会让「队列为空」的断言**空洞通过**（永远匹配不到 ⇒ 恒为 0 条 ⇒ 恒 PASS）。要用 `-o "%%T"` 全称字段。
- **队列为空的断言必须同时断言「解析器确实跑过」**（例如 `sacct` 返回行数 > 0、`squeue` 行数与解析结果行数相等），否则「一个作业都没查到」和「解析失败」不可区分。

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
"""
para = para.replace("__TS__", TS).replace("__ROUND__", str(ROUND))

for path in (PROJ_LOG, UPPER_LOG):
    with io.open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if ("第 %d 次独立复核" % ROUND) in txt:
        print("round-%d log already present in %s -> skip" % (ROUND, path))
        continue
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(para)
    print("appended round-%d log to %s" % (ROUND, path))

# ---------------------------------------------------------------- 3) automation memory
auto_note = """
---

## __TS__（本次运行 · 第 __ROUND__ 次独立复核）

**判定：远端 E45 全部完成且与前二十轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283（各 7 子任务全 `COMPLETED`）、评测 274/282/290 全 `COMPLETED`、汇总 291 `COMPLETED`（00:14:23）、21 份评测齐、队列空；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- md5 13/13 全同（远端 `runs/e45/` 13 个 json vs 本地 `tools/_evals/e45/`）⇒ 无需重新下载。
- S3 独立重算（新脚本 `tools/_e45_verify_round21.py`，不 import 旧脚本）：六臂原始增益 CI 与配平后 CI 全跨零；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ 六臂判负成立。**POS 182/0、NEG 16/0、VERDICT: PASS**。
- 3-seed 中位数/过地板通过率（地板 0.0924、Oracle 0.1653）：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 高于对照者仅 A4/A4b 且 BIAS 爆表；对照 B0 自己只 2-3 过地板 ⇒ 地板口径不稳。
- 机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致；写回脚本 0 改动、两个校验器 `VERDICT: PASS`。
- 本轮新增：① 远端评测清单 ↔ 本地聚合产物覆盖比对（21/21 双向差集为空）；② 晋级资格合取判定（合格臂集合为空，逐臂给出阻断原因）；③ 文档引用数字一致性与「规则仍在＋拒绝也在」的阶梯自洽检查。
- 本轮新增可复用经验：`tr -d '[:space:]'` 不能用于多行 `sacct` 输出（会把 136 行粘成 1 行并让状态断言全误报），应改用 `sacct -n -P --format=...,...` 管道分列；squeue 的 `%.2t` 是缩写状态（`R`/`PD`），与全称比对会让「队列为空」断言空洞通过，应改用 `-o "%%T"`；队列类断言须同时断言解析器确实跑过。
"""
auto_note = auto_note.replace("__TS__", TS).replace("__ROUND__", str(ROUND))

with io.open(AUTO_MEM, "r", encoding="utf-8") as f:
    txt = f.read()
if ("第 %d 次独立复核" % ROUND) in txt:
    print("automation memory: round-%d already present -> skip" % ROUND)
else:
    with io.open(AUTO_MEM, "a", encoding="utf-8") as f:
        f.write(auto_note)
    print("appended automation memory summary")
