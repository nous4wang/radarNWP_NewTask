#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 round-23: append review records to memory files + fix MEMORY.md round-label drift.

Anchor rule (from round-15 lesson): round numbers must be derived from the
automation log's timestamped headings, never from a regex on "第 N 次".
"""
import io
import os
import re

UP = "R:/1radarNWPprediction/.workbuddy/memory"
PROJ = "R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory"
AUTO = os.path.join(PROJ, "automations", "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

ROUND = 23
TS = "2026-11-12 15:0"  # 15:01-15:0x local, session started 15:01:54

BLOCK = """
## {ts}（本次运行 · 第 {r} 次独立复核）

**判定：远端 E45 全部完成且与前二十二轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283（各 7 子任务全 `COMPLETED`）、评测 274/282/290 全 `COMPLETED`、汇总 291 `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**（`tools/_e45_md5_check.py` 实拉远端 `runs/e45/*.json` 的 md5，与本地 `tools/_evals/e45/` 逐一比对）⇒ 无需重新下载。
- **S3 独立重算**（新脚本 `tools/_e45_verify_round23.py`，不 import 旧脚本、直接从原始产物重取）：阈值 2.0 mm、epoch 00、control=B0、n_boot 10000、538 窗；六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率 / 超 Oracle**（地板 0.0924、Oracle 0.1653，由 `scores_all.json` 重算并与 `scores_summary.json` **逐记录互校**）：B0 0.1218/2-3/0-3、A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、A3a 0.1100/3-3/0-3、A3b 0.1161/3-3/1-3、A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3（A4b 离散 9.23×）⇒ 中位数高于对照者仅 A4/A4b，**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳，不构成晋级依据。晋级合取判定：合格臂集合为空。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），`mask_at_peak` 0.7224～1.0（全 >0.5）⇒ **两臂 3/3 PASS，门指向结构上已解决**；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支结掉**；A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ `scale_is_constant` = True，退化成常数）；`mean_out_equals_T` A4 [True,False,True]、A4b 三 seed 全 False ⇒ 仍属实现缺陷；**`maxratio_as_trained` 三 seed 全 `nan`（读数缺失）**。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（幂等，五个目标文件均 `unchanged`）、`_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**（v2.44/v2.45 均 4 cells，与 §12 表头一致）；MASTER 头部 **v2.45**、§4.47 卡片、§9 E45 行、§12 v2.44/v2.45 两行、INDEX E45 行、上层 `MEMORY.md` 指针 v2.45 全部在位。
- **VERDICT: PASS**（POS 151/151、NEG 15/15、PAT 16/16）。

### 本轮新增可复用经验（复核脚本自身的坑）

- **「爆表」没有数值定义，别临时发明阈值**：本轮首版把「BIAS 爆表」写成「> 2 × 对照中位数」⇒ A4（中位 0.8362 vs 对照 0.5095，倍数只有 1.64）被误判失败。项目预注册口径是 E45 判据文件 L463 的「**A4/A4b 的 `BIAS@2.0` < 0.9** ⇒ 该支就此结掉」⇒ A4 恰恰**必须** < 0.9 才符合「结掉」的预注册条件。**判据阈值只能从冻结的判据文件取，不能从「爆表」这种修辞反推倍数。**
- **NEG 通道的参数是「违规事件是否发生」，不是「违规事件是否没发生」**：本轮把拓扑支的负向断言写成 `any(share>=0.5) or any(fmax>30)` ⇒ 该式在「确实仍失败」时恒为真 ⇒ 反而被判成「负结果消失」。正确写法：`violation = any(该 seed 同时满足 share<0.5 且 fmax<=30)`（即「某个 seed 意外通过了判据」才算违规）。
- **文档定位别猜行号**：MASTER 的版本行不在第 4 行而在第 11 行（`| 最后更新 | …` 行）；§12 变更日志行以日期开头（`| 2026-11-11 | **v2.44** |`），只匹配 `^\\|\\s*v2\\.4[45]` 会漏判两条记录都存在。**定位应用内容锚点（如「最后更新」「日期 + 版本」），不用行号。**
- **禁写 lint 的禁令标记不止「禁写/不得写/不能写/禁止」**：MASTER L1536 的「LABEL 有孤立针状尖峰」是 §10.2 的**禁令引用**（上下文写「新增两条不可再写」），本轮首版漏了「不可再写」⇒ 误报违规。**lint 的禁令正则要按本项目实际用词枚举，窗口也要放宽到前后各 3 行。**
- **脚本自身 bug 与项目 bug 必须分开记账**：本轮 4 项 FAIL 全部是复核脚本的口径错误（`rall`/`sall` 拼写、INDEX 路径 `docs/experiments/INDEX.md` 而非 `docs/INDEX.md`、BIAS 阈值、NEG 极性），**项目文档与结论本身无误**；修脚本后 POS/NEG/PAT 全绿。**先判「是脚本错还是产物错」，再动手改 docs。**

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


block = BLOCK.replace("{ts}", TS).replace("{r}", str(ROUND))

# ---- 1. daily logs (append-only) -------------------------------------------
for d in (UP, PROJ):
    p = os.path.join(d, "2026-11-12.md")
    txt = read(p) if os.path.exists(p) else ""
    if "第 %d 次独立复核" % ROUND in txt:
        print("skip (already present):", p)
        continue
    if not txt.endswith("\n"):
        txt += "\n"
    write(p, txt + block)
    print("appended:", p)

# ---- 2. upstream MEMORY.md status section ----------------------------------
mp = os.path.join(UP, "MEMORY.md")
mem = read(mp)
lines = mem.split("\n")

# fix the round-label drift: 12:52 is round 21, not 22 (per automation log headings)
fixed = 0
for i, l in enumerate(lines):
    if "（2026-11-12 12:52 第 22 次独立复核" in l:
        lines[i] = l.replace("（2026-11-12 12:52 第 22 次独立复核",
                             "（2026-11-12 12:52 第 21 次独立复核")
        fixed += 1
print("MEMORY.md label fixes:", fixed)

# the 13:58 (round 22) note is missing from MEMORY.md -> insert a concise one
if "13:58 第 22 次独立复核" not in "\n".join(lines):
    note22 = ("**（2026-11-12 13:58 第 22 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、"
              "远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**；新脚本 `tools/_e45_verify_round22.py` "
              "不 import 旧脚本、从原始产物重算：S3 六臂原始增益 CI 与配平后 CI **全跨零**（A1/A3a/A3b/A4 = "
              "`GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）；3-seed 中位数/过地板通过率/超 Oracle："
              "B0 0.1218/2-3/0-3、A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、A3a 0.1100/3-3/0-3、A3b 0.1161/3-3/1-3、"
              "A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3 ⇒ 中位数高于对照者仅 A4/A4b 且 BIAS 同时爆表、"
              "**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳；机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致"
              "（门 3/3 PASS；A3a/A3b 0/3 ⇒ 拓扑支结掉；A4 3/3、A4b 2/3；`maxratio_as_trained` 三 seed 全 `nan`）；"
              "写回脚本连跑 **0 改动**、两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未下载、未 sbatch**。"
              "本轮新增三项检查：远端评测清单 ↔ 本地聚合产物覆盖比对（21/21 双向差集为空）、晋级资格合取判定"
              "（合格臂集合为空）、文档引用数字一致性与阶梯规则自洽检查。前二十一轮复核结论完全一致，无新增偏离。）**")
    # insert right after the line that holds the 12:52 note
    for i, l in enumerate(lines):
        if "（2026-11-12 12:52 第 21 次独立复核" in l:
            lines.insert(i + 1, note22)
            print("MEMORY.md: inserted missing round-22 note")
            break
    else:
        print("WARN: anchor for round-22 note not found")

note23 = ("**（2026-11-12 15:01 第 23 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、"
          "提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、远端 `runs/e45/` 13 个 json "
          "与本地 md5 **13/13 全同**；新脚本 `tools/_e45_verify_round23.py` 不 import 旧脚本、从原始产物重算："
          "**POS 151/151、NEG 15/15、PAT 16/16、VERDICT: PASS**；S3 六臂原始增益 CI 与配平后 CI **全跨零**"
          "（A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）；"
          "3-seed 中位数/过地板通过率/超 Oracle：B0 0.1218/2-3/0-3、A1 0.1184/2-3/0-3、A2 0.1120/2-3/0-3、"
          "A3a 0.1100/3-3/0-3、A3b 0.1161/3-3/1-3、A4 0.1439/2-3/1-3、A4b 0.1376/2-3/0-3 ⇒ 中位数高于对照者仅 "
          "A4/A4b 且 BIAS 同时爆表（0.8362/2.8215）、**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳；"
          "机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致（门 3/3 PASS；A3a/A3b 0/3 ⇒ 拓扑支结掉；"
          "A4 3/3、A4b 2/3；`maxratio_as_trained` 三 seed 全 `nan`）；写回脚本连跑 **0 改动**、"
          "两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未下载、未 sbatch**。"
          "本轮唯一实质改动＝修复核脚本自身 4 处口径 bug（BIAS 阈值须取判据文件的 <0.9、NEG 通道极性、"
          "MASTER 版本行/变更日志定位、禁写 lint 的禁令标记集合），并订正本文件一处轮次标注错位"
          "（12:52 实为第 21 次）+ 补录缺失的第 22 次注记。前二十二轮复核结论完全一致，无新增偏离。）**")

if "15:01 第 23 次独立复核" not in "\n".join(lines):
    # append at the end of the E45 status bullet block: after the last line that
    # contains a round note
    last = -1
    for i, l in enumerate(lines):
        if "次独立复核" in l and l.startswith("**（2026-11-12"):
            last = i
    if last < 0:
        print("ERROR: no round-note anchor in MEMORY.md")
        raise SystemExit(1)
    lines.insert(last + 1, note23)
    print("MEMORY.md: appended round-23 note")

write(mp, "\n".join(lines))
print("written:", mp)

# ---- 3. automation memory (summary only) ----------------------------------
ap = AUTO
txt = read(ap)
if "第 %d 次独立复核" % ROUND in txt:
    print("skip automation memory (already present)")
else:
    if not txt.endswith("\n"):
        txt += "\n"
    txt += block.replace("### 本轮新增可复用经验（复核脚本自身的坑）",
                         "### 本轮新增可复用经验（复核脚本自身的坑，本轮记录于自动化记忆）")
    write(ap, txt)
    print("appended:", ap)

print("DONE")
