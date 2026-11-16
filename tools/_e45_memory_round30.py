#!/usr/bin/env python
"""Append the round-30 E45 verification note to the four memory locations.

Kept as a script (not an inline heredoc) because the note text contains shell
-hostile characters when piped through bash.
"""
import io
import os

MEM_ADD = """**（2026-11-12 22:47 第 30 次独立复核：远端 273/291 全 `COMPLETED`、队列空、21 份评测齐、提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**（本轮并首次把 21 份 `*.eval.json` 的远端 md5 一并取回比对，21/21 全同）；核心闸门 `tools/_e45_verify_round27.py` **POS 141/141、NEG 6/6、PAT 11/11、VERDICT: PASS**、`_e45_verify_round28.py` **POS 40/40、NEG 3/3、VERDICT: PASS**、`_e45_verify_round29.py` **POS 74/74、NEG 21/21、VERDICT: PASS**；S3 六臂原始增益 CI 与配平后 CI **全跨零**（A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）⇒ 六臂判负、不投十 seed 算力成立；3-seed 中位数/过地板通过率与 MASTER v2.45 逐项一致（B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3）⇒ 合格臂集合为空；写回脚本连跑 **0 改动**、两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未 sbatch**。本轮新增一项此前未做的检查（`tools/_e45_verify_round30.py`，**POS 61/61、NEG 0/4、VERDICT: PASS**）：**把 21 份远端原始评测文件的列联表一路重算到本地聚合产物**——远端只读探针 `tools/_e45_eval_probe.py` 转储 H/M/FA → 按 `radar_nwp_bench/metrics/precipitation.py` 的权威公式重算 **csi/bias/pod/far 336/336 逐位精确**；**ETS 与 HSS 需 `correct_negatives`（评测 JSON 未存）⇒ 由 ETS 反解 CN 后回代 HSS，84 个阈值块零不一致、反解 CN 全部非负、隐含像素总数按 538 窗口归一后 spread=0.0（唯一值）⇒ ETS 与 HSS 互相自洽且物理量合法**；重算值 ↔ `scores_summary.json`（逐 seed csi@2.0/4.0/0.5、bias@2.0、中位数、过地板通过率、seed 极差）与 ↔ `scores_all.json` 零不一致；HANDOVER seed-0 七臂锚点逐位重现。前二十九轮复核结论完全一致，无新增偏离。）**
"""

LOG = """
### 2026-11-12 22:47（第 30 次独立复核）

**判定：远端 E45 全部完成且与前二十九轮完全一致；裁决与写回均无需改动；本轮无远端改动（未 sbatch、未改 docs），新增只读探针 `tools/_e45_eval_probe.py` 与复核脚本 `tools/_e45_verify_round30.py`。**

- 远端：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283（各 7 子任务全 `COMPLETED`）、评测 274/282/290 全 `COMPLETED`、汇总 291 `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- md5：远端 `runs/e45/` 13 个 json 与本地 **13/13 全同**；本轮新增取回 21 份 `*.eval.json` 的远端 md5，**21/21 全同** ⇒ 原始评测文件未被改动。
- 核心闸门：`tools/_e45_verify_round27.py` **POS 141/141、NEG 6/6、PAT 11/11、VERDICT: PASS**；`_e45_verify_round28.py` **POS 40/40、NEG 3/3、VERDICT: PASS**；`_e45_verify_round29.py` **POS 74/74、NEG 21/21、VERDICT: PASS**。S3 六臂原始增益与配平后增益 CI **全跨零**（A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH`）⇒ 六臂判负、不投十 seed 算力成立；合格臂集合为空。
- **本轮新增此前未做过的检查**（`tools/_e45_verify_round30.py`，**POS 61/61、NEG 0/4、VERDICT: PASS**）：**21 份远端原始评测文件 → 本地聚合产物的端到端列联表重算链**。
  - 远端只读探针 `tools/_e45_eval_probe.py` 转储 21 个 `*.eval.json` 的 H/M/FA 与 md5（**不写远端产物，只输出 stdout**）。
  - 按权威公式（`radar_nwp_bench/metrics/precipitation.py` 的 `VerificationCounts`）重算 **csi / bias / pod / far = 336/336 逐位精确**。
  - **ETS 与 HSS 需要 `correct_negatives`（评测 JSON 未存）⇒ 由 ETS 反解 CN 再回代 HSS**：84 个阈值块 **零不一致**、反解 CN **全部非负**、隐含像素总数按 538 窗口归一后 **spread = 0.0（唯一值）** ⇒ ETS 与 HSS 互相自洽且物理量合法，这是此前 29 轮都没做过的一条独立证据链。
  - 重算值 ↔ `scores_summary.json`（逐 seed csi@2.0/4.0/0.5、bias@2.0、中位数、过地板通过率、seed 极差）**零不一致**；↔ `scores_all.json`（21 条记录、csi/bias/n_samples）**零不一致**；HANDOVER seed-0 七臂锚点**逐位重现**。
  - 3-seed 中位数/过地板通过率（地板 0.0924、Oracle 0.1653）：B0 0.121817/2-3、A1 0.118397/2-3、A2 0.112003/2-3、A3a 0.110015/3-3、A3b 0.116055/3-3、A4 0.143896/2-3、A4b 0.137567/2-3 ⇒ 中位数高于对照者仅 A4/A4b 且 BIAS 同时爆表（0.8362/2.8215）；对照 B0 自己只 2-3 过地板 ⇒ 地板口径不稳；**合格臂集合为空**（与 S3 全部 CI 跨零一致）。
- 写回复核：`_sync_v245.py` 连跑 **0 改动**（五个目标文件均 `unchanged`）、`_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**（v2.44/v2.45 均 4 cells）。

### 本轮新增可复用经验（复核脚本自身的坑）

- **指标能否从评测 JSON 反算，取决于该指标是否只依赖 H/M/FA**：`csi/bias/pod/far` 可以；**`ets`/`hss` 需要 `correct_negatives`（评测 JSON 未存），`fss` 是空间邻域量（需原始场）**。首版脚本按「全部 7 个指标都可反算」来写 ⇒ 报出 252 处假 FAIL（21 文件 × 4 阈值 × 3 指标）。正确做法：**先读 `radar_nwp_bench/metrics/precipitation.py` 的权威实现**，再决定哪些指标可反算。
- **反解 `correct_negatives` 能把两个「不可反算」的指标变成互相校验**：`chance = (H+M)(H+FA)/T`、`ets = (H-chance)/(D-chance)` ⇒ `T = (H+M)(H+FA)/chance`、`CN = T - D`；再把 CN 回代 HSS 公式。这样 ETS 与 HSS 必须指向同一个 CN，**任何一方被篡改都会立刻暴露**，且 CN ≥ 0 与「隐含像素总数按 538 窗口归一后为唯一值」构成额外的物理一致性检查。
- **先判「脚本错还是产物错」**：本轮 4 项 FAIL 全部是脚本自身口径错（2 处键名/阈值键位 + 2 处指标可反算性误判），**项目产物与结论本身无误**，修脚本即可，**未改任何 docs**。
- **读回远端 md5 时顺带把 21 份原始评测文件的 md5 也取回**，等于给「原始输入未被改动」加了一层此前缺失的证据。

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
"""

AUTOM = """
---

## 2026-11-12 22:47（本次运行 · 第 30 次独立复核）

**判定：远端 E45 全部完成且与前二十九轮完全一致；裁决与写回均无需改动；本轮无远端改动（未 sbatch、未改 docs），新增只读探针 `tools/_e45_eval_probe.py` 与复核脚本 `tools/_e45_verify_round30.py`。**

- 远端：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 全 `COMPLETED`、评测 274/282/290 全 `COMPLETED`、汇总 291 `COMPLETED`、21 份评测齐、队列空；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- md5：13 份产物 **13/13 全同**；本轮新增 21 份 `*.eval.json` 远端 md5，**21/21 全同**。
- 核心闸门：round27 **POS 141/141、NEG 6/6、PAT 11/11**；round28 **POS 40/40、NEG 3/3**；round29 **POS 74/74、NEG 21/21** —— 全部 `VERDICT: PASS`。S3 六臂原始与配平后增益 CI 全跨零 ⇒ 六臂判负成立，合格臂集合为空。
- **本轮新增检查**（round30，**POS 61/61、NEG 0/4、VERDICT: PASS**）：**21 份远端原始评测文件的列联表端到端重算到本地聚合产物**。按 `radar_nwp_bench/metrics/precipitation.py` 权威公式重算 csi/bias/pod/far **336/336 逐位精确**；**ETS 与 HSS 需 `correct_negatives`（未存）⇒ 由 ETS 反解 CN 再回代 HSS，84 块零不一致、CN 全非负、隐含像素总数按 538 窗口归一后 spread=0.0**；重算值 ↔ `scores_summary.json` ↔ `scores_all.json` ↔ HANDOVER seed-0 锚点零不一致。
- 写回复核：`_sync_v245.py` **0 改动**、`_check_tables.py` **VERDICT: PASS**、`_check_v245_row.py` **VERDICT: PASS**；MASTER 头部 v2.45 全部在位。
- 经验：只有只依赖 H/M/FA 的指标（csi/bias/pod/far）可从评测 JSON 反算；`ets`/`hss` 需 `correct_negatives`、`fss` 需原始场 —— 首版按「7 指标均可反算」写导致 252 处假 FAIL。**先读权威实现再决定可反算范围**；且可由 ETS 反解 CN 来互相校验 HSS。
"""

TARGETS = [
    (r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md", MEM_ADD),
    (r"R:\1radarNWPprediction\.workbuddy\memory\2026-11-12.md", LOG),
    (r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\2026-11-12.md", LOG),
    (r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\automations"
     r"\e3cc6636-3e60-4a58-8f7e-67d4c896b0b6\memory.md", AUTOM),
]


def main():
    for path, text in TARGETS:
        if not os.path.exists(path):
            raise SystemExit("MISSING: " + path)
        with io.open(path, "r", encoding="utf-8") as fh:
            cur = fh.read()
        if "第 30 次独立复核" in cur:
            print("already present, skip:", path)
            continue
        with io.open(path, "a", encoding="utf-8", newline="\n") as fh:
            if cur and not cur.endswith("\n"):
                fh.write("\n")
            fh.write(text)
        print("appended:", path, "+%d chars" % len(text))


if __name__ == "__main__":
    main()
