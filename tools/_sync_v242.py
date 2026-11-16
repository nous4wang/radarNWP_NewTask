#!/usr/bin/env python3
"""Write E45's v2.42 revision into MASTER.md and INDEX.md.

Four places have to change, and they have to change together:
the version line at the top, the section 4.47 card, the section 9
status row, and the section 12 changelog.  Doing it by hand is how the
four drift apart, so it is done here.

The script is idempotent in the sense that it refuses to double-write:
it looks for the marker of the revision it is writing and stops if it
is already there.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "docs" / "MASTER.md"
INDEX = REPO / "docs" / "experiments" / "INDEX.md"

MARKER = "v2.42"


def replace_line(lines: list[str], marker: str, new_line: str) -> int:
    """Replace the single line that starts with ``marker``."""
    hits = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not hits:
        print(f"NOT FOUND: {marker[:70]}")
        return 0
    if len(hits) > 1:
        print(f"AMBIGUOUS ({len(hits)} hits): {marker[:70]}")
        return 0
    lines[hits[0]] = new_line
    return 1


def main() -> int:
    changed = 0

    text = MASTER.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{MARKER} is already in MASTER.md; nothing to do")
        return 0
    lines = text.split("\n")

    # 1. the version line at the top
    header_old = "| 最后更新 | 2026-11-10（v2.41，"
    header_new = (
        "| 最后更新 | 2026-11-11（v2.42，**E45 第一批全废的根因＝四个传参 bug，"
        "已修并重跑**）：**判据一字未改**（S0–S6 原文、臂的构成、每臂 3 seed、"
        "读数口径全部照旧），改的是**执行机制**。四个 bug **全部静默**——作业正常退出、"
        "日志看起来健康、结果是错的或缺的：**① `--export` 用逗号分隔变量**（Slurm 语义），"
        "`ARM_LIST=0,1,2` 被切成 `ARM_LIST=0` + 变量 `1` + 变量 `2` ⇒ **每个训练作业只训 1 个臂**，"
        "**21 个臂-种子组合里 8 个从未训练**；**② `e45_train.sbatch` 丢了 `#SBATCH --array=0-6`** ⇒ "
        "作业以单任务跑、`SLURM_ARRAY_TASK_ID` 为空；**③ 拓扑旗标走环境变量而提交器从未导出** ⇒ "
        "脚本收到 0.0，而 **0.0 的意思是「关闭」** ⇒ **A3a 根本没测幅度截断、A3b 根本没测真值平滑**"
        "（远端 `config.json` 实测：`A3a_s0 topo_clamp_mm=0.0`、`A3b_s0 topo_truth_smooth_sigma=0.0`）；"
        "**④ 探针被传了 `--runs`（探针无此参数）与 `--arm \"A1|model=xxx\"`（`model` 不是合法内联选项）** "
        "⇒ argparse 报错 ⇒ 作业 FAILED ⇒ `afterok` 永不可满足 ⇒ 后续作业连锁 "
        "`DependencyNeverSatisfied`（这才是用户看到的「dependency 出错」，**不是缺库**）。"
        "**修法**：臂**不再经过 --export**（臂 = 数组任务号，臂表写死在 `scripts/_e45_arms.sh`，"
        "训练与评测共用同一份）；加回 `--array=0-6` 并让脚本**显式拒绝**无 array id 的运行；"
        "旗标**硬编码进臂表**（训练侧第 4 列、探针侧第 5 列 ⇒ 两侧必然一致）；探针只带 `topo_*` 内联选项、"
        "模型由探针自己的 ARMS 表提供。**修复已被产物证实**：`A3a_s0` 的 `topo_clamp_mm` 由 **0.0 → 20.0**、"
        "`A3b_s0` 的 `topo_truth_smooth_sigma` 由 **0.0 → 1.0**，两臂各自只多一处改动（合 §11.8 单变量要求）。"
        "**第一批产物整体作废**，改名归档为 `runs/e45_brokenargs_20261111`（**保留为证据、不删**），"
        "全部 21 个组合重跑（B0 必须同批次重跑）。**回归防护**：`tests/test_e45_submission.py`（23 项）"
        "逐条锁住这四个 bug；顺带修好 `tools/remote.py` 的 `--sync`（原为 `nargs=\"+\"`，"
        "**第二个 `--sync` 会被当成第一个的补充值** ⇒ 一条命令里只有第一组生效、其余静默不传）。"
        "**⑤ 提交器的状态解析**：`sacct` 的状态列带空格（`\" COMPLETED \"`），"
        "模式 `COMPLETED*` **不匹配整词** ⇒ 提交器把跑完的训练数组误判为 FAILED 并停链；已改为先 trim。"
        "⑥ 分批落地方式改为**提交器自己串行推进**（提交一个 seed → 等它真跑完 → 提交下一个 → 汇总），"
        "因为提交额度按**已提交（含等待中）**计，一次投 21 个子任务必被拒；且**被拒不重试**，直接报错停下"
        "（第一版的重试循环把同一个被拒作业反复要了 4 小时，最后一个评测批次从未提交而日志看不出）。 |"
    )
    changed += replace_line(lines, header_old, header_new)

    # 2. the section 4.47 card
    card_marker = "| §4.47 |"
    hits = [i for i, line in enumerate(lines) if line.startswith(card_marker)]
    if len(hits) != 1:
        print(f"the section 4.47 card was not found uniquely ({len(hits)})")
        return 1
    card = lines[hits[0]]
    card_add = (
        " **v2.42（2026-11-11 09:00，判据零改动，执行机制重写）**："
        "第一批 21 个训练作业里 **8 个臂-种子组合从未训练**（`--export` 的逗号把臂列表截成了首元素，"
        "且 `e45_train.sbatch` 丢了 `--array` 指令 ⇒ 每个作业只训 1 个臂）；"
        "**A3a/A3b 的机制根本没生效**（旗标走环境变量而提交器从未导出 ⇒ 脚本收到 0.0，而 0.0 = 关闭；"
        "远端 `config.json` 实测 `topo_clamp_mm=0.0`、`topo_truth_smooth_sigma=0.0`）；"
        "**探针调用参数错**（`--runs` 探针无此参数、`model=` 不是合法内联选项）⇒ 作业 FAILED ⇒ "
        "后续作业连锁 `DependencyNeverSatisfied`。**第一批结果整体作废**，"
        "产物归档为 `runs/e45_brokenargs_20261111`（保留为证据），**21 个组合全部重跑**。"
        "**修法**：臂 = 数组任务号（不经 `--export`）、臂表单一来源 `scripts/_e45_arms.sh`、"
        "旗标硬编码进臂表、探针只带 `topo_*`。**修复已被产物证实**："
        "`A3a_s0 topo_clamp_mm` 0.0→**20.0**、`A3b_s0 topo_truth_smooth_sigma` 0.0→**1.0**。"
        "回归防护 `tests/test_e45_submission.py`（23 项）。"
    )
    # The card ends with the reference cell; insert before it.
    last_cell = card.rfind("| `docs/experiments/E45")
    if last_cell == -1:
        last_cell = card.rfind("|")
    lines[hits[0]] = card[:last_cell] + card_add + card[last_cell:]
    changed += 1

    # 3. the section 9 status row
    status_marker = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
    hits = [i for i, line in enumerate(lines) if line.startswith(status_marker)]
    if len(hits) != 1:
        print(f"the section 9 row was not found uniquely ({len(hits)})")
        return 1
    status = lines[hits[0]]
    status_add = (
        " **v2.42（2026-11-11 09:00）：第一批全废并已重跑**——查明**四个传参 bug**"
        "（`--export` 逗号截断臂列表 ⇒ 8 个组合从未训练；训练脚本丢了 `--array` ⇒ 每作业只训 1 臂；"
        "拓扑旗标从未导出 ⇒ 0.0 = 关闭 ⇒ **A3a/A3b 机制没生效**；探针参数错 ⇒ 作业 FAILED ⇒ "
        "后续连锁 `DependencyNeverSatisfied`）。**判据一字未改**，改的是执行机制："
        "臂 = 数组任务号、臂表单一来源（`scripts/_e45_arms.sh`）、旗标硬编码进臂表。"
        "**产物证实修复生效**：`A3a_s0 topo_clamp_mm` 0.0→**20.0**、"
        "`A3b_s0 topo_truth_smooth_sigma` 0.0→**1.0**。旧产物归档 `runs/e45_brokenargs_20261111`（不删）。"
        "**待判读**：重跑完成后按新口径（中位数 + 通过率）出结论。"
    )
    last_cell = status.rfind("| **待判读**")
    if last_cell == -1:
        last_cell = status.rfind("|")
    lines[hits[0]] = status[:last_cell] + status_add + status[last_cell:]
    changed += 1

    # 4. the section 12 changelog
    changelog_marker = "| 2026-11-10 | **v2.41** |"
    hits = [i for i, line in enumerate(lines) if line.startswith(changelog_marker)]
    if len(hits) != 1:
        print(f"the v2.41 changelog row was not found uniquely ({len(hits)})")
        return 1
    insert_at = hits[0] + 1
    changelog = (
        "| 2026-11-11 | **v2.42** | **E45 第一批全废的根因＝四个传参 bug（判据零改动，执行机制重写并重跑）**："
        "① **用户问「远程出现 dependency 是不是要装什么库」——不是缺库**，是**作业链被自己的参数写错打断**。"
        "四个 bug **全部静默**（作业正常退出、日志健康、结果错或缺）："
        "**① `--export` 用逗号分隔变量**（Slurm 语义）⇒ `ARM_LIST=0,1,2` 被切成 `ARM_LIST=0` + 变量 `1` + 变量 `2` "
        "⇒ **每个训练作业只训 batch 的第一个臂** ⇒ **21 个臂-种子组合里 8 个从未训练**"
        "（远端实查：s1/s2 各只有 B0/A3a/A4b 三个）；"
        "**② `e45_train.sbatch` 丢了 `#SBATCH --array=0-6`** ⇒ 作业以单任务跑、`SLURM_ARRAY_TASK_ID` 为空；"
        "**③ 拓扑旗标走环境变量而提交器从未导出** ⇒ 脚本收到 0.0，而 **0.0 = 关闭** ⇒ "
        "**A3a 根本没测幅度截断、A3b 根本没测真值平滑**（远端 `config.json` 实测 `topo_clamp_mm=0.0`、"
        "`topo_truth_smooth_sigma=0.0`）⇒ **两臂测的不是它们声称的机制**；"
        "**④ 探针被传了 `--runs`（探针无此参数）与 `--arm \"A1|model=xxx\"`（`model` 不是合法内联选项，只允许 `topo_*`）** "
        "⇒ argparse 报错 ⇒ 作业 FAILED ⇒ `afterok` 永不可满足 ⇒ 后续作业连锁 `DependencyNeverSatisfied`。"
        "**修法**：**臂不再经过 `--export`**（臂 = 数组任务号）；**臂表单一来源** `scripts/_e45_arms.sh`"
        "（训练与评测共用，杜绝两处映射漂移）；**拓扑旗标硬编码进臂表**（训练侧第 4 列、探针侧第 5 列 ⇒ 两侧必然一致）；"
        "加回 `--array=0-6` 并让脚本**显式拒绝**无 array id 的运行。"
        "**修复已被产物证实**：`A3a_s0 topo_clamp_mm` **0.0→20.0**、`A3b_s0 topo_truth_smooth_sigma` **0.0→1.0**，"
        "两臂各自只多一处改动（合 §11.8 单变量要求）。"
        "**处置**：第一批产物**整体作废**，改名归档 `runs/e45_brokenargs_20261111`（**保留为证据、不删**），"
        "**21 个组合全部重跑**（B0 必须同批次重跑）。"
        "**⑤ 提交器的状态解析**：`sacct` 状态列带空格（`\" COMPLETED \"`），模式 `COMPLETED*` **不匹配整词** ⇒ "
        "提交器把跑完的训练数组误判为 FAILED 并停链（已 trim 修复）。"
        "**⑥ 分批落地**：改为提交器自己串行推进（提交一个 seed → 等它真跑完 → 下一个 → 汇总），"
        "因为提交额度按**已提交（含等待中）**计，一次投 21 个子任务必被拒；且**被拒不重试**，"
        "直接报错停下（第一版的重试循环把同一个被拒作业反复要了 4 小时，最后一个评测批次从未提交而日志看不出）。"
        "**⑦ 工具缺陷**：`tools/remote.py` 的 `--sync` 原为 `nargs=\"+\"`，**第二个 `--sync` 会被当成第一个的补充值** "
        "⇒ 一条命令里只有第一组生效、其余**静默不传**（曾让我误判「文件没同步上去」，白查一轮）；改为 "
        "`action=\"append\", nargs=\"+\"` 并展平。"
        "**回归防护**：`tests/test_e45_submission.py`（**23 项**）逐条锁住这四个 bug ＋ 状态解析；"
        "全量单测 **805 passed / 5 failed**（5 个失败与既有基线**逐条相同**，零回归）。"
        "**已跑通**：提交器 273 → seed 0 评测 274 在跑（seed 0 训练 266 已 7/7 `COMPLETED`，旗标已核实）。 |"
        " `docs/experiments/E45-gate-direction-amount-separation-baseline.md` · `scripts/_e45_arms.sh` · "
        "`scripts/_submit_e45.sbatch` · `tools/_e45_merge_struct.py` · "
        "`tests/test_e45_submission.py` |"
    )
    lines.insert(insert_at, changelog)
    changed += 1

    MASTER.write_text("\n".join(lines), encoding="utf-8")
    print(f"MASTER.md: {changed} place(s) written")

    # 5. INDEX.md
    index_text = INDEX.read_text(encoding="utf-8")
    if MARKER in index_text:
        print("INDEX.md already mentions v2.42; nothing to do")
    else:
        ilines = index_text.split("\n")
        hits = [i for i, line in enumerate(ilines) if line.startswith("| E45 |")]
        if len(hits) != 1:
            print(f"the INDEX E45 row was not found uniquely ({len(hits)})")
            return 1
        row = ilines[hits[0]]
        add = (
            "**v2.42 修订并重跑（2026-11-11 09:00，判据零改动）**："
            "**第一批 21 个训练作业里 8 个臂-种子组合从未训练，且 A3a/A3b 的机制根本没生效**——"
            "四个静默的传参 bug：**① `--export` 用逗号分隔变量**（`ARM_LIST=0,1,2` ⇒ 只剩 `0`）"
            "＋ **② 训练脚本丢了 `--array=0-6`** ⇒ 每个作业只训 1 个臂；"
            "**③ 拓扑旗标从未导出 ⇒ 0.0 = 关闭** ⇒ `A3a` 没测幅度截断、`A3b` 没测真值平滑"
            "（远端 `config.json` 实测 `topo_clamp_mm=0.0`）；"
            "**④ 探针参数错**（`--runs` 不存在、`model=` 不是合法内联选项）⇒ 作业 FAILED ⇒ "
            "后续连锁 `DependencyNeverSatisfied`（**用户看到的 dependency 出错不是缺库**）。"
            "**修法**：臂 = 数组任务号（不经 `--export`）、臂表单一来源 `scripts/_e45_arms.sh`、"
            "旗标硬编码进臂表（训练侧第 4 列 / 探针侧第 5 列）。"
            "**产物证实修复生效**：`A3a_s0 topo_clamp_mm` **0.0→20.0**、"
            "`A3b_s0 topo_truth_smooth_sigma` **0.0→1.0**。"
            "**第一批产物归档** `runs/e45_brokenargs_20261111`（保留为证据），21 个组合全部重跑。"
            "**回归防护** `tests/test_e45_submission.py`（23 项）"
        )
        last = row.rfind("|")
        ilines[hits[0]] = row[:last] + add + row[last:]
        INDEX.write_text("\n".join(ilines), encoding="utf-8")
        print("INDEX.md: 1 place written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
