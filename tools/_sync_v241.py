#!/usr/bin/env python3
"""E45 v2.41 -- write the criterion revision back into MASTER (Gate 3).

The revision itself is in
``docs/experiments/E45-gate-direction-amount-separation-baseline.md``
(section 2.5).  The master document must show it in the four places the
protocol names: the header line, the section 4.47 card, the section 9
table row and the section 12 change log.

The edits are made as whole-line replacements found by their leading
marker, so that a line that has already been edited is not edited twice
and a line that is not there is reported rather than silently skipped.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

MASTER = Path("docs/MASTER.md")

HEADER_OLD = "| 最后更新 | 2026-11-10（v2.40，**E45 已预注册（判据 19:40 冻结）**"
HEADER_NEW = (
    "| 最后更新 | 2026-11-10（v2.41，**E45 判据修订（用户两条裁决，投算力前冻结）**："
    "**① A3 拆成两臂**——v2.40 把「平滑真值 + 面积归一 + 幅度截断」三处捆在一个臂里，"
    "「A3 失败」对任何组合都成立 ⇒ 无法按机制族给结论（违 §11.8）；现改为 **A3a 只加幅度截断**"
    "（治针尖冲到天花板 402.43 mm）、**A3b 只加重分配**（治 top 1 % 像素承担 0.9399 梯度 L1），"
    "**每臂相对 E44-A3 只差一处**；两臂都判负 ⇒ 拓扑支结掉；两臂都成立 ⇒ 下一轮必须做合并臂。"
    "**② A4 增「不要新头」的二版本**——`A4` 的量由新增标量头承担，`A4b`（`scaled_structure_own`）"
    "由**模型自己已有的普通解码头**的空间均值承担；**并新增退��审查**：`scale_cv ≥ 0.05` 且 "
    "`scale_truth_corr ≥ 0.3` 才算量通道真的逐样本地做量（§6.3 那条风险变成可判定数字）。"
    "**③ 规模**：7 臂 × 3 seed = **21 个训练作业**（B0 同批次重跑）。"
    "**④ 报告口径**：单点分数不得单独出现，必须给中位数与通过率（实测峰值端离散 9.59×）"
  )

CARD_MARK = "| §4.47 | 修三处机制缺陷 + 把比较基准摆正：门指向、梯度集中、量与结构解耦 |"
CARD_NEW = (
    "| §4.47 | 修三处机制缺陷 + 把比较基准摆正：门指向、梯度集中、量与结构解耦 | "
    "E44 四臂全负面，但产出了三条可操作的事实与一个基准错误。"
    "**① 门指向**：`corr(mask, amp) = −0.858320`、峰值处 `mask = 0.195087`，"
    "而 `mask ≡ 1` 的完全旁路与原始幅度同为 **1.000** ⇒ 投影无罪，门有罪；"
    "零算力机理检验：**真值外的针尖从 2 mm 到 400 mm 损失只变 0.005**（免费）、"
    "**门乘残差时 `d(mm)/d(mask) = (amp−low)·exp(out)`**（峰值杠杆最大）、"
    "**真值 0 处 reg 权重 1.0 而真值 5 mm 处 2.973** ⇒ 模型的最优策略是「关掉所有强峰」。"
    "**② 梯度集中**：top 1 % 像素承担 **0.9399** 的梯度 L1，针尖免费 ⇒ 冲到 `expm1` 天花板 **402.43 mm**。"
    "**③ 量与结构**：A2/A4 结构改善（`kernel_px` ↑ 且 `tail_max` ↑，A4 `tail_max` 4.1323 最接近真值 4.5475）"
    "但 `BIAS` 都下降 ⇒ 两者被耦合。**④ 基准**：全量 393 份评测调查显示**全历史最高分 "
    "`base_concat_s0_ep00` = 0.186511 是十 seed 里最好的一颗**，十 seed 中位仅 **0.0720**（4/10 过地板），"
    "`qpe_0` 中位 **0.0959**（5/10） | "
    "**v2.41 修订（用户 2026-11-10 两条裁决，投算力前冻结，已开跑）**："
    "**① A3 拆成两臂**（v2.40 的三处捆绑无法按机制族归因，违 §11.8）："
    "**A3a「幅度护栏」＝只加 `--topo-clamp-mm 20`**（治针尖冲天花板）、"
    "**A3b「梯度重分配」＝只加 `--topo-truth-smooth-sigma 1.0` + `--topo-area-normalize`**"
    "（治梯度集中）；**每臂相对 E44-A3 只差一处**；两臂**模型相同**（`topo_simvp`）、只差训练侧损失旗标。"
    "新增预注册：**两臂都判负 ⇒ 拓扑支结掉**（本轮唯一的结掉判据）；"
    "**两臂都成立 ⇒ 下一轮必须做一次合并臂**（否则漏掉叠加增益）。"
    "**② A4 增「不要新头」的二版本**：**A4**（`scaled_structure`）的量由**新增标量头**承担；"
    "**A4b**（`scaled_structure_own`）的量由**模型自己已有的普通解码头** `dec[-1]` 的空间均值承担，"
    "**不新增任何标量头**；两臂共用结构算子 ⇒ `mean(out) = T` 的算术保证都成立"
    "（结构改不动总量）；**必须注册成两个模型名**（运行时开关不改 state_dict，评测端漏传就悄悄评测被关掉的机制）。"
    "**新增退化审查**：`scale_cv ≥ 0.05` 且 `scale_truth_corr ≥ 0.3` 才算「量通道真的逐样本地做量」，"
    "否则 `scale_is_constant` 为真 ⇒ 该臂的机制主张不成立；`mean_out_equals_T` 不达标 ⇒ **实现缺陷**，"
    "修后重测、不得当机制结论。**A4 的结构算子沿用 E44-A2 的加性分解**（`low + sparse`）——"
    "否则测的是「凭空新头」而不是「解耦能否救回已知的结构改善」。"
    "**③ 规模＝阶 1 三 seed**：7 臂 × 3 seed = **21 个训练作业**（B0 含在内，同批次重跑）；"
    "**④ 报告口径**：单点分数不得单独出现，必须给中位数与通过率。"
    "**⑤ 判据主干零改动**：S0–S6 沿用 E41 §5 原文 | "
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` |"
)

STATUS_MARK = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
STATUS_NEW = (
    "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** | "
    "**v2.41 已修订并开跑（2026-11-10 24:00）**：四臂 → **七臂**"
    "（A1 幅度单调门、A2 硬支撑、**A3a 只加幅度截断**、**A3b 只加重分配**、"
    "**A4 标量头版**、**A4b 不要新头版**、B0 基线），每臂 3 seed = **21 个训练作业**；"
    "**新增 `scale_cv`／`scale_truth_corr` 退化审查**（量通道若退化成常数 ⇒ 机制主张不成立）；"
    "**S0 门本地与远端各跑一次，5 臂逐位等价全部 PASS**；"
    "**远端已投**：提交器 110 → 训练 111–116 → 评测 117–121 → 汇总作业；"
    "**seed 0 训练已完成**（job 94，7/7 `COMPLETED`）；"
    "**待判读**：按新口径（中位数 + 通过率）出结论，再写回本行与 §12 |"
)

CHANGELOG_ANCHOR = "| 2026-11-10 | **v2.40** |"
CHANGELOG_NEW = (
    "| 2026-11-10 | **v2.41** | "
    "**E45 判据修订（用户两条裁决）+ 首批训练已开跑（v2.41 / §4.47）**："
    "① 用户裁决一「A3 拆成两臂」：v2.40 把三处改动捆在一个臂里，「A3 失败」对任何组合都成立 ⇒ "
    "**无法按机制族给结论**（违 §11.8）⇒ 拆为 **A3a 只加幅度截断 20 mm**（治针尖冲到天花板 402.43 mm）"
    "与 **A3b 只加真值平滑 + 面积归一**（治 top 1 % 像素承担 0.9399 梯度 L1），**每臂相对 E44-A3 只差一处**；"
    "两臂模型相同（`topo_simvp`）只差训练侧旗标 ⇒ 评测命令一致；"
    "新增预注册「**两臂都判负 ⇒ 拓扑支结掉**」「**两臂都成立 ⇒ 下一轮必须做合并臂**」。"
    "② 用户裁决二「A4 的 scale 先测一个不要新头的二版本 + 审查 scale 是否变成常数」："
    "**A4**（`scaled_structure`，量由新增标量头承担）与 **A4b**（`scaled_structure_own`，"
    "量由模型**自己已有的普通解码头** `dec[-1]` 的空间均值承担，**不新增任何标量头**）"
    "**共用结构算子** ⇒ `mean(out) = T` 的算术保证都成立；**必须注册成两个模型名**"
    "（运行时开关不改 state_dict，评测端漏传会悄悄评测被关掉的机制——本项目吃过的亏）；"
    "**新增退化审查**：`scale_cv ≥ 0.05` 且 `scale_truth_corr ≥ 0.3` 才算量通道真的逐样本地做量，"
    "否则 `scale_is_constant` 为真 ⇒ 该臂机制主张不成立；`mean_out_equals_T` 不达标 ⇒ 实现缺陷（修后重测）。"
    "**A4 的结构算子改回 E44-A2 的加性分解**（原实现是凭空新 `struct_head`，那样测的是「凭空新头」"
    "而不是「解耦能否救回已知的结构改善」）→ **我自己的设计偏差，已纠**。"
    "③ 实现：`models/{amp_monotone_gate,amp_hard_support,scaled_structure}/`；"
    "`topology_losses.py` 加 `clamp_mm`／`truth_smooth_sigma`／`area_normalize` 三开关，"
    "**默认全关 ⇒ 逐位走 `_forward_reference`**（与 E44 逐位相同）；"
    "面积图用 **GPU 最小标签传播**（**不得**用 numpy union-find：E44 踩过，梯度会断）。"
    "④ 工具：`tools/_e45_mechanism_probe.py`（读数，**import** E44 探针实现不另写）、"
    "`tools/_e45_s0_equivalence.py`（5 臂逐位等价）、`tools/_e45_collect_scores.py`"
    "（**中位数 + 通过率**，单点分数不得单独出现）。"
    "⑤ **工程坑（本轮第三次踩提交器）**：登录会话的 nohup 提交器**第三次**被回收（只提交了 seed 0 就断了）"
    "⇒ 改为 `scripts/_submit_e45.sbatch`（提交器本身做成 Slurm 作业）；"
    "**平台运行作业上限 4 是按子任务算的** ⇒ 7 子任务的数组占满额度会让下一批 `sbatch` **阻塞**在提交器里"
    "⇒ 训练与评测都按 **3 个一批**拆；**改了脚本必须先同步再提交**——本轮改完 `ARM_LIST` 解析就提交，"
    "job 103 秒失败（`unknown ARM_LIST 0`），**其后整条 afterok 链 `DependencyNeverSatisfied`** ⇒ "
    "提交前先在远端 `bash -c` 验证参数解析；提交额度不足时 `sbatch` 被拒（`QOSMaxSubmitJobPerUserLimit`）"
    "⇒ 提交器必须重试（本轮重试机制正常工作）；**数组日志名用 `%A_%a`**（`%x-%j` 会让同数组子任务"
    "**互相覆盖同一个日志文件**，seed 0 的日志即因此丢失，训练本身不受影响）。"
    "⑥ 规模＝阶 1 三 seed：**21 个训练作业** + 21 份评测 + 结构量 + S3 + 汇总；"
    "**S0 门本地与远端各跑一次，5 臂逐位等价全部 PASS**；"
    "**远端已投**：提交器 110 → 训练 111–116 → 评测 117–121 → 汇总；**seed 0 已完成**（job 94，7/7 `COMPLETED`）。 | "
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` · "
    "`models/{amp_monotone_gate,amp_hard_support,scaled_structure}/` · "
    "`radar_nwp_bench/metrics/topology_losses.py` · "
    "`tools/_e45_{mechanism_probe,s0_equivalence,collect_scores}.py` · "
    "`scripts/e45_{train,stage2,summary}.sbatch` · `scripts/_submit_e45.sbatch` |"
)


def replace_line(lines: list[str], marker: str, new_line: str) -> int:
    """Replace the single line that starts with ``marker``."""
    hits = [i for i, line in enumerate(lines) if line.startswith(marker)]
    short = marker[:60]
    if not hits:
        print(f"NOT FOUND: {short}")
        return 0
    if len(hits) > 1:
        print(f"AMBIGUOUS ({len(hits)} hits): {short}")
        return 0
    index = hits[0]
    lines[index] = new_line
    print(f"replaced line {index + 1}: {marker[:60]}")
    return 1


def append_line(lines: list[str], marker: str, new_line: str) -> int:
    """Insert ``new_line`` right after the line that starts with ``marker``."""
    hits = [i for i, line in enumerate(lines) if line.startswith(marker)]
    short = marker[:60]
    if not hits:
        print(f"NOT FOUND: {short}")
        return 0
    if len(hits) > 1:
        print(f"AMBIGUOUS ({len(hits)} hits): {short}")
        return 0
    lines.insert(hits[0] + 1, new_line)
    print(f"appended after line {hits[0] + 1}: {short}")
    return 1


def main() -> int:
    text = MASTER.read_text(encoding="utf-8")
    lines = text.split("\n")

    changed = 0
    changed += replace_line(lines, HEADER_OLD, HEADER_NEW)
    changed += replace_line(lines, CARD_MARK, CARD_NEW)
    changed += replace_line(lines, STATUS_MARK, STATUS_NEW)
    # The change log grows: the new version is appended after the last
    # entry of the day, not written over it.
    changed += append_line(lines, CHANGELOG_ANCHOR, CHANGELOG_NEW)

    if changed != 4:
        print(f"FATAL: {changed} of 4 places were written")
        return 1

    MASTER.write_text("\n".join(lines), encoding="utf-8")
    print("MASTER.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
