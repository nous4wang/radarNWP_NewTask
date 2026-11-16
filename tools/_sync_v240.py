#!/usr/bin/env python3
"""One-shot patcher for the MASTER and INDEX entries of E45.

Kept as a file rather than a heredoc: the string literals here are long
and a heredoc through the shell is where quoting goes wrong.
"""
from __future__ import annotations

from pathlib import Path

CARD = """
| §4.47 | 修三处机制缺陷 + 把比较基准摆正：门指向、梯度集中、量与结构解耦 | E44 四臂全负面，但产出了三条可操作的事实与一个基准错误。**① 门指向**：`corr(mask, amp) = −0.858320`、峰值处 `mask = 0.195087`，而 `mask ≡ 1` 的完全旁路与原始幅度同为 **1.000** ⇒ 投影无罪，门有罪；零算力机理检验：**真值外的针尖从 2 mm 到 400 mm 损失只变 0.005**（免费）、**门乘残差时 `d(mm)/d(mask) = (amp−low)·exp(out)`**（峰值杠杆最大）、**真值 0 处 reg 权重 1.0 而真值 5 mm 处 2.973** ⇒ 模型的最优策略是「关掉所有强峰」。**② 梯度集中**：top 1 % 像素承担 **0.9399** 的梯度 L1，针尖免费 ⇒ 冲到 `expm1` 天花板 **402.43 mm**。**③ 量与结构**：A2/A4 结构改善（`kernel_px` ↑ 且 `tail_max` ↑，A4 `tail_max` 4.1323 最接近真值 4.5475）但 `BIAS` 都下降 ⇒ 两者被耦合。**④ 基准**：全量 393 份评测调查显示**全历史最高分 `base_concat_s0_ep00` = 0.186511 是十 seed 里最好的一颗**，十 seed 中位仅 **0.0720**（4/10 过地板），`qpe_0` 中位 **0.0959**（5/10） | **`PRE-REGISTERED / 判据已冻结（19:40）`**：① 判据沿用 E41 §5 的 S0–S6 **原文一字未改**，追加三条区分性读数（`corr(mask, amp) ≥ 0`、`mask_at_peak > 0.5`、`maxratio > 0.9`；A3 的 `grad_top1pct_share < 0.5` 与场最大值 ≤ 30 mm）；② 四臂各只改一环：**A1 幅度单调门**（`mask = sigmoid((resid−thr)/T)`，正相关**由构造保证**）、**A2 硬支撑 + 直通**、**A3 梯度均衡的拓扑项**（平滑真值 + 梯度按分量面积归一 + 幅度截断）、**A4 总量标定器 + 结构算子**（`out = scale·structure`，结构改动**在数学上不改总量**）；③ **小波被定量否掉**：真实 LABEL 场上低频带走 **99.7–100 %** 的量、且低频 max 10.9 远低于真值 18.8 ⇒ 低频**没有重尾**，小波分离的是**尺度**不是**量 vs 结构**；④ **规模＝阶 1 三 seed**（用户 2026-11-10 裁决），每臂 3 seed + 同批次 B0 3 seed = 15 个训练作业，**基线必须同批次重跑**；⑤ **报告口径改**：单点分数不得单独出现，必须同时给中位数与通过率（依据：峰值端离散度实测 **9.59×**） | `docs/experiments/E45-gate-direction-amount-separation-baseline.md` |

---

## 5. 核心代码结构"""

E45_ROW = ("| **E45 / §4.47 修三处机制缺陷 + 摆正基准** | **`PRE-REGISTERED / 判据已冻结（19:40）`** | "
 "四臂：**A1 幅度单调门**、**A2 硬支撑 + 直通**、**A3 梯度均衡的拓扑项**、**A4 总量标定器 + 结构算子**。"
 "**S0–S6 原文一字未改**，追加三条区分性读数。**规模＝阶 1 三 seed**（每臂 3 seed + 同批次 B0 3 seed）。"
 "**小波已定量否掉**（低频带走 99.7–100 % 的量且无重尾）。**报告口径改**：必须给中位数与通过率，"
 "单点分数不得单独出现 |")

COMPUTE = ("| **算力去向** | **`E45 已预注册（19:40 冻结），下一步按此投算力**` | "
 "E44 的四族已舍弃，但产出了三条可操作事实："
 "**① 门指向**（`corr(mask,amp) = −0.858`，针尖免费 + 峰值杠杆最大 ⇒ 模型学会关掉强峰）→ "
 "**A1 幅度单调门**（正相关由构造保证）；"
 "**② 梯度集中**（top 1 % 像素承担 0.9399 梯度 L1 + 针尖免费 ⇒ 冲天花板 402 mm）→ **A3 梯度均衡**；"
 "**③ 量与结构耦合**（A2/A4 结构好了但 `BIAS` 掉）→ **A4 显式总量标定器**；"
 "**④ 基准摆正**（全历史最高分是十 seed 里最好的一颗，中位仅 0.0720；峰值端离散 9.59×）→ "
 "**从本轮起只报中位数与通过率**。**§11.9 P2 的 ≥10 seed 仍未做** |")

LOG = ("| 2026-11-10 | **v2.40** | **E45 预注册：审全部结果 → 修门指向／梯度集中／量结构分离，并把比较基准摆正（E45 / §4.47，判据 19:40 冻结）**："
 "① 用户要求「审查所有结果，找出未来改进方向」并具体追问三件事（A3 梯度集中怎么解决、门与幅度反相关怎么保证、A2/A4 的改善怎么保留、"
 "量和结构分离用小波还是别的、每任务 3 seed）；"
 "② **三项零算力机理检验（本地，`balanced` 损失真实实现）**："
 "**针尖免费**（真值外的针尖从 2 mm 冲到 400 mm，总损失只变 0.374→0.380；因 `reg` 权重由**真值**决定，真值 0 处 weight = 1.0、"
 "真值 5 mm 处 = 2.973）；**门乘残差时 `d(mm)/d(mask) = (amp−low)·exp(out)`**（峰值处杠杆最大）；"
 "**幅度在真值外被高估时损失上升**（真值 5 mm 处 amp 20 mm → 损失 0.377→0.536）⇒ "
 "**模型的最优策略就是「把所有强峰当背景关掉」**，与实测 `corr(mask,amp) = −0.858320`、`mask_at_peak = 0.195087` 完全吻合；"
 "③ **⚠️ 发现四（全量调查，`tools/_e45_bias_csi_survey.py`，393 份 538 窗评测）**："
 "**全历史最高 `CSI@2.0` = 0.186511（`base_concat_s0_ep00`）不是稳定水平，而是十 seed 里最好的一颗**；"
 "`tools/_e45_baseline_seeds.py` 给出真实图景：`base_concat` ep00 十 seed **中位 0.0720、4/10 过地板、离散 9.59×**，"
 "`qpe_0` ep00 **中位 0.0959、5/10、离散 3.21×** ⇒ **此前「A1 = 0.1492 是最高分」是与一颗幸运种子比较**；"
 "同时 `CSI@2.0` 对 `BIAS@2.0` 的 logit 拟合 **R² = 0.5886**（残差 |median| 0.0193）⇒ "
 "**BIAS 只解释约六成，结构仍有四成空间**（「只要把量做对就能赢」是错的）；"
 "④ **小波被定量否掉**（真实 LABEL 场 `CP_01_05171710`）：尺度 1/3、1/7、1/11 的低频带分别带走 "
 "**99.9 %／100.0 %／99.7 %** 的量，且低频 max（10.9）远低于真值 max（18.8）⇒ **低频没有重尾**，"
 "「低频 = 平滑背景」与「低频 = 量」是同一件事的两面 ⇒ **小波分离的是尺度，不是量 vs 结构**；"
 "另排除逐窗保序校准（**E27 已证逐窗真值动态标量不可部署**）；"
 "⑤ **E45 四臂**：**A1 幅度单调门**（`mask = sigmoid((resid−thr)/T)`，正相关**由构造保证**、峰值幅度完整保留）、"
 "**A2 硬支撑 + 直通**、**A3 梯度均衡的拓扑项**（平滑真值 + 梯度按分量面积归一 + 幅度截断 20 mm）、"
 "**A4 总量标定器 + 结构算子**（`out = scale·structure`，结构改动**在数学上不改总量**）；"
 "⑥ **规模＝阶 1 三 seed**（用户裁决），4 臂 × 3 seed + **同批次 B0 × 3 seed** = 15 个训练作业，"
 "**基线必须同批次重跑**（A4 改了输出参数化）；"
 "⑦ **报告口径改**（依据 9.59× 的实测离散度）：**单点分数不得单独出现**，必须同时给中位数与通过率；"
 "⑧ **判据零改动**：S0–S6 沿用 E41 §5 原文，只追加三条区分性读数；"
 "⑨ **工程**：远端 heredoc 的 f-string 不能含转义引号（写成 `.py` 文件再同步）；"
 "`sacct` 对排队中的数组只给父行。 | "
 "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` · "
 "`tools/_evals/e45_{bias_csi_survey,baseline_seeds}.json` · "
 "`tools/_e45_{bias_csi_survey,baseline_seeds}.py` |")

HEADER = ("| 最后更新 | 2026-11-10（v2.40，**E45 已预注册（判据 19:40 冻结）**："
 "修三处机制缺陷——**门指向**（幅度单调门，正相关由构造保证）、**梯度集中**（梯度均衡的拓扑项）、"
 "**量与结构解耦**（显式总量标定器 `out = scale·structure`）；"
 "**小波被定量否掉**（低频带走 99.7–100 % 的量且无重尾）；"
 "**比较基准已摆正**——全历史最高分 0.1865 是十 seed 里最好的一颗，`base_concat` 中位仅 0.0720、"
 "峰值端离散 9.59× ⇒ **从本轮起只报中位数与通过率**） |")

INDEX = ("| E45 | §4.47 | 修三处机制缺陷 + 把比较基准摆正：门指向、梯度集中、量与结构解耦 | "
 "**零算力调查 + 机理检验**（`tools/_e45_bias_csi_survey.py` 393 份评测 + "
 "`tools/_e45_baseline_seeds.py` + `balanced` 损失与门组合式的检验）+ 判据 19:40 冻结 | "
 "`E45-gate-direction-amount-separation-baseline.md` | "
 "**`PRE-REGISTERED / 判据已冻结`**：**S0–S6 原文一字未改**，追加三条区分性读数；"
 "四臂＝**A1 幅度单调门**（`corr(mask,amp) ≥ 0` 由构造保证）、**A2 硬支撑 + 直通**、"
 "**A3 梯度均衡的拓扑项**（`grad_top1pct_share < 0.5`、场最大值 ≤ 30 mm）、"
 "**A4 总量标定器 + 结构算子**（`out = scale·structure`，结构改动不改总量）；"
 "**规模＝阶 1 三 seed**（每臂 3 seed + 同批次 B0 3 seed = 15 作业，基线必须同批次重跑）；"
 "**小波已定量否掉**（低频带走 99.7–100 % 的量且低频无重尾）；"
 "**报告口径改**：单点分数不得单独出现，必须给中位数与通过率 |")


def main() -> int:
    p = Path("docs/MASTER.md")
    lines = p.read_text(encoding="utf-8").split("\n")

    anchor = "\n---\n\n## 5. 核心代码结构"
    text = "\n".join(lines)
    assert text.count(anchor) == 1, text.count(anchor)
    text = text.replace(anchor, CARD)

    lines = text.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("| **算力去向** |"))
    lines[i] = COMPUTE
    lines.insert(i + 1, E45_ROW)
    text = "\n".join(lines)

    lines = text.split("\n")
    idx = next(k for k, l in enumerate(lines)
               if l.startswith("| 2026-11-10 | **v2.39**"))
    lines.insert(idx + 1, LOG)
    text = "\n".join(lines)

    lines = text.split("\n")
    for k, l in enumerate(lines):
        if l.startswith("| 最后更新 |"):
            lines[k] = HEADER
            break
    text = "\n".join(lines)

    p.write_text(text, encoding="utf-8")
    print("MASTER updated: 4.47 card, 9.z E45 row, compute, 12 v2.40, header")

    p2 = Path("docs/experiments/INDEX.md")
    lines2 = p2.read_text(encoding="utf-8").split("\n")
    j = next(k for k, l in enumerate(lines2) if l.startswith("| E44 |"))
    lines2.insert(j + 1, INDEX)
    p2.write_text("\n".join(lines2), encoding="utf-8")
    print("INDEX E45 row inserted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
