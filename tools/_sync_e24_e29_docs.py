#!/usr/bin/env python3
"""Sync E24-E29 into MASTER 4.0 / 4.N / 9 / 12 and INDEX.md (R6).

Kept as provenance: the criteria files were written at 21:40, this document
sync happens immediately after and still before any compute is launched.
Every replacement asserts a unique match so a stale MASTER cannot be silently
corrupted.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "docs" / "MASTER.md"
INDEX = ROOT / "docs" / "experiments" / "INDEX.md"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[{label}] expected 1 match, found {n}")
    return text.replace(old, new, 1)


TABLE_ROWS = """| §4.26 | 把每一次降水过程单独拆开，看雷达反演雨量的偏差分布在三个数据划分上怎么变 | 待跑（判据已先写） | 预注册，未投算力；明确禁止写成因果链 | `docs/experiments/E24-per-case-zr-bias-distribution.md` |
| §4.27 | 用邻域评分和局部最大匹配，量一量位置偏了几公里吃掉多少分 | 待跑（判据已先写） | 预注册，未投算力；必须配等偏置的纯幅度对照 | `docs/experiments/E25-multiscale-fss-and-phase-relaxation.md` |
| §4.28 | 把十个随机种子的预测场直接平均或投票，看零成本合成能不能稳过地板 | 待跑（判据已先写） | 预注册，未投算力；禁止写『压平方差带』 | `docs/experiments/E26-ten-seed-field-ensemble.md` |
| §4.29 | 每个降水过程用历史真值雨量现算一个校正系数 | 待跑（判据已先写） | 预注册，未投算力；降级为理想上界诊断，不得与地板直接比 | `docs/experiments/E27-per-case-dynamic-bias-scalar.md` |
| §4.30 | 同一份第 0 轮权重只切换缺帧插值开关做配对 | 待跑（判据已先写） | 预注册，未投算力；收尾 §9 待办 7 | `docs/experiments/E28-ep00-no-interpolate-paired.md` |
| §4.31 | 面向分数损失的全部模型重测高频谱 | 待跑（判据已先写） | 预注册，未投算力；收尾 §9 待办 9，算力优先级最低 | `docs/experiments/E29-score-aware-spectral-rescan.md` |
"""

CARDS = [
    ("4.26", "把每一次降水过程单独拆开看反演偏差",
     "把『训练划分偏置 1.78、测试划分偏置 0.90』这两个聚合数字拆到每一次降水过程上，看差距是所有个例一起平移，还是被少数几次过程带出来的；三个数据划分用同一套完全不训练的解析反演场各算一遍。",
     "零算力重打分 + 数据集统计", "E24-per-case-zr-bias-distribution.md"),
    ("4.27", "用邻域评分和局部最大匹配测位移代价",
     "落实 §16 第一步与第二步：补 5×5 / 9×9 / 17×17 邻域评分，再允许预测在正负三像素内局部最大匹配，量出相位松弛后的分数上界。关键是必须配一个把预测乘常数到同等偏置的纯幅度对照，否则『多报面积』会被误当成『修正位置』。",
     "数据集实测（重新推理，零训练算力）", "E25-multiscale-fss-and-phase-relaxation.md"),
    ("4.28", "十个种子的预测场做平均与投票合成",
     "不额外训练，只把十个随机种子第 0 轮的预测场做算术平均与多数投票，看合成场能否稳定超过 0.0924 地板。事先把『压平 9.59 倍方差带』从判据里拿掉：十场平均得到的是一个模型，它没有跨种子方差可压。",
     "数据集实测（十次重新推理，零训练算力）", "E26-ten-seed-field-ensemble.md"),
    ("4.29", "逐过程用真值雨量现算一个校正系数",
     "每个起报窗口用同过程的历史地面雨量真值（无延迟 / 延迟两帧 / 延迟四帧）估一个均值场偏差系数去乘反演雨量，检验能否绕开训练与测试之间的标定漂移。因为用了真值，事先降级为理想上界诊断，不与地板直接比。",
     "零算力重打分（含真值标量，仅上界诊断）", "E27-per-case-dynamic-bias-scalar.md"),
    ("4.30", "第 0 轮权重上的缺帧插值配对",
     "收尾 §9 待办 7：同一份第 0 轮权重、只切换缺帧是否补插这一个变量，在二十三处断帧过程上做配对重采样，回答『容错有没有代价』。此前全矩阵的容错结果用末轮崩塌端权重，不能代表峰值。",
     "数据集实测（断帧划分，零训练算力）", "E28-ep00-no-interpolate-paired.md"),
    ("4.31", "面向分数损失的全模型高频谱重测",
     "收尾 §9 待办 9：十八个模型各重新推理一次，算高频段能量占比与 2 mm 档分数的相关，检验 §4.6 的『相关 0.950』并控制偏置做偏相关，维持『模糊是损失造成的、不是方法族性质』的因果修正。算力优先级最低。",
     "数据集实测（十八次重新推理，零训练算力）", "E29-score-aware-spectral-rescan.md"),
]

INDEX_ROWS = """| E24 | §4.26 | 把每一次降水过程单独拆开看反演偏差 | 零算力重打分 + 数据集统计 | `E24-per-case-zr-bias-distribution.md` | 预注册 |
| E25 | §4.27 | 用邻域评分和局部最大匹配测位移代价 | 数据集实测（重推理） | `E25-multiscale-fss-and-phase-relaxation.md` | 预注册 |
| E26 | §4.28 | 十个种子的预测场做平均与投票合成 | 数据集实测（十次重推理） | `E26-ten-seed-field-ensemble.md` | 预注册 |
| E27 | §4.29 | 逐过程用真值雨量现算一个校正系数 | 零算力重打分（理想上界） | `E27-per-case-dynamic-bias-scalar.md` | 预注册 |
| E28 | §4.30 | 第 0 轮权重上的缺帧插值配对 | 数据集实测（断帧划分） | `E28-ep00-no-interpolate-paired.md` | 预注册 |
| E29 | §4.31 | 面向分数损失的全模型高频谱重测 | 数据集实测（十八次重推理） | `E29-score-aware-spectral-rescan.md` | 预注册 |
"""

QUEUE_ROWS = """| 3m | **A.1 逐过程反演偏差分布（零算力）** | 基础 | 拆分训练 1.7820 / 测试 0.9048 的漂移形态；判据先写后跑 | ⏳ 判据已预注册（E24 / §4.26），待跑 |
| 3n | **A.2 多尺度邻域评分 + 局部最大匹配上界** | 基础 | 落地 §16 第一、二步；必须配等偏置幅度对照 | ⏳ 判据已预注册（E25 / §4.27），待跑 |
| 3o | **A.4 十种子预测场集成** | 基础 | 检验零成本合成能否稳过地板；不得写『压平方差带』 | ⏳ 判据已预注册（E26 / §4.28），待跑 |
| 3p | **B.1 逐过程动态偏差标量（理想上界）** | 基础 | 用历史真值绕开跨过程漂移；不同信息集不可与地板比 | ⏳ 判据已预注册（E27 / §4.29），待跑 |
"""

CHANGELOG = ("| 2026-11-06 | **v2.17** | **六项新尝试按 §11.7 预注册（零算力，判据早于算力）**："
             "① 用户提出两优先级共六方向，Gate 1 按机制审查后建立 E24–E29 / §4.26–§4.31 六份详情文件，"
             "判据写入时间 21:40 早于任何算力投入；② 修正三处概念错位：A.1『钉死因果链』降级为分布描述、"
             "A.4『压平 9.59× 方差带』因十种子不支持而登记未测、B.1『解决时滞翻转』因单标量不消除种子方差而删除；"
             "③ **实地核查的前置事实**：远端 `runs/` 无任何预测场文件（`.npy`/`.npz` 均为 0，只有 755 份权重与 "
             "1066 份报表 / 逐窗口列联表）→ A.2 的局部最大匹配、A.4 的场集成、待办 9 的谱重测都必须**重新推理**，"
             "属零训练算力但非零算力；④ 评测器**早已实现**邻域评分（默认半径 3）且既有报表已有 `fss` 字段，"
             "§16 原文『当前没有邻域 FSS』与代码现状不符，本轮只补多尺度；⑤ 同步 §4.0 / §4.N 卡片 / §9 3m–3p / "
             "INDEX.md / 顶部版本戳；⑥ 不改任何实验结论、不动 §10.2 / §10.3 | "
             "`docs/experiments/E24–E29*.md`、`tools/_register_e24_e29.py`、`tools/_sync_e24_e29_docs.py`；"
             "远端 `runs/` 文件清点、`scripts/evaluate.py --fss-radii`、`radar_nwp_bench/metrics/precipitation.py` |\n")


def main() -> None:
    t = MASTER.read_text(encoding="utf-8")

    t = replace_once(t, "| 最后更新 | 2026-11-06（v2.16） |",
                     "| 最后更新 | 2026-11-06（v2.17） |", "version stamp")

    anchor = "| §4.25 | 开放量子耗散跃迁门控 | 未执行 | ⛔ 否决作为科学方向 | — |\n"
    t = replace_once(t, anchor, anchor + TABLE_ROWS, "4.0 table")

    card_block = ""
    for num, title, goal, evidence, slug in CARDS:
        card_block += f"""### {num} {title}

**当时想干什么（大白话）**：{goal}

**结果怎么样**：待跑。判据已在投入任何算力之前写死（21:40），完整结果尚未产生。

**裁决**：待判（预注册）。

**证据层级**：{evidence}。

→ 完整结果、逐条判据核对与复现产物见 `docs/experiments/{slug}`。

"""
    t = replace_once(t, "---\n\n## 5. 核心代码结构",
                     card_block + "---\n\n## 5. 核心代码结构", "4.N cards")

    q_anchor = "| 11 | **四梯队攻坚（外部方案审查后执行）**"
    idx = t.index(q_anchor)
    line_end = t.index("\n", idx) + 1
    t = t[:line_end] + QUEUE_ROWS + t[line_end:]

    t = replace_once(
        t,
        "| 7 | **`--no-interpolate` 配对实验** | 基础 | 容错机制有效性的唯一合法对照 | ⏳ 待跑（**注意**：必须**同一 ckpt** 只切插值开关；**不得用 `last.pt`**（=ep19 崩塌端）；`robustness_missing` 仅 **23 个断帧 case** → 样本量小，须报逐时效差值 + CI） |",
        "| 7 | **`--no-interpolate` 配对实验** | 基础 | 容错机制有效性的唯一合法对照 | ⏳ 待跑（**注意**：必须**同一 ckpt** 只切插值开关；**不得用 `last.pt`**（=ep19 崩塌端）；`robustness_missing` 仅 **23 个断帧 case** → 样本量小，须报逐时效差值 + CI）。**判据已预注册**（E28 / §4.30） |",
        "queue 7")
    t = replace_once(
        t,
        "| 9 | score-aware 全矩阵的谱重测 | 基础 | 支撑\"塌陷是损失的性质\"的完整论证 | ⏳ 待跑 |",
        "| 9 | score-aware 全矩阵的谱重测 | 基础 | 支撑\"塌陷是损失的性质\"的完整论证 | ⏳ 待跑（**判据已预注册** E29 / §4.31；需十八次重推理，算力优先级最低） |",
        "queue 9")

    # Append changelog row directly after the v2.16 row (last row of the table).
    v216_line_start = t.index("| 2026-11-06 | **v2.16** |")
    v216_line_end = t.index("\n", v216_line_start) + 1
    t = t[:v216_line_end] + CHANGELOG + t[v216_line_end:]

    MASTER.write_text(t, encoding="utf-8")

    i = INDEX.read_text(encoding="utf-8")
    i_anchor = "| E23 | §4.24 | 审查最优传输损失探针的两条判据是否成立 | 合成场零算力机制探针（**不写入 §10**）+ 代码事实 | `E23-swd-mechanism-probe.md` | 已执行 |\n"
    i = replace_once(i, i_anchor, i_anchor + INDEX_ROWS, "index rows")
    INDEX.write_text(i, encoding="utf-8")
    print("MASTER + INDEX synced; 6 cards, 6 index rows, 4 queue rows, 1 changelog row")


if __name__ == "__main__":
    main()
