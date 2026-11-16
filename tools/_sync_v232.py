#!/usr/bin/env python3
"""v2.32 atomic sync: E41 task-structure reparameterisation (SOTA route).

Order follows MASTER header line 4: authoritative source first, derived
layers (INDEX / README) second.  Every replacement asserts exactly one match
and nothing is written until all of them succeed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "docs/MASTER.md"
I = ROOT / "docs/experiments/INDEX.md"
R = ROOT / "README.md"

cache: dict[Path, str] = {}
for p in (M, I, R):
    cache[p] = p.read_text(encoding="utf-8").replace("\r\n", "\n")

count = 0


def rep(p: Path, old: str, new: str, label: str) -> None:
    global count
    n = cache[p].count(old)
    if n != 1:
        raise SystemExit(f"ABORT {label}: expected 1 match, found {n}")
    cache[p] = cache[p].replace(old, new, 1)
    count += 1
    print(f"  ok {label}")


def insert_before(p: Path, anchor: str, block: str, label: str) -> None:
    global count
    n = cache[p].count(anchor)
    if n != 1:
        raise SystemExit(f"ABORT {label}: expected 1 anchor, found {n}")
    cache[p] = cache[p].replace(anchor, block + anchor, 1)
    count += 1
    print(f"  ok {label}")


def insert_after_line(p: Path, prefix: str, new_line: str, label: str) -> None:
    global count
    lines = cache[p].split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(idx) != 1:
        raise SystemExit(f"ABORT {label}: expected 1 line, found {len(idx)}")
    lines.insert(idx[0] + 1, new_line)
    cache[p] = "\n".join(lines)
    count += 1
    print(f"  ok {label}")


# ---------------------------------------------------------------- MASTER ----
rep(M,
    "| 最后更新 | 2026-11-08（v2.31，基于 C0 的新机制证据另立「跨模态碎→整结构错配」数据前提 E40 / P1，20:28 冻结判据、未执行；零训练零评测零远端） |",
    "| 最后更新 | 2026-11-09（v2.32，按用户裁决重开 SOTA 优化路线：E41 用 62 个真实 case 重新刻画标签结构，09:29 冻结单变量对照判据、可执行组件就绪；零训练零评测零远端） |",
    "M1 version stamp")

rep(M,
    "> 覆盖 §4.1–§4.42 的全部条目（含负面、作废、预注册与未执行），共 40 次尝试 +",
    "> 覆盖 §4.1–§4.43 的全部条目（含负面、作废、预注册与未执行），共 41 次尝试 +",
    "M2 4.0 range and count")

ROW_443 = (
    "| §4.43 | 用真实 case 重新理解 task 的数学结构，把「换输出参数化 / 换损失几何」写成单变量对照实验 | "
    "本地只读探针 62 case 实测：LABEL 零值率 **0.912**、`tv_z` **0.0312** 对 CR 0.1165、`tail_max_over_p99` **4.0** 对 1.33、"
    "连通核 **108.8 px** 对 18.2；局部峰环带对比 1.03 对 1.11（**怪在幅度重尾，不在空间尖峰**，与 C0 的「更连贯」一致）；"
    "12 个复现方法共享同一「稠密 `log1p(mm)` 回归」输出参数化，`BIAS@2.0` 全在 0.035–0.435 | "
    "**`PRE-REGISTERED / COMPONENTS READY`**：B0/A0/A1/A2 五臂矩阵与 S0–S6 判据 09:29 冻结；"
    "S0 逐位等价 PASS（`max_abs_diff 0.0`）、CPU 单测 12/12；**零训练零评测零远端，无任何性能数字**；"
    "§11.9 放回结论出口门 | `docs/experiments/E41-task-structure-reparameterization-single-variable-ablation.md` |"
)
insert_after_line(M, "| §4.42 |", ROW_443, "M3 4.0 E41 row")

CARD_443 = """
### 4.43 重新理解 task：标签是「零膨胀 + 重尾 + 连通支撑」的稀疏测度，先做单变量对照

**当时想干什么（大白话）**：用户 2026-11-09 裁决把方向从「证否收尾」整体改为
「**SOTA 是第一步，并且要知道 why SOTA**；达不到就说明现有方法不适配这个数据集；
重新理解 task 有哪些数学／物理结构，找相似数学形式并迁移技术」。
用户同时要求：拿真实 case 比 radar 与 LABEL 的平滑程度，不要空口讨论；
扩大调研范围；逐条列出与目标相违背的地方；不要反复拿某个 barrier 当结论。

**拿数据说话（本地只读，62 个 case，每 case 取降水最强帧，同帧、无量纲化）**：
`tools/_struct_probe_one_case.py` → `tools/_evals/struct_probe_case.json`。
LABEL 相对 radar CR：`zero_frac` **0.912 对 0.000**、`tv_z` **0.0312 对 0.1165**（背景平滑 3.7 倍）、
`tail_max_over_p99` **4.0 对 1.33**（幅度重尾）、`kernel_px_gt_p99` **108.8 对 18.2**（强值连成 6 倍大的块）、
最强帧跨模态相关中位 **0.3749**（全样本 r = +0.17，§3.5）。
**两条如实校准**：① VIL 的 `tail_max_over_p99 = 8.32` 比 LABEL 更极端、`peak_ring_contrast = 1.79`
比 LABEL 的 **1.03** 更尖 → 不能写「只有 LABEL 有孤立针状尖峰」；
② LABEL 的极端幅度落在**连贯大块**上（C0 的 3.37× 与本探针 6× 方向一致），
C0 的 `VERDICT: C0 FAIL` 与 E39 整族舍弃**不变**。用户的疑问 1 成立，
但我此前用 C0 的绝对阈值口径（问「强降水区尺度」）去否掉相对分位口径的观察（问「幅度极端性」），
是**口径错配**，已在 E41 §1.2 订正。

**统一后的结构刻画**：LABEL ＝ 零膨胀（91 % 零）+ 平滑背景 + **幅度重尾** +
**连通大块支撑** + 与 radar 支撑错位的稀疏正测度；既不是「高频点状场」，也不是稠密高斯场。
12 个复现论文方法 + 6 个自建变体**不是 18 次独立尝试，是 1 个错误假设的 18 个变体**：
都在改编码器／时序模块，却共享单通道 `expm1` 头 + `PrecipitationLoss` 的 `log1p(mm)` 逐像素回归，
后果一致是塌向条件均值、系统性欠报（`BIAS@2.0` 0.035–0.435）。
`0.0924` 是未优化持久性基线的分数，`0.1653` 是「用真值 QPE 替换网络 QPE 且不平流」这一
**特定构造**的 Oracle 上界，**不是任何输出参数化都不能超过的证明**。

**数学形式映射（调研已扩大，不局限降水临近预报模型）**：零膨胀／重尾 → hurdle、Tweedie、
Hurdle–IMDL（arXiv:2510.20486）、SMACNP 三头（GMD 19:8895, 2026）；支撑错位／质量不守恒 →
非平衡最优传输（GeomLoss、POT、UOTS 10.1029/2023JH000112）；连通核 → clDice（CVPR 2021）、
Betti matching；平滑背景 + 大块 → RPCA／MCA；跨模态病态反演 → FNO、GenONet、QPE-BiasINR；
点观测融合 → conditional merging（Dai 等，Remote Sensing 17(15):2622, 2025）。

**单变量对照（只改一个数学形式，同骨干、同划分、评测器一行不改）**：
`B0` `simvp_v2` + `standard`；`A0` `struct_hurdle(head_mode=rain)` + `standard`（实现等价废止门）；
`A1` `struct_hurdle(head_mode=hurdle)` + `--loss hurdle`（输出参数化：occurrence × amount）；
`A2` `struct_hurdle(head_mode=rain)` + `--loss uot`（损失几何：标准损失 + 非平衡 Sinkhorn 项）；
`A3` 组合臂，仅在前两臂过 S1/S3 后启动。输出恒为 `(B,10,1,H,W)` mm 场。

**判据（2026-11-09 09:29 冻结，先写后跑）**：`S0` 同权重 `max|Δ| < 1e-6`（不满足则整个对照作废）；
`S1` 四个结构量（**首要判据**：`BIAS@2.0`、预测场 `tail_max_over_p99`、`zero_frac`、
`kernel_px_gt_p99` 方向一致，且 4 mm 档 CSI 不显著下降）；`S2` `CSI@2.0`（次要）；
`S3` **等偏置纯幅度对照**（配平后增益消失即判面积／幅度效应、该臂舍弃）；
`S4` 单变量归因（两臂都不动结构量 → 判「参数化不是瓶颈」、按 §11.8 舍弃该族）；
`S5` 不放宽条款；`S6` 统计出口（正式写「SOTA」前过 §11.9 P1–P6）。完整版见 E41 §5。

**结果 / 裁决**：**`PRE-REGISTERED / COMPONENTS READY`，无任何训练结果**。
组件已落地：`models/struct_hurdle/`、`configs/models/struct_hurdle.yaml`、
`radar_nwp_bench/metrics/struct_losses.py`、`scripts/train.py` 的 `--loss hurdle|uot`、
`tests/test_struct_hurdle.py`、`tools/_s0_equivalence.py`。
本地校验：`pytest` **12/12**、`S0 PASS`（`max_abs_diff 0.0`）、探针加 `--root/--out` 后结果
hash 逐位不变、资格门 9/9。**零训练、零推理（除 CPU 小张量单测）、零数据集评测、零远端作业**。
→ 完整数据、11 条冲突清单的逐条处置、不可再写清单与远端第一批命令草案见
`docs/experiments/E41-task-structure-reparameterization-single-variable-ablation.md`。
"""
insert_before(M, "\n---\n\n## 5. 核心代码结构", "\n" + CARD_443.strip("\n") + "\n",
              "M4 4.43 card")

SECTION_9Z = """
### 9.z v2.32 新立项（用户裁决：以 SOTA 为第一步，重新理解 task 结构）

> v2.28 五项结案、v2.30 E39 舍弃、v2.31 E40 预注册**均不变**。
> 本项是用户 2026-11-09 的**方向性裁决**，合法性来自 §11.8.2 第 1 条
> 「除非出现新的机制证据」与 §11.8.5 的「按机制族界定」：
> 62 case 本地只读探针给出了 v2.31 时不存在的标签结构证据，
> 且本次动的是**输出参数化与损失几何**（不同机制族），不是已判负 8 族的派生变体。

| 项 | 状态 | 说明 |
|---|---|---|
| **E41 / §4.43 重新理解 task 结构 → 单变量对照实验** | **`PRE-REGISTERED / COMPONENTS READY`（B0–A2 未跑）** | 判据 09:29 冻结（S0–S6）；组件全部落地并通过本地校验；**零训练、零评测、零远端、零性能数字**。§11.9 第十一项判「不适用」并给理由，不伪造 PASS |
| **算力去向** | `远端第一批命令草案已备，待用户授权执行` | S0（零训练）→ B0/A0 复现臂 → A1/A2 各 1–3 seed 探索 → 过 S1/S3 才进 A3 与 S6 正式多 seed 批次。**不再把「不投算力」写成默认结论** |
| **§11.9 的位置** | `已订正：结论出口门，不是立项／探索前置门` | 见 §11.9.2。v2.28–v2.31 把它当准入前置门是**范畴误用**；P1–P6 原文一字不改 |
| **§11.8.2 的适用范围** | `已订正：停止派生按机制族界定` | 见 §11.8.5。原文三条不改 |
| **§15 三幕** | `原文不改，另加第四幕附记` | 诊断层保留为「why」的一部分，新增 SOTA 优化路线为双轨；无 A 臂结果就重写三幕会造成状态倒挂 |
"""
insert_before(M, "\n---\n\n## 10. 已确立的结论（分层）",
              "\n" + SECTION_9Z.strip("\n") + "\n", "M5 9.z section")

NOTE_102 = """
> **v2.32 / E41 适用范围注记（C0 口径订正，不改任何既有裁决；与 §10.2 表格同等效力）**
>
> C0（E39 §10，v2.30）用**绝对阈值**问「强降水区谁更大块」，答「LABEL 更连贯」；
> 62 case 结构探针用**相对分位**问「幅度是否极端」，答「`max/p99 = 4.0` 对 CR 1.33」。
> 两者看的是不同东西，**C0 的 `VERDICT: C0 FAIL`、E39 生成式条件采样族整族舍弃不变**，
> 但不得再用它否定「LABEL 存在幅度极端的高值区域／重尾」这一事实。
>
> 新增两条不可再写（§10.2 表格原文一条不删）：
> ① **「LABEL 有孤立针状尖峰」** —— `peak_ring_contrast 1.0325` 反而低于 CR 1.1059，
> 极端值是**幅度重尾 + 空间连贯大块**（连通核 108.8 px），VIL 才是更尖的那个（1.7894）。
> ② **「`0.1653` 是任何模型都不能超过的物理天花板」** —— 它是「用真值 QPE 替换网络 QPE
> 且不平流」这一特定 Oracle 构造的上界，不是对任意输出参数化的上界证明。
"""
insert_before(M, "\n### 10.3 未解释 / 待验证",
              "\n" + NOTE_102.strip("\n") + "\n", "M6 10.2 scope note")

NOTE_1185 = """
#### 11.8.5 「停止派生」按机制族界定（v2.32 / E41，订正适用范围，**不改上文三条原文**）

§11.8.2 第 1 条禁止的是「**同一机制**的新变体」，§11.2 Gate 1 明文「按机制不按名词」。
因此：

- **同一机制族内的改写是派生**，禁止：例如幅度标定族里换一个乘数／门控形式／阈值扫描。
- **跨机制族的新形式不是派生**，允许立项：例如从「稠密 `log1p(mm)` 逐像素回归」
  换成「零膨胀混合分布的输出参数化」或「最优传输损失几何」（E41 A1 / A2）。
  判据是**改了哪一环**（§11.8.1 自评①），不是换了什么名字。
- 用户 2026-11-09 明确：「每次试错的方向不应该是帮我们排除了一个错误方向，
  仍然有大量的方向可以去结合和思考」。本条把这句话写进规则解释：
  **舍弃一个机制族 ≠ 关闭探索空间**；§9 的「已舍弃」清单是机制族清单，不是方向上限。
- 被证否后的重新立项仍须给出**新的机制证据**或**新的机制族**，并先写判据后投算力
  （§11.7 R4 / E41 §5）。E41 的新证据是 62 case 结构实测，见 §4.43。
"""
insert_before(M, "\n---\n\n### 11.9 候选方法准入的通过率资格门",
              "\n" + NOTE_1185.strip("\n") + "\n", "M7 11.8.5 note")

NOTE_1192 = """
#### 11.9.2 位置与适用范围（v2.32 / E41，**订正范畴误用，P1–P6 原文一字不改**）

> **§11.9 是「结论出口门」，不是「立项／探索前置门」。**

- 原文一句话是「任何方法若不能先复现同批次的 10 seed 基线通过率，就没有资格**谈提升**」，
  P1–P6 约束的是**报性能数字时的证据标准**：防止把训练种子噪声（峰值端离散 9.59×、
  单 seed 极差 0.1671 > 可用空间 0.0729）当成方法提升。这个标准本身正当，审稿人必问。
- 它**从未禁止**提出新假设、调研、结构诊断、设计新模型，或在拿到 10 seed 之前做探索。
  v2.28–v2.31 把它读成「没资格立项」、用「零算力 + 不适用」把回避包装成合规，
  是**范畴误用**，本轮一次性订正。
- **不受它约束**：找方向、只读数据探针、换输出参数化／损失几何、单／3 seed 探索
  （只要不宣称「提升成立」、不写「SOTA」）。
- **受它约束**：正式写「提升成立 / SOTA / 超过地板」之前，须按 P1–P6 补齐
  ≥10 独立训练种子、同批次 `qpe_0` 5/10 与 `base_concat` 4/10 及 9.59× 复现、
  通过率绑定 ep00、两级配对 bootstrap、分端差值门槛（E41 判据 S6）。
- `--check <manifest>` 与 `VERDICT` 只用于**结论出口**；纯诊断／预注册仍按 §11.9.1
  在第十一项写「不适用」并给理由，**不得伪造 `VERDICT: PASS`**。
"""
insert_before(M, "\n---\n\n## 12. 变更日志",
              "\n" + NOTE_1192.strip("\n") + "\n", "M8 11.9.2 note")

CHANGELOG = (
    "| 2026-11-09 | **v2.32** | **按用户裁决重开 SOTA 优化路线：重新理解 task 结构 → 数学形式映射 → 单变量对照实验（E41 / §4.43，09:29 判据先写后跑、组件就绪、训练算力未投）**："
    "① 用户要求「SOTA 且知道 why SOTA、达不到说明现有方法不适配数据集、扩大调研范围、逐条列出与目标相违背处、不要反复拿 barrier 当结论」；"
    "② 本地只读探针 62 个真实 case 给出新机制证据：LABEL `zero_frac 0.912`、`tv_z 0.0312` 对 CR 0.1165、"
    "`tail_max_over_p99 4.0` 对 1.33、连通核 108.8 对 18.2 px、局部峰环带对比 1.03 对 1.11、"
    "最强帧跨模态相关 0.3749 → **怪在幅度重尾 + 空间连贯大块，不是孤立针状尖峰**；"
    "③ 三处订正：**§11.9 放回结论出口门**（§11.9.2，P1–P6 原文不改）、"
    "**§11.8.2 按机制族界定**（§11.8.5，三条原文不改）、**C0 绝对阈值口径错配**（§10.2 注记，C0 verdict 与 E39 舍弃不变）；"
    "④ 落地 `models/struct_hurdle/`（rain 模式与 `simvp_v2` 逐位等价、hurdle 模式 occurrence × amount）、"
    "`radar_nwp_bench/metrics/struct_losses.py`（hurdle NLL + 非平衡 Sinkhorn）、`--loss hurdle|uot`、"
    "`configs/models/struct_hurdle.yaml`、`tests/test_struct_hurdle.py`、`tools/_s0_equivalence.py`；"
    "⑤ 校验：`pytest` 12/12、**S0 PASS `max_abs_diff 0.0`**、探针加 `--root/--out` 后结果 hash 逐位不变、"
    "`_gate_11_9.py --self-test` 9/9、表格 mismatched=0；"
    "⑥ **零训练零推理（除 CPU 小张量单测）零数据集评测零远端作业零性能数字**；"
    "§15 三幕原文不改，另加第四幕附记；11 条冲突清单中 3 条已订正、8 条待用户裁决 | "
    "`docs/experiments/E41-task-structure-reparameterization-single-variable-ablation.md` · "
    "`tools/_evals/struct_probe_case.json` · `tools/_sync_v232.py` |"
)
insert_after_line(M, "| 2026-11-08 | **v2.31** |", CHANGELOG, "M9 changelog v2.32")

ACT4 = """
### 第四幕 · v2.32 方向重定位（附记，**上方三幕原文不改**）

> 用户 2026-11-09 裁决：目标改为「**SOTA 是第一步，并且知道 why SOTA**；
> 若做不到，结论是现有方法不适配这个数据集，而不是任务有天花板」。
> 本附记把研究线改为**双轨**，不删、不否定前 3 幕的任何证据与结论。

- **诊断层（保留，作为 why 的一半）**：地板 0.0924、Oracle 上界 0.1653、
  单 seed 极差 0.1671 是可用空间 0.0729 的 2.29 倍、§11.9 的证据标准 ——
  这些继续回答「为什么领域里多数单点提升报告在统计上不可识别」。
- **优化层（v2.32 新增，作为 why 的另一半）**：12 个复现方法共享同一个
  「稠密 `log1p(mm)` 回归」输出参数化，而标签是零膨胀 + 幅度重尾 + 连通支撑的
  稀疏正测度（§4.43 的 62 case 实测）。**中介链**是：
  识别标签分布 → 指出 baseline 的参数化错配 → 用 hurdle × UOT × 拓扑逐项修复
  零值率／尾部／连通核／BIAS → `CSI@2.0` 随之提升。
  这条链**尚未验证**，由 E41 的 S1（结构指标优先）与 S3（等偏置纯幅度对照）检验；
  过不了就按 §11.8 舍弃该族，转而检验 §4.43 表中拓扑／低秩+稀疏／算子学习三支。
- **paper positioning**：即使纯诊断稿被质疑 "just a negative result"，
  v2.32 起稿子不再是 negative result —— 它有一个可证伪的机制解释和一个
  可迁移的**统计资格协议**（§11.9）作为方法主体，模型只当 case study。
  **但在 B0–A2 跑完并过 S3 之前，不得写「已实现 SOTA」「已解决 novelty」**
  （§10.2 注记 / E41 §9）。
"""
insert_before(M, "\n---\n\n## 16. 绝境迭代方法论",
              "\n" + ACT4.strip("\n") + "\n", "M10 15 fourth act")

# ----------------------------------------------------------------- INDEX ----
INDEX_ROW = (
    "| E41 | §4.43 | 用 62 个真实 case 重新理解 task 的数学结构，把「换输出参数化 / 换损失几何」写成单变量对照实验 | "
    "本地只读结构探针（62 case 实测）+ 代码事实 + 预注册（**零训练零评测零远端，判据 09:29 冻结**） | "
    "`E41-task-structure-reparameterization-single-variable-ablation.md` | "
    "**`PRE-REGISTERED / COMPONENTS READY`**：B0/A0/A1/A2 五臂与 S0–S6 判据冻结；"
    "S0 逐位等价 PASS（0.0）、CPU 单测 12/12；组件全部落地、训练算力未投；"
    "§11.9 第十一项判「不适用」并订正其位置为结论出口门 |"
)
insert_after_line(I, "| E40 |", INDEX_ROW, "I1 INDEX E41 row")

# ---------------------------------------------------------------- README ----
rep(R, "(v2.31, 2026-11-08)", "(v2.32, 2026-11-09)", "R1 status stamp")
rep(R, "As of **MASTER v2.31**", "As of **MASTER v2.32**", "R2 status version")
rep(R, "all 40 registered attempts", "all 41 registered attempts", "R3 attempt count")

README_ADD = """
>
> **v2.32 (2026-11-09) reopens a SOTA optimisation route** (user adjudication).
> A read-only local probe on 62 real cases (`tools/_evals/struct_probe_case.json`)
> re-characterises LABEL as a zero-inflated (0.912 zeros), amplitude-heavy-tailed
> (`max/p99` 4.0 vs radar CR 1.33), spatially coherent (connected core 108.8 px
> vs 18.2) sparse positive measure whose support is displaced from radar.
> The 12 reproduced baselines share ONE dense `log1p(mm)` regression
> parameterisation, so they are 18 variants of one wrong assumption rather than
> 18 independent attempts. E41 (§4.43) pre-registers a single-variable ablation
> matrix (B0/A0/A1/A2) over output parameterisation (hurdle NLL) and loss
> geometry (unbalanced Sinkhorn), with criteria frozen at 09:29 **before** any
> training. Executable components are in place; `pytest` 12/12 and the S0
> bit-equivalence gate PASS (max_abs_diff 0.0). **No training, no evaluation,
> no remote work, no CSI number, no SOTA claim.** MASTER §11.9 is clarified as
> a conclusion-exit gate (not an entry gate); §11.8.2 is scoped by mechanism
> family. The three-act narrative in MASTER §15 is kept verbatim, with a fourth
> act added as an addendum. See `docs/MASTER.md` §4.43 / §9.z / §11.9.2.
"""
rep(R,
    "> no local test run backs a corrected number. Do not cite them as evidence.\n",
    "> no local test run backs a corrected number. Do not cite them as evidence.\n"
    + README_ADD.strip("\n") + "\n",
    "R4 v2.32 status paragraph")

# ------------------------------------------------------------------ write ----
for p, text in cache.items():
    t = text.replace("\r\n", "\n")
    p.write_bytes(t.replace("\n", "\r\n").encode("utf-8"))
print(f"SYNC: PASS ({count} replacements, {len(cache)} files, CRLF kept)")
