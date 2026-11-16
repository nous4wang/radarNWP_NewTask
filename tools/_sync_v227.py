#!/usr/bin/env python3
"""One-shot, verified sync of MASTER.md / INDEX.md to v2.27 (E37).

Every replacement asserts an exact single match before writing, so a missed
anchor aborts the whole sync instead of silently producing a half-updated
document.  Keeps the repository's CRLF convention.  No training, no inference.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MASTER = ROOT / "docs" / "MASTER.md"
INDEX = ROOT / "docs" / "experiments" / "INDEX.md"

E37 = "`docs/experiments/E37-physics-root-cause-two-packages-gate-review.md`"
E37_SHORT = "`E37-physics-root-cause-two-packages-gate-review.md`"

REPLACEMENTS: list[tuple[pathlib.Path, str, str, str]] = []

# ---------------------------------------------------------------- M1 stamp
REPLACEMENTS.append((MASTER,
"| 最后更新 | 2026-11-07（v2.26，§9 3d/3e 收口 + §11.9 通过率资格门 + E35/E36 与 GitHub 方法准入审查） |",
"| 最后更新 | 2026-11-07（v2.27，两份物理根因方案审查 + §11.9 资格门生效 + §15 三幕按已证否事实重写） |",
"M1 version stamp"))

# ------------------------------------------------------------ M2 section 4.0
REPLACEMENTS.append((MASTER,
"""> 覆盖 §4.1–§4.38 的全部条目（含负面、作废、预注册与未执行），共 36 次尝试 +
> 2 个未执行的预留编号：§4.1–§4.24 的 23 次为 v2.16 迁出，§4.26–§4.38 十三次按
> §11.7 预注册，§4.33–§4.35 于 2026-11-07 12:05–13:35 预注册、§4.36 于 15:31 预注册、
> §4.37 / §4.38 于 16:30 预注册，""",
"""> 覆盖 §4.1–§4.39 的全部条目（含负面、作废、预注册与未执行），共 37 次尝试 +
> 2 个未执行的预留编号：§4.1–§4.24 的 23 次为 v2.16 迁出，§4.26–§4.39 十四次按
> §11.7 预注册，§4.33–§4.35 于 2026-11-07 12:05–13:35 预注册、§4.36 于 15:31 预注册、
> §4.37 / §4.38 于 16:30 预注册、§4.39 于 17:24 预注册，""",
"M2 section 4.0 header"))

# ------------------------------------------------------- M3 section 4.0 row
REPLACEMENTS.append((MASTER,
"| `docs/experiments/E36-split-ensemble-proxy.md` |\n\n### 逐次尝试（大白话卡片）",
"""| `docs/experiments/E36-split-ensemble-proxy.md` |
| §4.39 | 统一审查两份「物理根因」方案包，同时把通过率资格门变成能跑的门禁、把论文三幕按已证否的事实重写 | 两份方案去重后是 5 类方向，【有真实可能成功】档为空：动态均值场偏差校正就是 E27（`tools/_dynamic_mfb.py` 已存在，4 mm 档 −0.0050 CI 严格为负、不可部署）；27 个 NWP 变量只有 `WS500/WS700/WS925` 风速标量、没有 U/V 风矢量；`x_radar` 是 11 帧 × 10 变量而非 t₀ 单时刻 10 层；3d / 3e 已封口；概率化不引入新观测信息。另订正一处历史笔误：逐 seed 配对差是 **4 正 6 负**（原记 5 正 5 负），通过门槛未改。 | **零算力审查，5 类方向全部舍弃或判「条件缺失不立项」**；唯一采纳「不造没有物理信息源的算子」与「单变量受控 + 真实通道参数配平」两条原则，并入 §11.2。§11.9 已生效（门禁 9/9 锚点复现 + 第十一项必填字段），§15 三幕整体重写。 | """ + E37 + """ |

### 逐次尝试（大白话卡片）""",
"M3 section 4.0 row"))

# ---------------------------------------------------------- M4 section 4.39
CARD = """### 4.39 统一审查两份「从物理根因出发」的方案包，并把资格门与论文叙事落地

**当时想干什么（大白话）**：用户给来两份都自称从物理／数理根因开始的方案，合起来 7 个方向；要我统一审查、说明跟任务目标的关系，再决定怎么推进。同一轮还要求两件落地：把 §11.9 从文字规则变成真正能检查的门禁，把 §15 三幕按已经被证否的事实重写。

**结果怎么样**：两份方案**不是两条正交路线，而是同一组三个想法的两份措辞**，去重后 5 类。共同的诊断前提是对的（不引入新观测信息，确定性网络补不出缺失的高频排布），但三个「新信息源」逐一落空：① **个例级动态均值场偏差校正就是 E27** —— `tools/_dynamic_mfb.py:2` 逐字写着「E27 per-window dynamic mean-field bias scalar (oracle)」，用历史地面雨量**真值**逐窗口现算，4 mm 档 **−0.0050 CI 严格为负**、`BIAS@2.0` 1.8248 → 0.8045，是逐窗口 oracle、不可部署，§10.2 已把「在线动态 MFB 可用」列为被推翻；② **低层风矢量根本不在数据里** —— `radar_nwp_bench/data/index.py:19–24` 的 27 个 NWP 变量只有 `WS500/WS700/WS925` 三个风速**标量**，没有 U/V 分量、没有 850 hPa 风分量，`Δx = V_wind × τ` 无法定向；而且 `x_radar (11,10)` 是 **11 个时间帧 × 10 变量**，历史帧本来就在输入里（§4.16.3 用满 11 帧最好 0.1564），方案把接口读反了；③ **概率／不确定性输出不引入任何新观测信息**，只是把「不知道」表达得更诚实。另外两处硬错：方案一「方向三」写的是**修复前**的 P3 代码（`5e184d1` 早已修复、E35 已封口），并宣称「10 seed × 3 epoch 是里程碑式基准」，而 E13 的 96 份评测 + 两级 bootstrap 早就完成；方案二要在 val 上扫概率阈值，但十份 float32 场只在 **test 的 538 窗**（E26 第 25 行），**val 无场**，零算力前提不成立。**【有真实可能成功】档为空**。副产物：门禁 `--self-test` 首跑 `ANCHOR: FAIL`（8/9），追查发现 §4.13 / §11.9 / E13 / E35 把逐 seed 配对差的方向计数写成了「5 正 5 负」，而既有分析器输出本就是 **4 正 6 负**（均值 −0.0063、通过率、Fisher p 全一致）→ 按数据订正，**通过门槛一字未改**，复测 **9/9 PASS**。

**裁决**：**零算力审查，5 类方向全部舍弃或判「条件缺失不立项」**，不投训练、不装依赖、不跑远端作业。唯一采纳的是两份方案共有的**方法论原则**——「不造没有物理信息源的算子」与「单变量受控 + 真实通道参数配平」，并入 §11.2 Gate 1。**§11.9 已生效**：新增 `tools/_gate_11_9.py`（锚点自测 + `--check` 出 `VERDICT`）、`TEMPLATE.md` / `README.md` 第十一项必填字段、§11.9.1 生效机制与一次性历史合规普查。**§15 三幕已整体重写**（用户裁决执行）。

**证据层级**：代码事实 + git 历史 + 既有远端 CUDA 评测产物的本地复算 + 文档一致性核查；不是新的数据集结果，不写入 §10.1 性能层。

**自评（§11.8）**：**有关** —— 动的是证据准入、算力去向与结论表述层。**证据**：`_dynamic_mfb.py:2`、§4.29 的 −0.0050、`data/index.py:19–24`、`5e184d1`、E13 的 96 份评测、E26 第 25 行「538 个 test 窗口」、E36 的 CSI–BIAS **+0.9774** 与 **0.039627 = 2.76 × 增益**。**去往**：**保留并推进**（资格门、三幕、两条原则）+ **舍弃**（5 类方向）。**不再做什么**：不再以「个例自适应」重跑 E27；不在没有风矢量时写漂移／落区对齐模块；不再为 P3 重训重评测；不再设计 PoE／证据回归／概率阈值门控；不再把 BIAS 漂移写成 ep0 崩塌的因果；不再用正确的物理前提反推某个确定性模块有效。

→ 完整去重表、七条硬事实 F1–F7、三档分级、若条件补齐的判据草案、门禁与合规普查、不可再写清单见 """ + E37 + """。
"""
REPLACEMENTS.append((MASTER,
"→ 完整十次划分表、三个可引用的数、正式判据与复现命令见 `docs/experiments/E36-split-ensemble-proxy.md`。\n\n---\n\n## 5. 核心代码结构",
"→ 完整十次划分表、三个可引用的数、正式判据与复现命令见 `docs/experiments/E36-split-ensemble-proxy.md`。\n\n---\n\n" + CARD + "\n---\n\n## 5. 核心代码结构",
"M4 section 4.39 card"))

# ------------------------------------------------------ M5 section 9 discard
DISCARD = """| 个例级／在线动态均值场偏差校正（Dynamic MFB）及其「自适应」变体；把 train/test BIAS 漂移当成 ep0 崩塌因果并据此立项 | 就是 E27（工具已存在、用真值、不可部署、4 mm 档显著为负），§10.2 已推翻；§4.26 明令禁止因果外推 | `tools/_dynamic_mfb.py:2`；§4.29 4 mm 档 −0.0050 CI [−0.0085,−0.0016] 严格为负；§4.26 删 top-3 只缩 5.7 % | """ + E37_SHORT + """ §2 F1–F2 |
| 风漂移／粒子下落时滞对齐的模型层落地；PoE／深度证据回归／最优概率阈值门控；val 阈值扫描及其派生变体 | 27 个 NWP 变量没有 U/V 风矢量（条件缺失，§11.2）；概率化不引入新观测信息；门控族 §4.22 显著为负；十份场只在 test 的 538 窗、val 无场；分档投票已由 E36 舍弃 | `data/index.py:19–24`；§4.15.3 平流负增益；§4.22 −0.0347 CI 严格为负；E26 第 25 行；E36 CSI–BIAS +0.9774 | """ + E37_SHORT + """ §2 F3/F5/F6 |"""
REPLACEMENTS.append((MASTER,
"| `E36-split-ensemble-proxy.md` |\n\n---",
"| `E36-split-ensemble-proxy.md` |\n" + DISCARD + "\n\n---",
"M5 section 9 discard rows"))

# -------------------------------------------------------- M6 section 10.2
REJECT = """| **「个例级／在线动态均值场偏差校正（Dynamic MFB）能规避全局常数的死穴，是可部署方法」** | 就是 E27（`tools/_dynamic_mfb.py:2`）：用历史地面雨量真值逐窗口现算，是逐窗口 oracle；4 mm 档 **−0.0050 CI [−0.0085,−0.0016] 严格为负**、`BIAS@2.0` 1.8248 → 0.8045；三种时滞差 < 0.0013。§4.29 已判不可部署、不写入 §10 | §4.29 / """ + E37_SHORT + """ §2 F1 |
| **「train 1.7820 / test 0.9048 的标定漂移解释了第 0 轮触顶崩塌，因此应当做动态纠偏」** | §4.26 实测三划分中位 1.888 → 1.513 → 1.228、删 train top-3 只缩 5.7 %，判为整体系统性平移；**无干预、无逐轮配对，因果链未钉死**。E27 是零训练离线重打分，根本没有 epoch 维度 | §4.26 / """ + E37_SHORT + """ §2 F2 |
| **「`x_nwp` 27 通道里有 850/700 hPa 的 U/V 风场，可直接做粒子下落漂移对齐」** ／ **「`x_radar` 是 t₀ 单时刻的 10 个高度层」** | `radar_nwp_bench/data/index.py:19–24` 只有 `WS500/WS700/WS925` 三个风速**标量**，无 U/V 分量、无风向、无 850 hPa 风分量；`index.py:14–17` 的 10 个雷达变量 × §1.2 的 **11 个时间帧**，历史帧本就在输入里（§4.16.3 用满 11 帧最好 0.1564） | §1.2 / """ + E37_SHORT + """ §2 F3 |
| **「PoE／深度证据回归／最优概率阈值能补上 E33 的 61 % 信息缺口」** ／ **「在 val 上扫 p* 是零算力实验」** | 概率／不确定性输出**不引入新观测信息**，属表述层；「仅在 P ≥ p_th 激活增益」是逐像素门控 × 标量调制，与 §4.22（−0.0347 CI 严格为负）同族；十份 float32 场只在 **test 的 538 窗**（E26 第 25 行），**val 无场**；分档投票已按 E36 舍弃（CSI–BIAS +0.9774） | §10.1bis / §4.22 / """ + E37_SHORT + """ §2 F5–F6 |"""
REPLACEMENTS.append((MASTER,
"| §4.38 / E36 |\n\n### 10.3 未解释 / 待验证",
"| §4.38 / E36 |\n" + REJECT + "\n\n### 10.3 未解释 / 待验证",
"M6 section 10.2 reject rows"))

# ---------------------------------------------------------- M7 section 11.2
REPLACEMENTS.append((MASTER,
"""⚠️ **"换个说法"是最常见的坑**。本项目已有两次实例：
逐像素非对称 BCE 与分层损失是同一个东西（第二次实测失败）；
"极限切片开采"的前提（样本被稀释）经实测不成立。

### 11.3""",
"""⚠️ **"换个说法"是最常见的坑**。本项目已有两次实例：
逐像素非对称 BCE 与分层损失是同一个东西（第二次实测失败）；
"极限切片开采"的前提（样本被稀释）经实测不成立。

**v2.27 补充两条准入原则**（E37 两份外部方案里唯一采纳的部分）：
① **不造没有物理信息源的算子**：若某模块不引入新的观测量或可核验的因果先验，它就只能在
条件期望 `E[Y|X]` 内打转；把 E33 用真值量出的**信息缺口**当成可训练模块能补回的
**优化缺口**即属此类（§10.1bis / §10.2）。② **单变量受控 + 真实通道参数配平**：对照只差
一个变量，参数量须按代码里的真实通道数核算（E34 的 +14.7 % 就是反面例子）。
两条都只是 Gate 1 的排查项，**不构成任何方向的准入**。

### 11.3""",
"M7 section 11.2 principles"))

# ------------------------------------------------------- M8 section 11.9.1
GATE_MECH = """#### 11.9.1 生效机制（v2.27 / E37，强制）

> §11.9 在 v2.26 只是文字；**v2.27 起它是可执行门禁**，不满足就没有资格谈提升。

- **工具**：`tools/_gate_11_9.py`。`--self-test` 从 96 份评测副本复现 5/10、4/10、
  Fisher 单侧 p = 0.5、9.59×、配对差 −0.0063（**4 正 6 负**）与 ep00/ep01/ep02 三组轮次
  通过率；`--check <manifest>` 逐条校验 P1–P6 并输出 `VERDICT: PASS / FAIL / PARTIAL`。
  常量（0.0924 / 0.1653 / 0.0729 / 1.78× / 9.6× / ≥10 seed）写死在脚本里，
  冻结时间 **2026-11-07 17:24**，早于本轮任何算力。
- **流程**：`docs/experiments/TEMPLATE.md` 与 `README.md` 的**第十一项字段「§11.9 资格门核对」**
  为必填；纯审查／文档／诊断类须写「不适用」并说明为什么没有性能主张；
  **缺失或无理由的「不适用」→ 该尝试不得进入 §10.1 性能层**。
- **历史合规普查**（E37 §5.3，一次性）：E13 的 96 份评测是锚点来源，保持 §10.1；
  E19 等单 seed 峰值类按 P1 只能留诊断层；E17/E18/E21/E24/E27/E29/E30/E32/E33/E36 等
  零算力／离线后处理类按 P4 不进性能层；§4.23 / §4.25、E36-A/B 等预注册未执行项状态不变。
  **新门不追溯作废任何历史结论**（§13 以 §10.2 为准），但自 v2.27 起新建详情文件一律必填。
- **订正留痕**：首次自测 8/9，唯一失败项是历史文档把配对差方向计数写成「5 正 5 负」；
  既有分析器 `tools/_analyze_seeds_final.py` 的输出本就是 **4 正 6 负**（均值 −0.0063 一致）
  → 按数据订正描述值，**通过门槛一字未改**，复测 **9/9 PASS**；「两臂峰值端不可分辨」不变。
"""
REPLACEMENTS.append((MASTER,
"""**基线锚点与复算**：`tools/_evals/route_seeds/evals/` 96 份评测（数据源为远端 CUDA），`tools/_analyze_seeds_final.py tools/_evals/route_seeds` 一行复算；完整同批次通过率表见 `E35-open-items-closure-and-github-survey.md` §4.37.3。

---

## 12. 变更日志""",
"""**基线锚点与复算**：`tools/_evals/route_seeds/evals/` 96 份评测（数据源为远端 CUDA），`tools/_analyze_seeds_final.py tools/_evals/route_seeds` 一行复算；完整同批次通过率表见 `E35-open-items-closure-and-github-survey.md` §4.37.3。

""" + GATE_MECH + """
---

## 12. 变更日志""",
"M8 section 11.9.1"))

# ---------------------------------------------------------- M9 changelog
CHANGELOG = """| 2026-11-07 | **v2.27** | **两份「物理根因」方案包统一审查 + §11.9 资格门生效 + §15 三幕整体重写（E37 / §4.39；零训练零推理零远端作业，判据 17:24 先写）**：① 两份方案去重为 5 类方向，【有真实可能成功】档**为空**并全部舍弃／判「条件缺失不立项」：动态 MFB ＝ E27（`tools/_dynamic_mfb.py:2`，用真值、4 mm 档 −0.0050 CI 严格为负、不可部署）；`NWP_VARS` 27 个变量只有 `WS500/WS700/WS925` 风速标量、无 U/V 风矢量（`data/index.py:19–24`）；`x_radar` 是 11 帧 × 10 变量、不是 t₀ 单时刻 10 层；3d / 3e 已由 E35 封口且方案读的是修复前代码；概率／不确定性不引入新观测信息，val 上也没有十份场（E26 第 25 行「538 个 test 窗口」）。② **§11.9 生效**：新增 `tools/_gate_11_9.py`（`--self-test` 复现锚点、`--check` 出 `VERDICT`）、`TEMPLATE.md` / `README.md` 第十一项必填字段、§11.9.1 生效机制与一次性历史合规普查；常量冻结 17:24。③ **§15 三幕按已证否事实整体重写**（用户裁决执行），禁「破局」类表述，承重数字全部可回溯 §10；§14 保持立项记录、不作论文主线。④ **事实订正**：`--self-test` 首跑 8/9，查出 §4.13 / §11.9 / E13 / E35 / §12 v2.7 把逐 seed 配对差方向写成「5 正 5 负」，既有分析器输出本就是 **4 正 6 负**（均值 −0.0063 一致）→ 按数据订正，**通过门槛一字未改**，复测 **9/9 PASS**；结论「两臂不可分辨」不变。⑤ 同步 §4.0（36 → 37 次）/ §4.39 卡片 / §9 舍弃清单 +2 条 / §10.2 +4 条 / §11.2 补两条准入原则 / `INDEX.md`；新增 """ + E37 + """、`tools/_gate_11_9.py`、`tools/_evals/gate_11_9_example.json`。⑥ **集体自评（§11.8.3）**：一次尝试（E37），**有关**（证据准入、算力去向、结论表述层），证据为 `_dynamic_mfb.py:2`、`data/index.py:19–24`、`5e184d1`、E13 的 96 份评测、E26 第 25 行、E36 的 +0.9774 与 2.76×；去向 **保留并推进**（资格门／三幕／两条原则）+ **舍弃**（5 类方向），并写明「不再做什么」。 | E37 详情文件、`docs/experiments/TEMPLATE.md` / `README.md`、`INDEX.md`、`tools/_gate_11_9.py`、`tools/_evals/gate_11_9_example.json`；代码事实核查 `radar_nwp_bench/data/index.py`、`tools/_dynamic_mfb.py`、`git show 5e184d1`；本地纯算术（96 份评测副本复算） |"""
REPLACEMENTS.append((MASTER,
"---\n\n## 13. 冲突裁定规则",
CHANGELOG + "\n\n---\n\n## 13. 冲突裁定规则",
"M9 changelog v2.27"))

# ------------------------------------------------------------- I1 INDEX row
REPLACEMENTS.append((INDEX,
"| E36 | §4.38 | 把十个种子拆成两组各自平均，先写死正式判据再跑零算力代理 | 数据集实测（十份既有 float32 场的离线算术，零训练零推理）+ ≥20 seed 判据预注册 | `E36-split-ensemble-proxy.md` | 已执行（代理完成；正式实验留档待裁决） |",
"""| E36 | §4.38 | 把十个种子拆成两组各自平均，先写死正式判据再跑零算力代理 | 数据集实测（十份既有 float32 场的离线算术，零训练零推理）+ ≥20 seed 判据预注册 | `E36-split-ensemble-proxy.md` | 已执行（代理完成；正式实验留档待裁决） |
| E37 | §4.39 | 统一审查两份「物理根因」方案包，让 §11.9 资格门生效，并按已证否事实重写 §15 三幕 | 代码事实 + git 历史 + 96 份既有评测复算 + 文档一致性核查（零训练零推理零远端作业） | """ + E37_SHORT + """ | 已执行（5 类方向全部舍弃；资格门生效；三幕重写；订正配对差计数） |""",
"I1 INDEX E37 row"))


def main() -> int:
    cache: dict[pathlib.Path, str] = {}
    for path, old, new, label in REPLACEMENTS:
        if path not in cache:
            cache[path] = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        text = cache[path]
        count = text.count(old)
        if count != 1:
            print(f"ABORT {label}: expected exactly 1 match, found {count}")
            print(f"  anchor first 120 chars: {old[:120]!r}")
            return 1
        cache[path] = text.replace(old, new, 1)
        print(f"OK    {label}: 1 match replaced (+{len(new) - len(old)} chars)")

    for path, text in cache.items():
        data = text.replace("\n", "\r\n").encode("utf-8")
        path.write_bytes(data)
        print(f"WROTE {path.relative_to(ROOT)}: {len(data)} bytes, "
              f"{len(text.splitlines())} lines, CRLF")
    print(f"SYNC: PASS ({len(REPLACEMENTS)} replacements, {len(cache)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
