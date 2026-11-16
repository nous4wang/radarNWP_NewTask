"""One-shot atomic sync for MASTER v2.28 / E38 open-items adjudication.

Every replacement must match exactly once; otherwise the whole run aborts before
writing anything. Keeps each file's original EOL style (CRLF for docs).

Zero compute: this script only edits Markdown. No training / inference / remote.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "docs/MASTER.md"
INDEX = ROOT / "docs/experiments/INDEX.md"
README = ROOT / "README.md"
E38 = ROOT / "docs/experiments/E38-open-items-final-adjudication.md"


def load(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def eol_of(path: Path) -> str:
    head = path.read_bytes()[:8192]
    return "\r\n" if b"\r\n" in head else "\n"


def save(path: Path, text: str) -> None:
    eol = eol_of(path)
    data = text.replace("\r\n", "\n").replace("\n", eol).encode("utf-8")
    path.write_bytes(data)


CARD_440 = """---

### 4.40 把 §9 最后五项悬空裁决一次性结案

**当时想干什么（大白话）**：主控文档 §9 还挂着 5 处「待用户裁决」，而 README 是权威优先级最末的派生展示层。用户裁决：先在唯一权威入口里把这些悬空项按既有数字证据全部关掉，再让 README 抄一份不会返工的事实；同时对每一项给出明确去向。

**结果怎么样**：5 项全部结案，零算力。**3g 舍弃** —— 单帧神经反演相对 test 调优的单一乘数 2.0 只 `+0.0027`、区间 `[−0.0022,+0.0077]` 含零，补 10 个种子只能再测一次标量等价性，按 §11.9 P2 还须先复现同批次 5/10、4/10、9.59× 基线。**3j 直接舍弃** —— 与「瓶颈在反演的空间结构、不在外推」及平流负增益（不动 0.1653 → 位移越大 0.1432 → 0.1303）方向相反，且默认权重路径不存在会静默回退随机初始化、单帧训出的反演网络逐帧套到 11 帧是分布外调用。**3l 维持否决** —— 27 个数值预报变量里没有 CIN/DCAPE，那个式子只是门控残差，没有密度矩阵、反交换子或保迹约束。**E36-A/B 用户裁决不投算力、判据原样留档** —— 5+5 代理只有 2/10 划分两组都过地板，组间差是增益的 2.76 倍、与偏置相关 +0.9774；**A4-B 仍是「登记未测」，不是舍弃**。**待办 10 维持「数据条件缺失，不立项」**。用户论证的三处事实也已校准：规定写入顺序的是**头部第 4 行**而非 §13（§13 只管冲突时谁优先）；版本按只增不改取 **v2.28**（当前 v2.27，v2.25 / v2.26 都是历史版本）；E38 建立后计数 37 → 38，但「全量闭环结案」被收窄 —— §9 现已无待裁决项，A4-B 与 §10.3 的未检验条目仍在。

**裁决**：**C1–C4 全 PASS，本轮不投任何算力**；§9 待办队列进入「**无待用户裁决项**」终态，§4.23 / §4.25 两个预留编号随 3j / 3l 一起结案。**「无悬空裁决项」不等于「所有实验已全量闭环」** —— README 状态声明按此收窄，且不擅自改正文里互相矛盾的 205 / 305 / 413 / 336 等历史测试计数。

**证据层级**：既有实测数字 + 代码事实 + 文档一致性核查；不是新的数据集结果，不写入 §10.1 性能层。

**自评（§11.8）**：**有关** —— 动的是权威源的状态完整性与算力去向。**证据**：`+0.0027 CI 含零`、0.1653→0.1432→0.1303、`qpe_ckpt_path` 静默回退、`data/index.py:19–24` 的 27 变量、E36 的 2/10 与 `0.039627 = 2.76 × 增益` 与 `+0.9774`、头部第 4 行。**去往**：**保留并推进**（结案状态、§4.40 卡片、索引去向、README 权威入口）+ **舍弃**（3g / 3j / 3l 及其派生）。**不再做什么**：不再补 10 seed 或写「升级后可能更可靠」；不再设计两阶段冻结 QPE；不再写 Lindblad 耗散叙事；不再为 E36 投 20 seed 或派生加权 / 裁剪 / 阈值变体；不在没有第二站与跨季节样本时把泛化列为可投算力待办；不再把历史「待裁决」描述当成当前状态。

→ 完整五项裁决表、用户论证三处事实校准、悬空字样清零核对、派生层处理与复现清单见 `docs/experiments/E38-open-items-final-adjudication.md`。
"""

DISCARD_ROWS = """| `PhysicalQPEModel` 补 10 seed / 两阶段冻结 QPE（`SimVPv2TwoStage`）/ Lindblad 耗散门控，以及三者的派生变体 | 3g 已证与单一标量不可分辨；3j 与上界结论方向相反且实验设计有两处静默失效；3l 无所需要物理变量、代码只是门控残差 | §4.19.3 `+0.0027 CI [−0.0022,+0.0077]`；§4.15 0.1653→0.1432→0.1303；§4.20.1.1(b) 默认 ckpt 路径不存在、11 帧 OOD；`data/index.py:19–24` 27 变量无 CIN/DCAPE；§4.20.1.1(c) 门控残差 | `E38-open-items-final-adjudication.md` §2 |
| 为 E36-A/B 正式 20-seed 集成投算力；把「§9 仍有待裁决项」或「所有 38 项已全量闭环结案」写进项目状态 | 5+5 代理负期望；前者已在 v2.28 清零，后者与 A4-B「登记未测」及 §10.3 未检验条目冲突 | E36 只有 **2/10** 划分两组都过地板、`0.039627 = 2.76 × 增益`、CSI–BIAS **+0.9774**；A4-B 要求 ≥20 独立种子（§11.7 R4） | `E38-open-items-final-adjudication.md` §1.3 / §2 |"""

NOT_WRITE_ROWS = """| **「`PhysicalQPEModel` 补 10 seed 后可能显示空间结构改进 / 仍是一条待验证的弱阳性路线」** | §4.19.3 与 test 调优标量 `k=2.0` 配对 **+0.0027 CI [−0.0022,+0.0077] 含零**，另三档标量显著更好；按 §11.9 P2 补跑还须先复现同批次 5/10、4/10、9.59× 基线。v2.28 / E38 已按 §11.8 舍弃 | §4.19.3 / E38 §2 |
| **「两阶段冻结 QPE 值得作为与 Oracle 结论并列的候选」** | §4.15 实测任何平流都更差（不动 **0.1653 → 0.1432 → 0.1303**）；§4.20.1.1(b) `qpe_ckpt_path` 默认路径不存在会静默回退**随机初始化 QPE**，v1 只训练过 `x_radar[:, -1]`，逐帧套 11 帧是分布外调用。v2.28 / E38 已舍弃 | §4.15 / §4.20.1.1(b) / E38 §2 |
| **「§9 还有若干项待用户裁决，项目状态未定」** ／ **「所有 38 项实验与假设已全量闭环结案」** | E38 C2 已把**当前状态**里「待用户裁决 / 须用户裁决 / 待裁决准入」清零；但 **A4-B 仍「登记未测」**、§10.3 机制推测仍标「未做直接检验」→ 两种绝对说法都不成立。§12 v2.26 行与 §4.37 卡片里的同类表述是**历史描述** | §9 / §10.3 / E38 §1.3 |
| **「README 里的模型数 / 测试数可作为结论依据」** | §13 第 8 位明确标注 README **已过时**；正文 8 baselines / 205 / 305 / 413 / 336 tests / nineteen baselines 互不一致，v2.28 未做本地复测、不凭猜测修订。权威计数与结论一律取 MASTER §10 / §12 | §13 / E38 §4 |"""

CHANGELOG_ROW = """| 2026-11-07 | **v2.28** | **§9 最后五项悬空裁决一次性结案 + 派生层按权威事实更新（E38 / §4.40；零训练零推理零远端作业，判据 17:52 先写）**：① 用户裁决把「§9 结案」提到 README 之前；Gate 1 核对后**采纳顺序结论**，但校准三处事实 —— 明文规定写入顺序的是**头部第 4 行「任何产出先落到本文件，再落到别处」**（§13 只规定冲突时的优先级）；版本按「只增不改」取 **v2.28**（当前 v2.27 / `2edea18`，v2.25 / v2.26 均为历史版本，不可倒退）；E38 建立后实验计数 **37 → 38**，「所有 38 项已全量闭环结案」按 C4 收窄（A4-B 仍登记未测、§10.3 机制推测仍在）。② **3g 舍弃**（与 `k=2.0` 配对 `+0.0027 CI [−0.0022,+0.0077]` 含零，补 10 seed 只再测一次标量等价性，按 §11.9 P2 还须先复现同批次基线）；**3j 舍弃**（与 §4.15 平流负增益 0.1653→0.1432→0.1303 方向相反，且 `qpe_ckpt_path` 默认路径不存在会静默回退随机初始化、v1 只训 `x_radar[:, -1]` 逐帧套 11 帧是 OOD）；**3l 维持否决**（27 变量无 CIN/DCAPE，`ψ + γ·tanh(Wψ)` 只是门控残差）；**E36-A/B 用户裁决不投算力、判据原样留档**（5+5 代理 2/10、`0.039627 = 2.76 × 增益`、CSI–BIAS +0.9774；**A4-B 仍登记未测，不是舍弃**）；**待办 10 维持「数据条件缺失，不立项」**。§4.23 / §4.25 两个预留编号随 3j / 3l 结案。③ **C2 悬空清零**：§9 / §4.0 / §4.38 / `INDEX.md` 当前状态里「待用户裁决 / 须用户裁决 / 待裁决准入」= 0；§12 v2.26 行与 §4.37 卡片里的同类表述保留为**历史描述**并加交叉注。④ 同步 §4.0（37→38）/ §4.40 卡片 / §9 舍弃清单 +2 条 / §10.2 +4 条 / §13 README 待办行 / `INDEX.md` / `README.md` 状态声明；新增 `E38-open-items-final-adjudication.md`、`tools/_sync_v228.py`。⑤ **集体自评（§11.8.3）**：一次尝试（E38），**有关**（权威源状态完整性与算力去向），证据为 `+0.0027 CI 含零`、0.1653→0.1303、`qpe_ckpt_path` 静默回退、27 变量无 CIN/DCAPE、2/10 与 2.76× 与 +0.9774、头部第 4 行；去向 **保留并推进**（结案状态、卡片、索引、README 权威入口）+ **舍弃**（3g / 3j / 3l 及派生）。 | 用户 2026-11-07 17:52 裁决；`git log -3`（`2edea18` / `0cb0122` / `4650f3f`）；`ls docs/experiments/E*.md` 计数；`docs/MASTER.md` 头部第 3–4 行 / §4.0 / §9 / §13；既有评测数字（`tools/_evals/route_seeds/evals/` 96 份、`tools/_evals/e36_split_ensemble.json`）；纯本地文件核查，**未跑训练 / 评测 / 远端作业** |"""

INDEX_E38_ROW = """| E38 | §4.40 | 把 §9 最后五项「待用户裁决」一次性结案，再让派生层抄权威事实 | 既有实测数字 + 代码事实 + 文档一致性核查（零训练零推理零远端作业） | `E38-open-items-final-adjudication.md` | 已执行（5 项全部裁决；§9 无悬空裁决项；版本 v2.28） |"""

README_STATUS = """

> **Status · authoritative source (v2.28, 2026-11-07)**
>
> This README is an **engineering snapshot** written early in the project. Per
> `docs/MASTER.md` §13 it sits last in the conflict-resolution order; it is
> **not the authority for conclusions**. The single authoritative entry point is
> `docs/MASTER.md` (header lines 3-4: "本文件是本项目研究的唯一权威入口 …
> 任何产出先落到本文件，再落到别处" — any output lands here first, then elsewhere).
>
> As of **MASTER v2.28**, the §9 backlog has **no items left awaiting user
> adjudication**: all 38 registered attempts and every unexecuted direction have
> an explicit disposition (executed / discarded with evidence / not eligible
> because required data are missing / pre-registered but not run). This is
> **not** a claim that every experiment is finished: A4-B remains
> "registered, untested" and the mechanism-level questions in §10.3 are still
> explicitly marked as untested.
>
> Current research line: **diagnosis + falsification + the Oracle upper bound
> 0.1653 + robustness**, not a new architecture. See `docs/MASTER.md` §9 / §10 /
> §15 and `docs/experiments/INDEX.md`.
>
> Model and test counts in the body below (eight baselines, 205 / 305 / 413 /
> 336 tests, "nineteen baselines") are snapshots from different dates and are
> **inconsistent with each other**. They were deliberately left unchanged:
> no local test run backs a corrected number. Do not cite them as evidence.
"""

# (label, path, old, new)
R = []


def add(label, path, old, new):
    R.append((label, path, old, new))


# --- MASTER: header version stamp -------------------------------------------
add("version-stamp", MASTER,
    "| 最后更新 | 2026-11-07（v2.27，两份物理根因方案审查 + §11.9 资格门生效 + §15 三幕按已证否事实重写） |",
    "| 最后更新 | 2026-11-07（v2.28，§9 最后五项悬空裁决一次性结案；README 状态声明按 v2.28 权威事实更新） |")

# --- MASTER: §4.0 header note ------------------------------------------------
add("4.0-header", MASTER,
    """> 覆盖 §4.1–§4.39 的全部条目（含负面、作废、预注册与未执行），共 37 次尝试 +
> 2 个未执行的预留编号：§4.1–§4.24 的 23 次为 v2.16 迁出，§4.26–§4.39 十四次按
> §11.7 预注册，§4.33–§4.35 于 2026-11-07 12:05–13:35 预注册、§4.36 于 15:31 预注册、
> §4.37 / §4.38 于 16:30 预注册、§4.39 于 17:24 预注册，
> §4.23 / §4.25 是预留编号（无正文、无详情文件）。
> 「简要结果」只给承重数字；**完整结果一律点「详情」列的文件**。
> ⚠️ §4.23 / §4.25 是**预留编号**（两阶段解耦 / 开放量子耗散，**均未执行**），
> 无正文、无详情文件，登记在这里表示「该方向未立项」，不是漏写。""",
    """> 覆盖 §4.1–§4.40 的全部条目（含负面、作废、预注册与未执行），共 38 次尝试 +
> 2 个未执行的预留编号：§4.1–§4.24 的 23 次为 v2.16 迁出，§4.26–§4.40 十五次按
> §11.7 预注册，§4.33–§4.35 于 2026-11-07 12:05–13:35 预注册、§4.36 于 15:31 预注册、
> §4.37 / §4.38 于 16:30 预注册、§4.39 于 17:24 预注册、§4.40 于 17:52 预注册，
> §4.23 / §4.25 是预留编号（无正文、无详情文件）。
> 「简要结果」只给承重数字；**完整结果一律点「详情」列的文件**。
> ⚠️ §4.23 / §4.25 是**预留编号**（两阶段解耦 / 开放量子耗散，**均未执行**），
> 无正文、无详情文件；**v2.28 / E38 已裁决两者均不立项、不再留悬空**（见 §4.40）。""")

# --- MASTER: §4.0 table rows -------------------------------------------------
add("4.0-row-4.23", MASTER,
    "| §4.23 | 两阶段解耦：冻结反演网络，只学雨量场的十帧演化 | 未执行 | ⏳ 待用户裁决准入（与 §4.15 上界结论冲突） | — |",
    "| §4.23 | 两阶段解耦：冻结反演网络，只学雨量场的十帧演化 | 未执行 | ⛔ **舍弃（v2.28 / E38）**：与 §4.15 上界和平流负增益方向相反，且默认权重路径不存在会静默回退随机初始化、11 帧调用是分布外 | — |")

add("4.0-row-4.25", MASTER,
    "| §4.25 | 开放量子耗散跃迁门控 | 未执行 | ⛔ 否决作为科学方向 | — |",
    "| §4.25 | 开放量子耗散跃迁门控 | 未执行 | ⛔ **维持否决、已结案（v2.28 / E38）**：数据无 CIN/DCAPE，代码只是门控残差 | — |")

add("4.0-row-4.40", MASTER,
    "\n\n### 逐次尝试（大白话卡片）",
    "\n| §4.40 | 把 §9 最后五项「待用户裁决」一次性结案，再让派生层抄权威事实 | 版本按只增不改取 v2.28（当前 v2.27）；E38 建立后计数 37 → 38；用户论证三处事实校准：写入顺序的明文依据是头部第 4 行而非 §13；「全量闭环」收窄为「无悬空裁决项」（A4-B 仍登记未测） | **零算力文档结案，C1–C4 全 PASS**：3g / 3j / 3l 舍弃，E36-A/B 不投算力判据留档，待办 10 维持条件缺失不立项；§9 当前状态悬空字样清零，§12 / §4.37 历史描述保留并加注 | `docs/experiments/E38-open-items-final-adjudication.md` |\n\n### 逐次尝试（大白话卡片）")

# --- MASTER: §4.40 card ------------------------------------------------------
add("4.40-card", MASTER,
    "\n---\n\n## 5. 核心代码结构",
    "\n" + CARD_440 + "\n---\n\n## 5. 核心代码结构")

# --- MASTER: §4.37 historical cross-reference -------------------------------
add("4.37-history-note", MASTER,
    "§15 三幕重写、README、3g / 3j / 3l 与 E36 正式实验仍待用户裁决。",
    "§15 三幕重写、README、3g / 3j / 3l 与 E36 正式实验仍待用户裁决（**该句是 v2.26 的历史状态**：§15 已于 v2.27 重写、README 状态声明已于 v2.28 更新、3g / 3j / 3l / E36 已于 v2.28 / E38 结案）。")

# --- MASTER: §4.38 card verdict ---------------------------------------------
add("4.38-verdict", MASTER,
    "正式 E36-A/B **建议不投算力**、判据原样留档，是否投入由用户裁决。",
    "正式 E36-A/B **用户已裁决不投算力**（v2.28 / E38）、判据原样留档；**A4-B 仍「登记未测」，不是舍弃**。")

add("4.38-self-assessment", MASTER,
    "不投 20 seed 训练 / 重推理（除非用户裁决）、",
    "不投 20 seed 训练 / 重推理（v2.28 已裁决不投）、")

# --- MASTER: §9 five rows ----------------------------------------------------
add("9-row-3g", MASTER,
    "| ⏳ **待用户裁决是否投入额外算力** |",
    "| ⛔ **按 §11.8 舍弃（v2.28 / E38）**：与 `k=2.0` 配对 CI 含零，补跑只再测一次标量等价性；不再补 10 seed |")

add("9-row-3j", MASTER,
    "| ⏳ **须用户裁决准入**（§4.23） |",
    "| ⛔ **舍弃（v2.28 / E38）**：与 §4.15 方向相反 + 两处静默失效；§4.23 预留编号同步结案 |")

add("9-row-3l", MASTER,
    "| ⛔ 否决 / 待用户裁决（§4.25） |",
    "| ⛔ **维持否决、已结案（v2.28 / E38）**；不再写能量叙事 |")

add("9-row-10", MASTER,
    "⛔ **改判「数据条件缺失，不启动」（E35 / §4.37）**",
    "⛔ **改判「数据条件缺失，不启动」（E35 / §4.37；v2.28 / E38 确认）**")

# --- MASTER: §9 discard list -------------------------------------------------
add("9-discard-rows", MASTER,
    "\n\n---\n\n## 10. 已确立的结论（分层）",
    "\n" + DISCARD_ROWS + "\n\n---\n\n## 10. 已确立的结论（分层）")

# --- MASTER: §10.2 -----------------------------------------------------------
add("10.2-rows", MASTER,
    "\n\n### 10.3 未解释 / 待验证",
    "\n" + NOT_WRITE_ROWS + "\n\n### 10.3 未解释 / 待验证")

# --- MASTER: §12 changelog ---------------------------------------------------
add("12-changelog", MASTER,
    "\n\n---\n\n## 13. 冲突裁定规则",
    "\n" + CHANGELOG_ROW + "\n\n---\n\n## 13. 冲突裁定规则")

# --- MASTER: §13 README todo -------------------------------------------------
add("13-readme-todo", MASTER,
    "待办：更新 README 并在 §12 记一行。",
    "**v2.28 / E38 处置**：权威源（本文件 §9）已完成结案；`README.md` 已按 v2.28 事实新增状态声明与权威入口（**优先级第 8 位不变**）。正文里 8 baselines / 205 / 305 / 413 / 336 tests 等互相不一致的历史计数仍待一次带本地复测的独立清理 —— 本轮未跑测试，不凭猜测修订；已在 §12 v2.28 行记一行。")

# --- INDEX -------------------------------------------------------------------
add("index-e38", INDEX,
    "\n| — | §4.23 | 两阶段解耦（冻结反演网络，只学十帧演化） | 未执行 | — | ⏳ 待用户裁决准入 |",
    "\n" + INDEX_E38_ROW + "\n| — | §4.23 | 两阶段解耦（冻结反演网络，只学十帧演化） | 未执行 | — | ⛔ 舍弃（v2.28 / E38） |")

add("index-4.25", INDEX,
    "| — | §4.25 | 开放量子耗散跃迁门控 | 未执行 | — | ⛔ 否决作为科学方向 |",
    "| — | §4.25 | 开放量子耗散跃迁门控 | 未执行 | — | ⛔ 维持否决、已结案（v2.28 / E38） |")

# --- README (derived layer, written AFTER MASTER) ---------------------------
add("readme-status", README,
    """A unified benchmark for **radar + NWP precipitation nowcasting**, with eight
state-of-the-art baselines behind one adapter interface, a rewritten data
pipeline, and a meteorological verification suite.""",
    """A unified benchmark for **radar + NWP precipitation nowcasting**, with eight
state-of-the-art baselines behind one adapter interface, a rewritten data
pipeline, and a meteorological verification suite.""" + README_STATUS)


def main() -> int:
    cache = {}
    for label, path, old, new in R:
        text = cache.get(path)
        if text is None:
            text = load(path)
        count = text.count(old)
        if count != 1:
            print(f"ABORT [{label}] {path.name}: expected exactly 1 match, found {count}")
            return 1
        cache[path] = text.replace(old, new, 1)
        print(f"  ok [{label}] {path.name}")

    for path, text in cache.items():
        save(path, text)

    # E38 detail file: enforce the docs' CRLF convention.
    save(E38, load(E38))
    print(f"SYNC: PASS ({len(R)} replacements, {len(cache)} files + E38 EOL normalised)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
