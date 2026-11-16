"""One-shot atomic sync for MASTER v2.29 / E39 (zero compute).

Every replacement asserts exactly one match, and every line insertion asserts
exactly one anchor line.  No file is written until all edits have succeeded.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "docs/MASTER.md",
    "docs/experiments/INDEX.md",
    "README.md",
    "docs/experiments/E35-open-items-closure-and-github-survey.md",
]
cache: dict[str, str] = {}
count = 0


def load(rel: str) -> str:
    if rel not in cache:
        cache[rel] = (ROOT / rel).read_text(encoding="utf-8")
    return cache[rel]


def rep(rel: str, old: str, new: str, label: str) -> None:
    global count
    text = load(rel)
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ABORT {label}: expected 1 match, found {n} in {rel}")
    cache[rel] = text.replace(old, new, 1)
    count += 1


def after_line(rel: str, prefix: str, payload: str, label: str) -> None:
    global count
    lines = load(rel).split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"ABORT {label}: expected 1 anchor, found {len(hits)}")
    i = hits[0]
    lines[i + 1:i + 1] = payload.split("\n")
    cache[rel] = "\n".join(lines)
    count += 1


M = "docs/MASTER.md"

# --- MASTER: version stamp --------------------------------------------------
rep(
    M,
    "| 最后更新 | 2026-11-07（v2.28，§9 最后五项悬空裁决一次性结案；README 状态声明按 v2.28 权威事实更新） |",
    "| 最后更新 | 2026-11-08（v2.29，生成式路线重审 + 12/6 模型口径订正 + IBCFM 预注册；零训练零评测零远端作业） |",
    "version stamp",
)

# --- MASTER section 2: the 18 = 12 + 6 counting correction -----------------
rep(
    M,
    "### 2.1 复现的论文方法（14 个）",
    "> **v2.29 / E39 口径订正**：`tools/_evals/score_matrix/` 的 18 份全矩阵是\n"
    "> **12 个复现论文方法 + 6 个本项目自建变体**，另有第 19 份 `pysteps_adv`\n"
    "> 是 `is_learnable=False` 的不训练物理基线。**「18 个 SOTA」是措辞错误**；\n"
    "> 全文一律改称「18 个可训练模型」，裁决与数字一字未改。另查明两个「扩散」\n"
    "> 基线（`diffcast` / `cascast_nwp`）都是**确定性单样本**（零起点 DDIM / 固定\n"
    "> `noise_seed`），随机条件采样从未测过 —— 见 §4.41 / E39 §1.2。\n\n"
    "### 2.1 复现的论文方法（14 个）",
    "section 2 count note",
)

# --- MASTER section 4.0 index header ---------------------------------------
rep(M, "覆盖 §4.1–§4.40 的全部条目（含负面、作废、预注册与未执行），共 38 次尝试 +",
    "覆盖 §4.1–§4.41 的全部条目（含负面、作废、预注册与未执行），共 39 次尝试 +",
    "4.0 coverage")
rep(M, "§4.26–§4.40 十五次按", "§4.26–§4.41 十六次按", "4.0 count")
rep(M, "§4.37 / §4.38 于 16:30 预注册、§4.39 于 17:24 预注册、§4.40 于 17:52 预注册，",
    "§4.37 / §4.38 于 16:30 预注册、§4.39 于 17:24 预注册、§4.40 于 17:52 预注册、§4.41 于 2026-11-08 18:03 预注册，",
    "4.0 registration times")

# --- MASTER section 4.0 table row ------------------------------------------
after_line(
    M, "| §4.40 |",
    "| §4.41 | 重审「18 个 SOTA」口径、查清生成式模型到底测没测过，为 TGRS 找一个新的可证伪角度 | 用户第 1 点成立：18 = 12 复现论文方法 + 6 自建变体；两个「扩散」基线是零起点 / 固定种子的确定性单样本；既有全矩阵 18 个可训练模型 BIAS@2.0 全在 0.035–0.435，物理基线 0.728；外部 8 篇论文 + 6 个真实仓库核实 | **零算力审查 + 预注册**：订正口径；用户裁决重新打开建设性路线，新模型 `ibcfm` 状态 `PRE-REGISTERED / NOT ELIGIBLE`，C0–C5 判据先写后跑；本轮只交付设计 + 代码 + CPU 单测 8/8，**无任何性能数字** | `docs/experiments/E39-generative-route-reexamination-and-ibcfm.md` |",
    "4.0 E39 row",
)

# --- MASTER 4.41 plain-language card ---------------------------------------
CARD = """### 4.41 重审生成式路线，为 TGRS 找一个可证伪的新角度

**当时想干什么（大白话）**：用户指出三件事：项目里没有 18 个 SOTA、只有 12 个；纯 benchmark / 诊断稿在 TGRS 达不到发表条件，得重新找 SOTA 或 novelty；这个任务的标签是融合了地面雨量的高频点状场，跟平滑雷达回波外推不是一类问题，应该更适合生成模型。用户要求完整调研论文与 GitHub 代码库，并构建一个新模型。

**结果怎么样**：三条逐一核到代码与文件。**① 用户正确** —— `score_matrix/` 18 份 = 12 个复现论文方法 + 6 个自建变体，另有一份不训练的 `pysteps_adv`，「18 个 SOTA」是措辞错误。**② 一半不成立** —— `diffcast` / `cascast_nwp` 本来就在 12 个里，且是真扩散实现（DDPM 时间嵌入 + 去噪 U-Net/DiT + DDIM）。**但找到了机制缝隙**：`models/diffcast/model.py:344–357` 的 DDIM 从**全零**起、`cascast_nwp` 用**固定 `noise_seed`**，两者都是确定性单样本，本项目从未测过「建模 p(Y|X) 并随机采样多个成员」。**③ 数据事实支持、但要收窄** —— §3.5 的 r = +0.17 跨模态鸿沟是真的；另外查出 18 个可训练模型的 `BIAS@2.0` 全部在 **0.035–0.435**、无一接近 1，而不训练的物理基线是 0.728。不过 E18 / E25 已经证明「把面积报够」用一个乘性常数就能做到，且被判为面积/幅度效应 → 新方案必须赢「等偏置纯幅度对照」。外部调研核实 8 篇论文（MTLDM / TGRS 2025、S3NN / TGRS 2026、Guilloteau / TGRS 2025、DGMR / Nature 2021、FlowCast / ICLR 2026、MW-Nowcast、StormDiT 等）与 6 个真实仓库；MW-Nowcast 的消融「潜空间抹平极端核」正好解释本项目 `cascast_nwp` 0.0282 < `diffcast` 0.0717。

**裁决**：**口径订正通过，只改标签不改数字**。**用户裁决重新打开建设性路线** —— 这是新机制类别（随机条件采样 + 集合阈值决策）+ 新判据 + 新详情文件，不是 §11.8 禁止的「换个说法再试一次」，也不是 v2.28 五项悬空裁决复活。新模型 `ibcfm`（确定性锚 + 像素空间 CFM 随机残差 + 强度分带 + 阈值概率决策）状态 **`PRE-REGISTERED / NOT ELIGIBLE`**；§10.2 关于 E33 的原禁令不删，只加适用范围注记，C3 不过即自动恢复并永久封口。**本轮零训练、零数据集评测、零远端作业**，只交付设计、代码与 CPU 小张量单测 8/8。

**证据层级**：代码事实 + 既有评测副本读取排序 + 外部论文页 / GitHub API 实测 + 本地 CPU 单测；不是新的数据集结果，不写入 §10.1 性能层。

**自评（§11.8）**：**有关** —— 目标已被用户改为「TGRS 可发表的 SOTA / novelty」。**证据**：18 = 12 + 6 文件计数、`diffcast:344–357` 零起点 DDIM、固定 `noise_seed`、BIAS 全表 0.035–0.435 vs 0.728、r = +0.17、8 篇论文与 6 个仓库。**去往**：保留并推进到预注册；**舍弃**「再搭一个去噪网络就 novel」「换 GitHub 新架构」「靠把面积报够就能发」三种说法。**不再做什么**：不再以「扩散没试过」立项、不用真值高频、不做相位对齐、不在 test 调阈值、不在同批次基线复现前报任何提升。

→ 完整调研表、代码事实、模型设计、C0–C5 预注册判据与 A0–A5 消融臂见 `docs/experiments/E39-generative-route-reexamination-and-ibcfm.md`。
"""
rep(M, "\n---\n\n## 5. 核心代码结构", "\n" + CARD + "\n---\n\n## 5. 核心代码结构",
    "4.41 card")

# --- MASTER section 9: new pre-registration (after the discard table) ------
NEW9 = """
### 9.x v2.29 新立项（用户裁决重新打开建设性路线）

> **这不是 v2.28 五项悬空裁决的复活**。那五项已结案且状态不变；下面是一项**新的**
> 用户裁决：新的机制类别 + 新的可证伪判据 + 新的详情文件 + 新的编号。

| 项 | 状态 | 说明 |
|---|---|---|
| **E39 / §4.41 IBCFM** | `PRE-REGISTERED / NOT ELIGIBLE` | 用户 2026-11-08 裁决：TGRS 需要 SOTA / novelty，重新打开建设性路线。判据 C0–C5 于 18:03 冻结（早于任何模型代码），含 §11.9 资格门、等偏置纯幅度对照、4 mm 档不为负、BIAS ∈ [0.8, 1.25]。**C0 数据前提未验证、C2 未跑、C3/C4 无任何数字 → 不投算力、不进 §10.1、不得写「有效 / 有望 SOTA」**。任一判据不过即按 §11.8 舍弃留档 |
| **C0 数据前提探测** | `登记，未执行` | 用 **train 划分**（不用 test 真值）量 label 的 ≥2 mm 占比、连通核尺度、RAPSD 高频能量，与同期 radar CR 比较；若不成立则整个方向立即舍弃。远端 CUDA 执行，本轮不做 |
| **§10.2 原禁令** | `保留，加适用范围注记` | E33 的「不得据此写任何可训练 / 可部署方案」不删不改；E39 只在「不使用真值高频、不声称恢复相位」的前提下开一个不同优化目标的例外。C3 失败即自动恢复全文 |
"""
rep(M, "\n\n---\n\n## 10. 已确立的结论（分层）",
    "\n" + NEW9 + "\n---\n\n## 10. 已确立的结论（分层）", "section 9 new item")

# --- MASTER 10.2: correction + new prohibitions + scope note ---------------
rep(M, "18 个 SOTA 全部低于地板 0.0924（最好 0.0896）",
    "18 个可训练模型（12 复现论文方法 + 6 自建变体）全部低于地板 0.0924（最好 0.0896）",
    "9 discard-table count correction")
rep(M, "：18 个 SOTA 全部低于地板 0.0924；",
    "：18 个可训练模型（12 个复现论文方法 + 6 个自建变体）全部低于地板 0.0924；",
    "10.2 count correction")

NEW102 = """| **「本项目测过 18 个 SOTA，扩散 / 生成式模型已被证否」** | 18 = 12 复现论文方法 + 6 自建变体；且两个「扩散」基线是零起点 DDIM / 固定种子的**确定性单样本**，随机条件采样从未测过 | §4.41 / E39 §1.1–1.2 |
| **「再搭一个去噪网络 / 换一个 GitHub 新架构就是 novelty」** | 机制未变，按 §11.2 Gate 1 属同族；E35 §4.37.6 的否决是统计准入论证，不是机制证否 | §4.41 / E39 §2.3 |
| **「生成模型只要把 BIAS@2.0 报接近 1 就能提升 CSI」** | E18 / E25 / E26 已证明等偏置纯幅度对照可复现，判为面积 / 幅度效应；E39 C3 把它设为决定性判据 | §4.25 / E39 §3.2 / §5 C3 |
| **「用 E33 的真值高频 / 相位信息指导新模型训练」** | §10.2 原禁令仍有效；IBCFM 不使用任何真值高频段，也不声称恢复不可观测相位 | §4.35 / E39 §4.4 |
| **「E39 已证明生成式路线有效 / 有望突破地板 / 可以投 TGRS」** | 本轮**零性能数字**，状态 `PRE-REGISTERED / NOT ELIGIBLE`；C0–C4 全过之前不得这样写 | E39 §6 / §7 |
| **「val 上扫阈值 / test 上调 `p*`」** | val 无场（E26）；E18 已判 test 调标量有信息泄漏；只许 train 划分标定 | E39 §4.1 / §7 |

> **v2.29 / E39 适用范围注记**：上表中关于 E33 的两条禁令**全文保留、数字不改**。
> 用户 2026-11-08 裁决重新打开的 E39 走的是不同优化目标（承认相位不可观测，
> 建模阈值超越概率分布并做 train 划分上的最优决策），**不使用真值高频、不做相位对齐**。
> 若 E39 的 **C3**（必须赢等偏置纯幅度对照）不通过，本注记自动失效、原禁令全文恢复，
> 该方向按 §11.8 永久封口。"""
after_line(M, "| **「README 里的模型数 / 测试数可作为结论依据」**",
           NEW102, "10.2 new prohibitions")

# --- MASTER section 12 changelog -------------------------------------------
CHLOG = "| 2026-11-08 | **v2.29** | **生成式路线重审 + 18/12 口径订正 + IBCFM 预注册（E39 / §4.41；零训练零数据集评测零远端作业，判据 18:03 先写）**：① 用户三条重审逐条核到代码 —— `score_matrix/` 18 份 = **12 个复现论文方法 + 6 个自建变体**，另 `pysteps_adv` 为不训练物理基线，「18 个 SOTA」是措辞错误，§2 / §10.2 / E35 只改标签不改数字；`diffcast:344–357` DDIM 从全零起、`cascast_nwp` 固定 `noise_seed` → 两个「扩散」基线其实是**确定性单样本**，「随机条件采样」是从未测过的机制类别。② 新查事实：18 个可训练模型 `BIAS@2.0` 全在 **0.035–0.435**，物理基线 0.728；但 E18 / E25 已证「把面积报够」可被单一乘数复现 → C3 设为「必须赢等偏置纯幅度对照」。③ 外部调研 8 篇论文（MTLDM / TGRS 2025、S3NN / TGRS 2026、Guilloteau / TGRS 2025、DGMR / Nature 2021、FlowCast / ICLR 2026、MW-Nowcast、StormDiT、物理先验扩散残差）+ 6 个 API 实测仓库；MW-Nowcast 的「潜空间抹平极端核」解释 `cascast_nwp` 0.0282 < `diffcast` 0.0717。④ 用户裁决重新打开建设性路线，新增 `models/ibcfm/`（确定性锚 + 像素空间 CFM 随机残差 + 强度分带 + 阈值概率决策）、`configs/models/ibcfm.yaml`、`tests/test_ibcfm.py`（CPU 小张量 **8/8 PASS**）、`scripts/train.py` 的 `--loss ibcfm` 分支（训练循环 / 评测器 / 数据管线零改动）；**状态 `PRE-REGISTERED / NOT ELIGIBLE`，无任何性能数字**。⑤ 同步 §2 口径注 / §4.0（38 → 39 次）/ §4.41 卡片 / §9 新立项 / §10.2 口径订正 + 6 条新禁令 + 适用范围注记 / §12；`INDEX.md` 新增 E39；`README.md` 状态块更新。⑥ **自评（§11.8）**：一次尝试（E39），**有关**（用户已把目标改为 TGRS 可发表的 SOTA / novelty），证据为文件计数 + 代码行 + BIAS 全表 + r = +0.17 + 外部调研；去向 **保留并推进到预注册** + **舍弃**三种说法，并写明「不再做什么」。 | 用户 2026-11-08 18:02 裁决；`tools/_evals/score_matrix/*.eval.json` 读取排序；`models/diffcast/model.py` / `models/cascast_nwp/model.py` / `radar_nwp_bench/models/base.py` / `scripts/train.py` 代码事实；GitHub API `search_repositories` 与论文页核查；本地 `python -m pytest tests/test_ibcfm.py -q`；**未跑训练 / 评测 / 远端作业** |"
after_line(M, "| 2026-11-07 | **v2.28** |", CHLOG, "12 changelog")

# --- INDEX ------------------------------------------------------------------
after_line("docs/experiments/INDEX.md", "| E38 | §4.40 |",
           "| E39 | §4.41 | 重审「18 个 SOTA」口径、查清生成式模型测没测过，为 TGRS 找可证伪的新角度并预注册新模型 | 代码事实 + 既有评测副本读取排序 + 外部论文页 / GitHub API 调研 + 本地 CPU 小张量单测（零训练零评测零远端作业） | `E39-generative-route-reexamination-and-ibcfm.md` | 已执行（口径订正；IBCFM 预注册 / `NOT ELIGIBLE`；C0–C5 判据冻结；无任何性能数字） |",
           "INDEX E39")

# --- README (derived layer, last) ------------------------------------------
rep("README.md",
    "> **Status · authoritative source (v2.28, 2026-11-07)**",
    "> **Status · authoritative source (v2.29, 2026-11-08)**", "README stamp")
rep("README.md",
    "> As of **MASTER v2.28**, the §9 backlog has **no items left awaiting user\n"
    "> adjudication**: all 38 registered attempts and every unexecuted direction have",
    "> As of **MASTER v2.29**, the §9 backlog has **no items left awaiting user\n"
    "> adjudication**: all 39 registered attempts and every unexecuted direction have",
    "README count")
rep("README.md",
    "> Current research line: **diagnosis + falsification + the Oracle upper bound\n"
    "> 0.1653 + robustness**, not a new architecture. See `docs/MASTER.md` §9 / §10 /\n"
    "> §15 and `docs/experiments/INDEX.md`.",
    "> Current research line remains **diagnosis + falsification + the Oracle upper\n"
    "> bound 0.1653 + robustness**. As of v2.29, §9 has one **new user-adjudicated\n"
    "> pre-registration**: E39 (§4.41, IBCFM), status `PRE-REGISTERED / NOT ELIGIBLE`,\n"
    "> zero training and zero evaluation, criteria C0-C5 frozen before any compute.\n"
    "> It is a new mechanism category, not a revival of the five v2.28 adjudications,\n"
    "> and **not a performance result**. See `docs/MASTER.md` §9 / §10 / §15 and\n"
    "> `docs/experiments/INDEX.md`.",
    "README research line")

# --- E35 historical citation ------------------------------------------------
rep("docs/experiments/E35-open-items-closure-and-github-survey.md",
    "18 个 SOTA 全部低于 0.0924",
    "18 个可训练模型（12 复现论文方法 + 6 自建变体）全部低于 0.0924（v2.29 / E39 口径订正）",
    "E35 count")

# --- write ------------------------------------------------------------------
for rel, text in cache.items():
    (ROOT / rel).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
print(f"SYNC: PASS ({count} replacements, {len(cache)} files)")
