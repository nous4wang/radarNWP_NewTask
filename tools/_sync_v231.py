"""v2.31 atomic sync: land E40 pre-registration in MASTER first, then derived layers.

Every replacement asserts exactly one match; nothing is written until all pass.
Zero training, zero evaluation, zero remote work.  Frozen criterion: 2026-11-08 20:28.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "docs/MASTER.md"
I = ROOT / "docs/experiments/INDEX.md"
R = ROOT / "README.md"
cache = {}


def text(p: Path) -> str:
    if p not in cache:
        cache[p] = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    return cache[p]


def rep(p: Path, old: str, new: str, label: str) -> None:
    t = text(p)
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: expected exactly 1 match, found {n}")
    cache[p] = t.replace(old, new, 1)
    print(f"  ok  {label}")


# ---------------------------------------------------------------- MASTER (authority)
rep(M,
    "| 最后更新 | 2026-11-08（v2.30，E39 的 C0 数据前提在远端 train 划分被证否，生成式路线整族舍弃；零训练零评测） |",
    "| 最后更新 | 2026-11-08（v2.31，基于 C0 的新机制证据另立「跨模态碎→整结构错配」数据前提 E40 / P1，20:28 冻结判据、未执行；零训练零评测零远端） |",
    "M1 header stamp")

rep(M,
    "> 覆盖 §4.1–§4.41 的全部条目（含负面、作废、预注册与未执行），共 39 次尝试 +",
    "> 覆盖 §4.1–§4.42 的全部条目（含负面、作废、预注册与未执行），共 40 次尝试 +",
    "M2 4.0 range/count")

rep(M,
    "> 2 个未执行的预留编号：§4.1–§4.24 的 23 次为 v2.16 迁出，§4.26–§4.41 十六次按",
    "> 2 个未执行的预留编号：§4.1–§4.24 的 23 次为 v2.16 迁出，§4.26–§4.42 十七次按",
    "M3 4.0 prereg count")

rep(M,
    "> §4.37 / §4.38 于 16:30 预注册、§4.39 于 17:24 预注册、§4.40 于 17:52 预注册、§4.41 于 2026-11-08 18:03 预注册，",
    "> §4.37 / §4.38 于 16:30 预注册、§4.39 于 17:24 预注册、§4.40 于 17:52 预注册、§4.41 于 2026-11-08 18:03 预注册、§4.42 于 2026-11-08 20:28 预注册，",
    "M4 4.0 prereg timeline")

rep(M,
    "> **§4.41 的 C0 数据前提已于 2026-11-08 19:48 在远端 train 划分执行并证否（v2.30）；该探测属 E39 内部判据，不新增实验编号，总次数仍为 39。**",
    "> **§4.41 的 C0 数据前提已于 2026-11-08 19:48 在远端 train 划分执行并证否（v2.30）；该探测属 E39 内部判据，不新增实验编号。**\n"
    "> **v2.31 / E40 是新增的第 40 次尝试**：仅预注册新数据前提 P1（20:28 冻结），**未执行、无结果、无性能数字**；总次数 39 → 40。",
    "M5 4.0 C0/E40 note")

rep(M,
    "| `docs/experiments/E39-generative-route-reexamination-and-ibcfm.md` |\n\n### 逐次尝试（大白话卡片）",
    "| `docs/experiments/E39-generative-route-reexamination-and-ibcfm.md` |\n"
    "| §4.42 | 另找角度：把 C0 的反直觉发现写成一个可证伪的跨模态「碎 → 整」结构错配数据前提 | "
    "**未执行（预注册 / P1 20:28 冻结）**：本轮零算力、零远端、零训练，只有判据与合规论证，无任何新实测或性能数字 | "
    "**`PRE-REGISTERED / NOT ELIGIBLE`**：§11.9 判定「不适用」（不宣称提升、不伪造 PASS）；P1 成立则保留为任务级机制证据，证否即按 §11.8 舍弃留档 | "
    "`docs/experiments/E40-cross-modal-structure-mismatch-premise.md` |\n\n### 逐次尝试（大白话卡片）",
    "M6 4.0 table row")

CARD_442 = """---

### 4.42 另找角度：跨模态「碎 → 整」结构错配这一新数据前提（预注册，未执行）

**当时想干什么（大白话）**：用户要求「另找角度，先写死新的数据前提并过 §11.9，
然后必然达到 TGRS 的 novelty」。Gate 0/1 核到三处必须先对齐：**① 不承诺「必然」**
（§11.8.2 第 2 条与 §11.9 P6 只许「可证伪 + 带证据 + 留档」）；**② §11.9 是
「候选方法准入的通过率资格门」**，管的是「提升是否成立」（P2 要 ≥10 个独立训练种子、
要训练算力），不是「是否 novel」的判定门；**③ 重开的合法入口是 §11.8.2 第 1 条的
「除非出现新的机制证据」** —— v2.30 的 C0 恰好产出了 v2.29 时不存在的新证据：
标签强降水是**稀疏但空间连贯的大块**（连通核约大 3.37 倍，22 正 1 负），
雷达回波反而更碎；另 §3.5 逐像素相关仅 r = +0.17。

**新假设（待证伪）**：这个任务不是「把高空回波平滑搬到未来」，也不是「给平滑场补
细节」，而是**从高空零碎斑块反演出地面连成片的强雨区**（碎 → 整的跨模态结构重组织）。
若成立，则领域主流方法（同模态结构平流外推、生成式高频细节补全）的结构先验方向
正好相反，可为「18 个可训练模型全部低于持久性地板 0.0924」提供机制解释 —— 这正是
用户要的「突出特点」。按 §11.2 Gate 1 逐条排除：不是幅度标定、不是门控、不是平流、
不是换损失／换架构、不是频域干预、不是集成派生、不是生成式条件采样、不是概率阈值；
**也不是「换个统计量重跑 C0」**（C0 问「标签自己是不是更碎」，P1 问「标签的大尺度
位置能不能由雷达预测」，主统计量是 C0 完全没有的跨模态分频段结构相关 + 纯低通对照）。

**数据前提 P1（2026-11-08 20:28 冻结，先写后跑，未执行）**：只读同一 train 划分
（预期 23 case / 548 帧、301×301、CR 索引 6），用二维实快速傅里叶变换按归一化径向
频率分三段（低频 < 0.125 / 中频 0.125–0.5 / 高频 ≥ 0.5，高频边界沿用 E39 §10），
算标签与雷达的**跨模态逐频段结构相关**。主判据 **H1-a**：`corr_low − corr_high`
的 95 % 置信区间**严格为负**；**H1-b**：`corr_low < 0.30`（写死）。护栏 **H2**：
把雷达做**纯低通滤波**（不允许任何位置重排）后若 `corr_low_lp ≥ 0.30`，则判
「只是尺度／滤波效应」、假设证否 —— 防止退化成又一个面积／幅度效应。
10000 次 case 级 bootstrap、seed 20261108、`n_cases ≥ 20`、方向严格、
`INCONCLUSIVE` 兜底、**不放宽条款**（不改判据 / 不删样本 / 不换阈值或频段边界重跑）。

**裁决**：**`PRE-REGISTERED / NOT ELIGIBLE`**；P1 **未执行**，无结果、无性能数字、
不进 §10.1、不改 §15 三幕。P1 PASS → 保留为**任务级机制证据**（novelty 落在反直觉
结构发现 + 对全败的机制解释 + §11.9 作为可迁移证据标准）；P1 FAIL → 按 §11.8
舍弃并留档。**§11.9 第十一项判定「不适用」**（§11.9.1：纯诊断类写不适用并说明理由），
**不填 `--check` manifest、不伪造 `VERDICT: PASS`**；本轮只复跑 `--self-test` 确认
9 个既有锚点未被破坏。将来若要把机制转成可训练方法并宣称提升，须另立详情文件、
补齐 P2 的同批次 ≥10 seed 基线复现（远端训练算力 + 用户显式授权），§10.2 关于
E33 的禁令不因此解除。

**证据层级**：代码事实 + 既有实测数字的再解释 + 预注册；本轮**零新增实测**。
**自评（§11.8）**：**有关**（动的是诊断工具／任务定义这一环，直接服务 §11.8.1 写死的
「为什么现在的深度方法做不到」）；证据为 C0 的 23 case / 548 帧、
`d_kernel +1.216473 CI [+0.965871,+1.442821]`（22 正 1 负）、§3.5 的 r = +0.17、
§10.1bis 的 +0.015524 / 39 % / 61 %；去向＝**预注册待执行**，P1 证否则舍弃且
「不再做什么」写进 E40 §7。**本轮零训练、零推理、零数据集评测、零远端作业。**

→ 完整合规论证、P1 判据、与 C0 的边界、§11.9 核对与 TGRS novelty 落点见
`docs/experiments/E40-cross-modal-structure-mismatch-premise.md`。

---

"""
rep(M,
    "\n---\n\n## 5. 核心代码结构",
    "\n" + CARD_442 + "## 5. 核心代码结构",
    "M7 4.42 card")

SECTION_9Y = """---

### 9.y v2.31 新立项（基于 C0 新机制证据，非悬空裁决复活）

> v2.28 的五项悬空裁决、v2.30 的 E39 结案状态**均不变**。本项是一次**新的用户裁决**：
> 「另找角度」。合法性来自 §11.8.2 第 1 条的「除非出现**新的机制证据**」——
> C0 的证否结果（标签是稀疏但连贯的大块、雷达更碎）是 v2.29 时不存在的证据。

| 项 | 状态 | 说明 |
|---|---|---|
| **E40 / §4.42 跨模态「碎 → 整」结构错配数据前提** | **`PRE-REGISTERED / NOT ELIGIBLE`（P1 未执行）** | 20:28 冻结判据：跨模态分频段结构相关 + 纯低通对照（H2 护栏）+ 10000 次 case 级 bootstrap + 不放宽条款。**零算力、零远端、零训练、零新增实测、零性能数字**；§11.9 第十一项判「不适用」并给理由，不伪造 PASS。P1 PASS → 保留为任务级机制证据；P1 FAIL → 按 §11.8 舍弃留档 |
| **P1 执行** | `未执行，待用户显式授权` | 只读 train 划分（预期 23 case / 548 帧）；脚本本轮刻意不写，判据必须先于任何实现与执行冻结。`--download` 须显式带 `--out` |
| **§10.2 / §15** | `§10.2 +2 条禁令；§15 三幕不改` | P1 无结果前不得写「结构错配已成立 / 已解决 TGRS novelty」；改 §15 会造成状态倒挂，故本轮不动 |

---

"""
rep(M,
    "\n---\n\n## 10. 已确立的结论（分层）",
    "\n" + SECTION_9Y + "## 10. 已确立的结论（分层）",
    "M8 9.y section")

rep(M,
    "| E39 §11 / `tools/_evals/c0_result.json` |\n\n> **v2.30 / E39 适用范围注记已失效（C5 自动触发）**",
    "| E39 §11 / `tools/_evals/c0_result.json` |\n"
    "| **「跨模态结构错配已成立 / 领域主流方法结构先验方向反了已被证实 / 本项目已解决 TGRS novelty」** | "
    "E40 的 P1 **未执行**，只有 20:28 冻结的判据，零新增实测；§11.8.2 与 §11.9 P6 不允许承诺「必然为正」；"
    "**本轮不宣称过 §11.9**（§11.9.1 判「不适用」、不伪造 PASS） | E40 §1 / §5 / §9 |\n"
    "| **「把 C0 换个统计量重跑可能显著」** | §4.41 与 E39 §11 的永久禁令；E40 §4 已划清"
    "「不同科学问题 + 新统计量 + 不重算 C0 verdict」的边界，不构成翻案 | §4.41 / E40 §4 |\n\n"
    "> **v2.30 / E39 适用范围注记已失效（C5 自动触发）**",
    "M9 10.2 two new bans")

CHANGELOG = """| 2026-11-08 | **v2.31** | **另找角度：基于 C0 新机制证据预注册跨模态「碎 → 整」结构错配数据前提（E40 / §4.42；零训练零推理零评测零远端，判据 20:28 先写）**：① 用户要求「另找角度、先写死新数据前提并过 §11.9、然后必然达到 TGRS novelty」。Gate 0/1 核到三处必须对齐：**不承诺「必然」**（§11.8.2 第 2 条 / §11.9 P6 只许可证伪 + 证据 + 留档）；**§11.9 是「候选方法准入的通过率资格门」**，管「提升是否成立」（P2 要 ≥10 独立训练种子、要训练算力），不是「是否 novel」的判定门，零算力下不可能满足；**重开合法入口是 §11.8.2 第 1 条「除非出现新的机制证据」** —— C0 的证否恰是 v2.29 不存在的新证据。② 新假设 H：任务是**从高空零碎回波斑块反演地面连成片强雨区**（碎→整跨模态结构重组织），与同模态平流外推 / 生成式细节补全的结构先验方向相反，可解释 18 个可训练模型全低于地板 0.0924、E33 高频几乎无损、r = +0.17。按 §11.2 逐条排除与已判负 8 族的同机制关系，**且不是「换个统计量重跑 C0」**。③ **P1 判据 20:28 冻结、未执行**：只读同一 train 划分（23 case / 548 帧），rfft2 分频段（低频 <0.125 / 中频 0.125–0.5 / 高频 ≥0.5，高频边界沿用 E39 §10），算跨模态逐频段结构相关；H1-a `corr_low − corr_high` 95 % CI 严格为负、H1-b `corr_low < 0.30`（写死）、**H2 纯低通对照护栏**防退化成幅度／滤波效应；10000 次 case 级 bootstrap、seed 20261108、`n_cases ≥ 20`、INCONCLUSIVE 兜底、不放宽条款。④ **§11.9 第十一项判「不适用」**（§11.9.1 纯诊断类明文），**不填 `--check` manifest、不伪造 `VERDICT: PASS`**；只复跑 `--self-test` 确认 9 个锚点未破坏；把 §11.9 的证据严格度（预注册 / 方向严格 CI / 单变量对照 / 不放宽 / 证否即停）搬到 P1。⑤ 同步 §4.0（39 → 40 次）/ §4.42 卡片 / §9.y 新立项 / §10.2 +2 条禁令 / §12；**§15 三幕刻意不改**（P1 无结果，改即状态倒挂）；`INDEX.md` 新增 E40 行、`README.md` 状态块升 v2.31 并写明「不是 novelty 已解决、不是必然保证」。⑥ **自评（§11.8）**：一次尝试（E40），**有关**（诊断工具／任务定义环，直接服务 §11.8.1 写死的「为什么现在的深度方法做不到」）；证据全为**既有锚点**（C0 23 case / 548 帧、`d_kernel +1.216473 CI 严格为正` 22 正 1 负、r = +0.17、+0.015524 / 39 % / 61 %、0.0729 与 2.29×），**本轮零新增实测**；去向 **预注册待执行**，P1 证否即舍弃并写明「不再做什么」。⑦ 新增 `docs/experiments/E40-cross-modal-structure-mismatch-premise.md`、`tools/_sync_v231.py`。 | 用户 2026-11-08 20:22 裁决；`docs/MASTER.md` §11.8.1 / §11.8.2 / §11.9 / §11.9.1 / §3.5 / §10.1bis / §4.41 / §10.2 原文核查；`tools/_evals/c0_result.json` 既有结果；`tools/_gate_11_9.py --self-test` 9/9 PASS；**未跑训练 / 未跑模型推理 / 未跑评测 / 未连远端 / 未碰 test 真值** |"""
rep(M,
    "\n\n---\n\n## 13. 冲突裁定规则",
    "\n" + CHANGELOG + "\n\n---\n\n## 13. 冲突裁定规则",
    "M10 changelog v2.31")

# ---------------------------------------------------------------- INDEX (derived)
rep(I,
    "\n| — | §4.23 | 两阶段解耦（冻结反演网络，只学十帧演化） | 未执行 | — | ⛔ 舍弃（v2.28 / E38） |",
    "\n| E40 | §4.42 | 另找角度：把 C0 的反直觉发现写成一个可证伪的跨模态「碎 → 整」结构错配数据前提 | "
    "代码事实 + 既有实测数字再解释 + 预注册（**零训练零评测零远端作业，P1 未执行**） | "
    "`E40-cross-modal-structure-mismatch-premise.md` | **预注册 / NOT ELIGIBLE**：P1 判据 20:28 冻结、未执行、无结果、无性能数字；"
    "§11.9 判「不适用」（不宣称提升、不伪造 PASS）；P1 成立则保留为任务级机制证据，证否按 §11.8 舍弃留档 |\n"
    "| — | §4.23 | 两阶段解耦（冻结反演网络，只学十帧演化） | 未执行 | — | ⛔ 舍弃（v2.28 / E38） |",
    "I1 E40 row")

# ---------------------------------------------------------------- README (derived)
rep(R,
    "> **Status · authoritative source (v2.30, 2026-11-08)**",
    "> **Status · authoritative source (v2.31, 2026-11-08)**",
    "R1 status stamp")

rep(R,
    "> As of **MASTER v2.30**, the §9 backlog has **no items left awaiting user\n> adjudication**: all 39 registered attempts",
    "> As of **MASTER v2.31**, the §9 backlog has **no items left awaiting user\n> adjudication**: all 40 registered attempts",
    "R2 status count")

rep(R,
    "> code is kept as evidence only. See `docs/MASTER.md` §9 / §10 / §15 and\n> `docs/experiments/INDEX.md`.",
    "> code is kept as evidence only. See `docs/MASTER.md` §9 / §10 / §15 and\n"
    "> `docs/experiments/INDEX.md`.\n"
    ">\n"
    "> **v2.31 adds one user-adjudicated pre-registration**, E40 (§4.42): a new\n"
    "> *falsifiable data premise* on cross-modal fragmented-to-coherent structure\n"
    "> mismatch, derived from C0's counter-intuitive finding (label rainfall is\n"
    "> sparser but spatially more coherent, connected core about 3.37x larger).\n"
    "> Its P1 criterion was frozen at 20:28 and **has not been executed** — no new\n"
    "> measurement, no CSI number, no performance claim. MASTER §11.9 is marked\n"
    "> **not applicable** for this diagnostic registration (§11.9.1); no\n"
    "> `VERDICT: PASS` is claimed. This is **not** a resolution of the TGRS novelty\n"
    "> question and **not** a guarantee that any result will be positive.\n"
    "> Zero training, zero inference, zero remote work.",
    "R3 E40 note")

# ---------------------------------------------------------------- write (CRLF)
for p, t in cache.items():
    p.write_bytes(t.replace("\n", "\r\n").encode("utf-8"))
    print(f"  wrote {p.relative_to(ROOT)} ({len(t.splitlines())} lines)")
print(f"SYNC: PASS ({len(cache)} files)")
