"""One-shot atomic sync for v2.30 (E39 C0 falsification).

Every replacement must match exactly once, otherwise NOTHING is written.
Order: MASTER (authority) -> INDEX -> README -> E39 (detail).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cache: dict[Path, str] = {}


def text(path: str) -> str:
    p = ROOT / path
    if p not in cache:
        cache[p] = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    return cache[p]


def rep(path: str, old: str, new: str, label: str) -> None:
    p = ROOT / path
    t = text(path)
    count = t.count(old)
    if count != 1:
        raise SystemExit(f"ABORT {label}: expected exactly 1 match, found {count}")
    cache[p] = t.replace(old, new, 1)
    print(f"  [{label}] ok")


M = "docs/MASTER.md"
I = "docs/experiments/INDEX.md"
R = "README.md"
E = "docs/experiments/E39-generative-route-reexamination-and-ibcfm.md"

# ---------------------------------------------------------------- MASTER
rep(M,
    "| 最后更新 | 2026-11-08（v2.29，生成式路线重审 + 12/6 模型口径订正 + IBCFM 预注册；零训练零评测零远端作业） |",
    "| 最后更新 | 2026-11-08（v2.30，E39 的 C0 数据前提在远端 train 划分被证否，生成式路线整族舍弃；零训练零评测） |",
    "M1 header stamp")

rep(M,
    "> §4.23 / §4.25 是预留编号（无正文、无详情文件）。",
    "> §4.23 / §4.25 是预留编号（无正文、无详情文件）。\n"
    "> **§4.41 的 C0 数据前提已于 2026-11-08 19:48 在远端 train 划分执行并证否（v2.30）；"
    "该探测属 E39 内部判据，不新增实验编号，总次数仍为 39。**",
    "M2 4.0 header note")

rep(M,
    "| **零算力审查 + 预注册**：订正口径；用户裁决重新打开建设性路线，新模型 `ibcfm` 状态 `PRE-REGISTERED / NOT ELIGIBLE`，C0–C5 判据先写后跑；本轮只交付设计 + 代码 + CPU 单测 8/8，**无任何性能数字** |",
    "| **口径订正保留；C0 证否、整族舍弃（v2.30）**：远端 train 划分 23 case / 548 帧实测 `d_hf = −0.004513 CI [−0.011703,+0.001625]` 跨零、`d_kernel = +1.216473 CI [+0.965871,+1.442821]` 严格为正（逐 case 22 正 1 负）；label ≥2 mm 占比 **0.004185** 比 radar ≥35 dBZ 的 **0.028337** 更稀疏，但强降水是**更连贯的大块**而非高频点状 → §3.3 的 task novelty 不成立。按 §5 C0 / C5 与 §11.8 舍弃，`ibcfm` 代码留档不派生 |",
    "M3 4.0 row 4.41")

CARD_NEW = """**C0 执行结果（v2.30，2026-11-08 19:48，远端 train 划分）**：操作化细则先于执行于
19:46 冻结（E39 §10），`split_cases()` 给出 train 23 / val 8 / test 8，**23 个 train case
全部用上、548 帧、零跳过、零异常**。结果与用户的第 3 条直觉相反：label 的 ≥2 mm 像素占比
**0.004185** 确实比 radar ≥35 dBZ 的 **0.028337** 更稀疏，但 RAPSD 高频能量差
`d_hf = −0.004513，95 % CI [−0.011703, +0.001625]` **跨零且点估计为负**；等覆盖率下
连通核尺度差 `d_kernel = +1.216473，95 % CI [+0.965871, +1.442821]` **严格为正**
（逐 case **22 正 1 负**，即 label 的强降水核平均比 radar 大约 **3.37 倍**）。
→ **label 是「稀疏但空间连贯的大块」，不是高频点状场**；§3.3 的 task-level novelty
不成立，B1 / B2 双双 FAIL。

**v2.30 裁决（§5 C0 + C5 自动生效）**：**整个生成式条件采样方向按 §11.8 舍弃留档**，
不投 C1–C4 任何训练算力；§10.2 适用范围注记失效、E33 原禁令全文恢复；
`models/ibcfm/` 与 `--loss ibcfm` 分支**保留作证据**（删文件＝销毁证据），但不训练、
不评测、不派生、不包装成论文贡献。**不得写「有望 SOTA / 突破地板」**。
本次仅有一次远端只读数据探测：零训练、零模型推理、零 checkpoint、零 test 真值。

**自评补充（§11.8）**：**有关**（目标是 TGRS 可发表的 novelty，本轮用极小代价否掉
最关键的数据前提）。**证据**：23 case / 548 帧、两组 10000 次 bootstrap CI、
22 正 1 负、0.004185 vs 0.028337。**去向：舍弃**。**不再做什么**：不再以
「label 高频 / 点状 / 跨模态尖峰」为由立项；不再换采样器、强度分带或换一个生成模型
派生新实验；不投 C1–C4；不改 §5 / §10 判据；不以「换个统计量可能显著」重跑 C0。

"""

rep(M,
    "→ 完整调研表、代码事实、模型设计、C0–C5 预注册判据与 A0–A5 消融臂见 `docs/experiments/E39-generative-route-reexamination-and-ibcfm.md`。",
    CARD_NEW + "→ 完整调研表、代码事实、模型设计、C0–C5 预注册判据、§10 操作化细则与 §11 执行结果见 `docs/experiments/E39-generative-route-reexamination-and-ibcfm.md`。",
    "M4 4.41 card")

rep(M,
    "新模型 `ibcfm`（确定性锚 + 像素空间 CFM 随机残差 + 强度分带 + 阈值概率决策）状态 **`PRE-REGISTERED / NOT ELIGIBLE`**；§10.2 关于 E33 的原禁令不删，只加适用范围注记，C3 不过即自动恢复并永久封口。**本轮零训练、零数据集评测、零远端作业**，只交付设计、代码与 CPU 小张量单测 8/8。",
    "新模型 `ibcfm`（确定性锚 + 像素空间 CFM 随机残差 + 强度分带 + 阈值概率决策）状态 **`PRE-REGISTERED / NOT ELIGIBLE`**；§10.2 关于 E33 的原禁令不删，只加适用范围注记，C3 不过即自动恢复并永久封口。**本轮零训练、零数据集评测、零远端作业**，只交付设计、代码与 CPU 小张量单测 8/8。**（本段是 18:03 预注册时的历史状态，作为留档保留；19:48 的 C0 证否与 v2.30 封口见下方新段落。）**",
    "M5 4.41 historical-state annotation")

rep(M,
    "### 9.x v2.29 新立项（用户裁决重新打开建设性路线）",
    "| 生成式条件采样 / `ibcfm` 及其任何派生（换采样器、换强度分带、换生成模型、换阈值决策） | **C0 数据前提在远端 train 划分被证否**：label 强降水稀疏但空间连贯，不是高频点状场；`d_hf = −0.004513 CI [−0.011703,+0.001625]` 跨零、`d_kernel = +1.216473 CI [+0.965871,+1.442821]` 严格为正（逐 case 22 正 1 负） | E39 §10 细则 / §11 执行结果；`tools/_evals/c0_result.json`；§5 C0 与 C5 | `E39-generative-route-reexamination-and-ibcfm.md` §11 |\n\n"
    "### 9.x v2.29 新立项 → v2.30 C0 证否结案",
    "M6 section 9 discard row + heading")

rep(M,
    """| **E39 / §4.41 IBCFM** | `PRE-REGISTERED / NOT ELIGIBLE` | 用户 2026-11-08 裁决：TGRS 需要 SOTA / novelty，重新打开建设性路线。判据 C0–C5 于 18:03 冻结（早于任何模型代码），含 §11.9 资格门、等偏置纯幅度对照、4 mm 档不为负、BIAS ∈ [0.8, 1.25]。**C0 数据前提未验证、C2 未跑、C3/C4 无任何数字 → 不投算力、不进 §10.1、不得写「有效 / 有望 SOTA」**。任一判据不过即按 §11.8 舍弃留档 |
| **C0 数据前提探测** | `登记，未执行` | 用 **train 划分**（不用 test 真值）量 label 的 ≥2 mm 占比、连通核尺度、RAPSD 高频能量，与同期 radar CR 比较；若不成立则整个方向立即舍弃。远端 CUDA 执行，本轮不做 |
| **§10.2 原禁令** | `保留，加适用范围注记` | E33 的「不得据此写任何可训练 / 可部署方案」不删不改；E39 只在「不使用真值高频、不声称恢复相位」的前提下开一个不同优化目标的例外。C3 失败即自动恢复全文 |""",
    """| **E39 / §4.41 IBCFM** | **`DISCARDED` / C0 FAILED（v2.30 结案）** | 用户 2026-11-08 裁决重新打开建设性路线；C0–C5 于 18:03 冻结、§10 操作化细则于 19:46 冻结。**19:48 远端 train 划分（23 case / 548 帧）执行 C0 → FAIL**：`d_hf = −0.004513 CI [−0.011703,+0.001625]` 跨零、`d_kernel = +1.216473 CI [+0.965871,+1.442821]` 严格为正（22 正 1 负）。按 §5 C0「立即舍弃整个方向、不投任何训练算力」与 C5，**整族舍弃留档**；C1–C4 永不执行、不进 §10.1、不得写「有效 / 有望 SOTA」 |
| **C0 数据前提探测** | `已执行（远端只读）` | 只用 train 划分，零训练零推理零 test 真值；`tools/_c0_probe.py` → `tools/_evals/c0_result.json`。两条 CI 与辅助一致性按 §10.4 判定，判据未放宽 |
| **§10.2 原禁令** | `适用范围注记已失效，禁令全文恢复` | C5 生效：E33 的「不得据此写任何可训练 / 可部署方案」全文恢复，E39 例外撤销，方向按 §11.8 永久封口 |""",
    "M7 section 9 table")

rep(M,
    "| **「val 上扫阈值 / test 上调 `p*`」** | val 无场（E26）；E18 已判 test 调标量有信息泄漏；只许 train 划分标定 | E39 §4.1 / §7 |",
    "| **「val 上扫阈值 / test 上调 `p*`」** | val 无场（E26）；E18 已判 test 调标量有信息泄漏；只许 train 划分标定 | E39 §4.1 / §7 |\n"
    "| **「label 是高频点状场，所以更适合生成模型 / diffusion」** | C0 在 train 划分 23 case / 548 帧上直接证否：label ≥2 mm 占比 0.004185 比 radar ≥35 dBZ 的 0.028337 更稀疏，但高频能量差跨零、连通核反而大 3.37 倍（22 正 1 负）→ 稀疏 ≠ 高频，强降水是连贯大块 | E39 §11 / `tools/_evals/c0_result.json` |",
    "M8 10.2 new prohibition")

rep(M,
    """> **v2.29 / E39 适用范围注记**：上表中关于 E33 的两条禁令**全文保留、数字不改**。
> 用户 2026-11-08 裁决重新打开的 E39 走的是不同优化目标（承认相位不可观测，
> 建模阈值超越概率分布并做 train 划分上的最优决策），**不使用真值高频、不做相位对齐**。
> 若 E39 的 **C3**（必须赢等偏置纯幅度对照）不通过，本注记自动失效、原禁令全文恢复，
> 该方向按 §11.8 永久封口。""",
    """> **v2.30 / E39 适用范围注记已失效（C5 自动触发）**：19:48 的 **C0 FAIL**
> 早于 C3，按 §5 C5「C0–C4 任一不过 → 方向按 §11.8 舍弃、E39 留档、MASTER §10.2
> 恢复原禁令全文」，上表中关于 E33 的两条禁令**全文恢复、无例外**；E39 的
> `ibcfm` 代码与配置保留留档但不训练、不评测、不派生。**该方向永久封口**，
> 不再以「换一个生成模型 / 换一个采样器 / 换一个强度分带」立项。""",
    "M9 10.2 scope note")

CHANGELOG_ROW = """| 2026-11-08 | **v2.30** | **E39 C0 数据前提在远端 train 划分证否 → 生成式路线整族舍弃（§4.41；零训练零推理，细则 19:46 先写、19:48:45 执行）**：① 用户 19:42 授权「在远端跑 C0」。执行前先把 §5 C0 未定义的口径写死为 E39 §10（CR 是 dBZ、label 是 mm，**不存在可比的 ≥2 mm 阈值**；主口径用等覆盖率分位阈值，RAPSD 高频定义为归一化径向频率 ≥0.5，连通核 4-连通，10000 次 case 级 bootstrap，`n_cases ≥ 20`，辅助一致性，INCONCLUSIVE 兜底）——**§5 原文一字不改，整体只加严**。② 远端只读执行 `tools/_c0_probe.py`：`split_cases()` → train **23** / val 8 / test 8（62 个 case 中 23 个 irregular 剔除），**23 个 train case 全部用上、548 帧、shape 全 (301,301)、CR 索引全 6、零跳过、零异常**。③ 结果：`p_label≥2mm = 0.004185` < `p_CR≥35dBZ = 0.028337`（辅助一致性 PASS、最小样本 PASS），但 **`d_hf = −0.004513` CI `[−0.011703,+0.001625]` 跨零**（B1 FAIL）、**`d_kernel = +1.216473` CI `[+0.965871,+1.442821]` 严格为正、逐 case 22 正 1 负**（B2 FAIL，label 强降水核约大 **3.37 倍**）→ **label 是稀疏但空间连贯的大块，不是高频点状场**，用户第 3 条直觉的数据前提不成立，§3.3 的 task novelty 不成立。④ **裁决**：按 §5 C0「立即舍弃整个方向、不投任何训练算力」+ C5 + §11.8，生成式条件采样方向**整族舍弃留档**；C1–C4 永不执行；§10.2 适用范围注记失效、E33 原禁令全文恢复；`models/ibcfm/`、配置、单测与 `--loss ibcfm` 分支**保留作证据**（删文件＝销毁证据）但不训练不评测不派生。⑤ 同步 §4.0 / §4.41 卡片 / §9 舍弃清单 +1 条与结案表 / §10.2 +1 条禁令与注记 / §12；`INDEX.md`、`README.md`、E39 §10–§11 派生层按权威事实更新。⑥ **自评（§11.8）**：一次尝试（C0），**有关**（用极小代价否掉 TGRS novelty 最关键的数据前提），证据为 23 case / 548 帧、两组 bootstrap CI、22 正 1 负、0.004185 vs 0.028337；去向 **舍弃**，并写明「不再做什么」。⑦ **工程坑**：`tools/remote.py --download` 不带 `--out` 会把文件按 basename 落到仓库根，已归位并记入日志。 | 用户 2026-11-08 19:42 授权；远端 `183.136.129.105:1003` 只读 mmap 统计；`tools/_c0_probe.py`、`tools/_evals/c0_result.json`；本地 `tools/_gate_11_9.py --self-test` 9/9 PASS；**未跑训练 / 未跑模型推理 / 未跑评测 / 未碰 test 真值** |
"""

rep(M,
    "\n\n---\n\n## 13. 冲突裁定规则",
    "\n" + CHANGELOG_ROW + "\n---\n\n## 13. 冲突裁定规则",
    "M10 changelog v2.30")

# ---------------------------------------------------------------- INDEX
rep(I,
    "| 已执行（口径订正；IBCFM 预注册 / `NOT ELIGIBLE`；C0–C5 判据冻结；无任何性能数字） |",
    "| **已结案（v2.30）**：口径订正保留；**C0 在远端 train 划分 23 case / 548 帧执行并 FAIL**（`d_hf −0.004513 CI 跨零`、`d_kernel +1.216473 CI 严格为正`、22 正 1 负）→ 按 §5 C0 / C5 与 §11.8 **整族舍弃留档**，C1–C4 永不执行、零性能数字；`ibcfm` 代码留档不派生 |",
    "I1 E39 row")

# ---------------------------------------------------------------- README
rep(R,
    "> **Status · authoritative source (v2.29, 2026-11-08)**",
    "> **Status · authoritative source (v2.30, 2026-11-08)**",
    "R1 status stamp")

rep(R,
    "> As of **MASTER v2.29**, the §9 backlog has **no items left awaiting user",
    "> As of **MASTER v2.30**, the §9 backlog has **no items left awaiting user",
    "R2 backlog stamp")

rep(R,
    """> Current research line remains **diagnosis + falsification + the Oracle upper
> bound 0.1653 + robustness**. As of v2.29, §9 has one **new user-adjudicated
> pre-registration**: E39 (§4.41, IBCFM), status `PRE-REGISTERED / NOT ELIGIBLE`,
> zero training and zero evaluation, criteria C0-C5 frozen before any compute.
> It is a new mechanism category, not a revival of the five v2.28 adjudications,
> and **not a performance result**. See `docs/MASTER.md` §9 / §10 / §15 and
> `docs/experiments/INDEX.md`.""",
    """> Current research line remains **diagnosis + falsification + the Oracle upper
> bound 0.1653 + robustness**. The one v2.29 user-adjudicated pre-registration,
> E39 (§4.41, IBCFM), is **closed in v2.30**: its C0 data premise was executed
> on the remote train split (23 cases / 548 frames, read-only) and **failed**.
> Label rainfall >=2 mm is sparser than radar >=35 dBZ (0.004185 vs 0.028337)
> but spatially **more coherent**, not higher-frequency: the RAPSD difference
> CI crosses zero and the connected-core log difference is strictly positive
> (+1.216473, CI [+0.965871, +1.442821], 22 of 23 cases positive). The whole
> generative conditional-sampling route is **discarded with evidence** under
> MASTER §5 C0 / C5 and §11.8; no training compute was spent. The `ibcfm`
> code is kept as evidence only. See `docs/MASTER.md` §9 / §10 / §15 and
> `docs/experiments/INDEX.md`.""",
    "R3 status body")

# ---------------------------------------------------------------- E39
rep(E,
    "# E39 · 生成式路线重审：口径订正、跨模态任务再认识与 IBCFM 预注册\n",
    "# E39 · 生成式路线重审：口径订正、跨模态任务再认识与 IBCFM 预注册\n\n"
    "> **v2.30 状态更新（2026-11-08 19:48，远端 C0 执行）**：**C0 = FAIL**，\n"
    "> 生成式条件采样方向按 §5 C0 / C5 与 §11.8 **整族舍弃留档**，C1–C4 永不执行。\n"
    "> 下表与 §1–§9 是 **18:03 预注册时的历史状态**，作为留档原样保留；\n"
    "> §10 是 19:46 冻结的操作化细则，§11 是执行结果与裁决。不得写「有望 SOTA」。\n",
    "E1 status banner")

E39_S11 = """

---

## 11. C0 执行结果与裁决（v2.30，2026-11-08 19:48，远端 train 划分）

### 11.1 执行边界（可审计）

| 项 | 事实 |
|---|---|
| 用户授权 | 2026-11-08 19:42「授权再远端跑C0」 |
| 判据顺序 | §5 C0 冻结 **18:03** → §10 操作化细则冻结 **19:46** → 首次执行 **19:48:45**（结果 JSON 的 `executed_at`）。**判据先写后跑成立** |
| 远端 | `wf@183.136.129.105:1003`，`/home/wf/radarNWP`，数据 `/home/wf/data/radarNWP/BJ_mmap`，解释器 `/home/wf/.conda/envs/radar_nwp/bin/python` |
| 命令 | `python tools/remote.py --sync tools/_c0_probe.py -- 'cd /home/wf/radarNWP && ... python tools/_c0_probe.py --out /tmp/c0_result.json'` |
| 边界 | **零训练、零模型推理、零 checkpoint、零 test 真值、零评测器改动**；只读 train 划分的 mmap 数组 |
| 产物 | `tools/_c0_probe.py`、`tools/_evals/c0_result.json`（远端 `/tmp/c0_result.json` 的副本） |

### 11.2 样本与审计（无样本问题）

`split_cases()` 给出 train **23** / val 8 / test 8（62 个 case 目录中 23 个 irregular 被剔除）。
**23 个 train case 全部用上**，共 **548 帧**，`shape` 全 `(301,301)`，CR 索引全 6，
`skipped_no_time_overlap = 0`、`skipped_no_usable_frame = 0`、`notes` 只有
「scipy.ndimage.label, 4-connectivity」与判据失败说明，`error = None`。
JSON 的 `split_audit.files_opened = "train case directories only"`；val / test 只记录首 case 名用于审计。
→ 不存在「删样本 / 挑 case / 换阈值」的空间，也没有触发 §10.4 的 INCONCLUSIVE 条件。

### 11.3 结果（§10.4 四项检查）

| 检查 | 实测 | 要求 | 判定 |
|---|---|---|---|
| 最小样本 | `n_cases = 23` | ≥ 20 | **PASS** |
| **B1 高频能量** | `d_hf = −0.004513`，95 % CI `[−0.011703, +0.001625]`；逐 case 10 正 13 负 | CI 严格为正 | **FAIL**（跨零，且点估计为负） |
| **B2 连通核尺度** | `d_kernel = +1.216473`，95 % CI `[+0.965871, +1.442821]`；逐 case **22 正 1 负** | CI 严格为负 | **FAIL**（明确反向，核约大 `exp(1.216) ≈ 3.37` 倍） |
| 辅助一致性 | `p_label≥2mm = 0.004185` < `p_CR≥35dBZ = 0.028337` | label 更稀疏 | **PASS** |

→ **`VERDICT: C0 FAIL`**（脚本 stdout 与 JSON 一致）。

### 11.4 裁决与影响

1. **§5 C0 自动触发**：「若不成立 → §3.3 的 task novelty 不成立，**立即舍弃整个方向**，
   不投任何训练算力」。**C5** 进一步规定「C0–C4 任一不过 → 方向按 §11.8 舍弃、
   E39 留档、MASTER §10.2 恢复原禁令全文」。
2. **用户第 3 条直觉的数据前提被证否**：label 的强降水确实**更稀疏**（0.42 % vs 2.83 %），
   但**不是高频点状场**——它是空间上更连贯的大块，radar CR 反而更碎、高频能量略高。
   「稀疏」不等于「高频」；把两者混为一谈会使整个生成式动机失效。
3. **§3.3 的 task-level novelty 表述作废**，不得写进摘要；「跨模态 r = +0.17」仍是
   §3.5 的既有事实，但它**不能**推出「因此更适合生成模型」。
4. `models/ibcfm/`、`configs/models/ibcfm.yaml`、`tests/test_ibcfm.py`、
   `scripts/train.py` 的 `--loss ibcfm` 分支**保留留档**（§11.8「删文件＝销毁证据」），
   但**不训练、不评测、不派生、不包装成论文贡献**。C1–C4 永不执行。
5. 本节结果不是 CSI / BIAS 性能数字，**不进 §10.1 性能层**。它与 E33 的
   「幅度 39 % / 排布 61 %」是不同量度（一个是真值高频替换的信息缺口，
   一个是 train 划分的原始场统计），**不得互相引用或混为一谈**。

### 11.5 自评（§11.8）

- **① 与任务目标的关系**：**有关**。用户已把目标改为「TGRS 可发表的 SOTA / novelty」；
  本轮用一次只读统计、零训练算力，否掉了三条重审中最关键的数据前提，避免把
  C2–C4 的大规模训练投在一个不成立的前提上。
- **② 带数字的证据**：23 train case / 548 帧 / 零跳过；`d_hf −0.004513 CI [−0.011703,+0.001625]`；
  `d_kernel +1.216473 CI [+0.965871,+1.442821]`（22 正 1 负）；`0.004185` vs `0.028337`；
  10000 次 case 级 bootstrap，`seed = 20261108`。
- **③ 结论与去向**：**舍弃整个生成式条件采样方向**，留档保留。
  **不再做什么**：不再以「label 高频 / 点状 / 跨模态尖峰」为由立项；不再换采样器、
  强度分带、生成模型或阈值决策派生新实验；不投 C1–C4；不把 `ibcfm` 代码包装成贡献；
  不改 §5 / §10 判据、不以「换个统计量可能显著」重跑 C0；不用 test 真值复核。

### 11.6 §11.9 资格门核对（第十一项，续）

本轮**无性能主张**（是 train 划分的数据描述统计，不是候选方法分数），按 **P1**
声明「不适用」；`tools/_gate_11_9.py --self-test` 复跑仍为 `anchors: 9/9` /
`ANCHOR: PASS`（工具与阈值一字未改）。E39 **永不进入 §10.1**。

### 11.7 工程留痕

`tools/remote.py --download` 不带 `--out` 时会把远端文件按 basename 落到**仓库根**
（而不是 `--out` 指定的路径）；本次先落到 `c0_result.json` 再归位到
`tools/_evals/c0_result.json`。后续远端取数一律显式写 `--out`。
"""

rep(E,
    "- **`C0: PASS` ≠ 模型立项通过**：E39 状态仍是 `PRE-REGISTERED / NOT ELIGIBLE`，\n"
    "  C1–C4 全未跑，不得写任何性能结论、不得写「有望突破地板」。",
    "- **`C0: PASS` ≠ 模型立项通过**：E39 状态仍是 `PRE-REGISTERED / NOT ELIGIBLE`，\n"
    "  C1–C4 全未跑，不得写任何性能结论、不得写「有望突破地板」。"
    "\n> **执行后注（19:48）**：本条写的是 PASS 分支；**实际结果为 FAIL**，见 §11。判据未改。\n"
    + E39_S11,
    "E2 section 11")

for p, t in cache.items():
    p.write_bytes(t.replace("\n", "\r\n").encode("utf-8"))
print(f"SYNC: PASS ({len(cache)} files: {', '.join(str(p.relative_to(ROOT)) for p in cache)})")
