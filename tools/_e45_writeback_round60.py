#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 60 -- writeback (idempotent).

Idempotency rule (established in earlier rounds): the marker is the exact
text being inserted; if the marker is already present in the target file,
the file is left untouched ("unchanged").  A missing anchor is an ERROR --
never a silent skip.

This round produces NO new criterion and NO verdict change, so MASTER is
NOT bumped (stays v2.48) and docs/ is NOT touched.  New evidence goes to
HANDOVER_E45.md, the upper MEMORY.md status section, the daily logs and
the automation memory.

The MEMORY.md status lines are edited with a regex rather than a fixed
string: that file gets compacted by the runtime, and a fixed anchor has
been observed to vanish underneath the writeback.
"""
import io
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPPER = r"R:\1radarNWPprediction\.workbuddy\memory"
PROJ = os.path.join(REPO, ".workbuddy", "memory")
AUTOM = os.path.join(PROJ, "automations",
                     "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

REPORT = io.StringIO()
ERRORS = []

HANDOVER = os.path.join(UPPER, "HANDOVER_E45.md")
MEMORY = os.path.join(UPPER, "MEMORY.md")
D14_U = os.path.join(UPPER, "2026-11-14.md")
D14_P = os.path.join(PROJ, "2026-11-14.md")
D11_U = os.path.join(UPPER, "2026-11-11.md")
AUTOM_F = AUTOM

# ----------------------------------------------------------------- texts
HANDOVER_BLOCK = """
**\u3244（2026-11-14 第 60 轮（决策价值轴的首次判读））**——脚本 `tools/_e45_verify_round60.py`（只读本地，**POS 18/18 VERDICT: PASS**，产物 `tools/_evals/e45/round60_decision_value.json`）：前 59 轮判过聚合层／机制探针／S1 结构量／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样稳定性／聚合算子＋评测契约／训练动力学／晋级规则标定共十七层，**全部问的是「准不准」**，**从未问过决策者真正要问的那句「我该部署哪一臂」**。本轮第一次把 E45 放到**决策价值轴**（成本损失 / CNL，Murphy 1977；Jolliffe & Stephenson 2008 第 9 章）上判读，**裁决零变化**（事后诊断，**不构成新判据**：S3 护栏固定 thr=2.0 mm、复制单元固定为 seed），得六条新事实：

1. **先立模型与可复算性**：以真值基率 `p`（thr=2.0 mm 处 **p = 0.014982**）与**条件虚警率 POFD = F/(F+CN)** 构造 `cost(r) = r·(1−POD)·p + POFD·(1−p)`，其中 `r = C_M/C_F` 是**漏报成本 ÷ 虚警成本**；`DC(r) = cost_arm(r) − cost_ctrl(r)` 对 r **线性**，故存在**盈亏平衡成本比** `r* = (1−p)·ΔPOFD/(p·ΔPOD)`（`ΔPOD > 0` 时该臂只在 **r > r*** 更便宜，`ΔPOD < 0` 时只在 **r < r*** 更便宜）。**真值域共享**⇒ p 在 21 格上完全相同（离散度 **0.000e+00**）；**自校**：由 21 份窗表复算已发布的 CSI/BIAS/POD/FAR **336 次比较 max|diff| = 0.000e+00**；**代数自校**：`DC(r*) = 0` 在 18 个（臂×seed）格上成立（max|DC(r*)| = **2.776e−17**）。
   ⚠️ **本轮实际踩到的坑（可复用）**：已发布的 `FAR` 是标准定义 **F/(H+F)**，而成本模型需要**条件 POFD = F/(F+CN)** ⇒ **两者不可混用**，首版把 POFD 直接与已发布 FAR 比对，得到 max|diff| **0.924** 的假失败（真值 CSI/BIAS/POD 逐位相同）。**凡把 FAR 送进成本/概率模型，必须先确认它是哪一种 FAR。**

2. **最硬的一条结论：在任何现实成本比下，六臂都不比对照更划算**。零模型（对照臂 B0 自己换种子，纯 seed 噪声、无机制）给出逐 r 的 |Δcost| 地板（0.00485 @ r=0.1 … 0.00176 @ r=3 … 0.00886 @ r=10）。⇒ **在 r ≤ 3（漏报成本不超过虚警成本 3 倍）的整个区间里，没有任何一臂的配对中位 |Δcost| 越过地板**（越界清单为空）；**唯一越界的格子是 A4b @ r=10**（漏报成本 10 倍于虚警成本）：配对中位 **−0.01628**，但**通过率只有 2/3**（逐 seed −0.01628／−0.02164／**+0.02098**，第三个种子翻正）、**效应只有地板的 1.84 倍**，且 **A4b 的 `r*` 中位 5.913 只有零模型自身 |r*| 尺度 3.605 的 1.64 倍** ⇒ **连「交叉点落在哪里」都与种子噪声不可分**。⇒ **裁决：六臂在决策价值轴上同样判负；不投十 seed 算力。**

3. **决策轴独立复现了「增益＝面积/幅度效应」，而且给出了第一次量化的「盈亏平衡成本比」**。六臂 `r*` 中位：**A1 2.666｜A2 3.124｜A3a 2.609｜A3b 2.845｜A4 3.878｜A4b 5.913**。⇒ **凡是要靠这六臂挣回价值，都必须先假定「漏报比虚警贵 2.6–5.9 倍」**；而涨分的两臂正是 POD/BIAS 同时爆表的两臂：**A4b 的 ΔPOD 中位 +0.34144**（POD 0.5054／0.4692 vs 对照 0.1639／0.1158）**同时 ΔPOFD +0.03178** ⇒ **它买命中率的方式就是多报**，与 S3 `GAIN_WAS_AREA_OR_AMPLITUDE`、第 51 轮 `ΔF/真值`、第 54 轮列联表域面积配平**四台仪器同向，本轮是第五台**（仪器不同：从成本比空间看同一件事）。

4. **A4 在决策空间里是最不稳的一臂，`r*` 连符号都翻**：`r*` 逐 seed **3.878／24.206／−3.779**（第三格为负 ⇒ `ΔPOD < 0` 与 `ΔPOFD < 0`，即那一格**在所有 r 下都更贵**），**成本剖面一致性仅 1/3**（`crossover`／`more_expensive_at_all_r`／`cheaper_at_all_r` 三种各一格）⇒ 与第 40 轮「A4 的 +0.0180 里 99.2% 来自 seed 0」在**成本空间**上同向。**A1／A2 则是另一类不稳**：`ΔPOD` 符号一致性 **1/3／0/3**（两臂基本都在少报），其低 r 区的「更便宜」只是**少报**的副产品（把漏报按 1 倍虚警计价时的记账效应），不是机制收益。

5. **配对口径与差值口径在决策轴上再次反向（A3a／A3b）**：r=10 处配对中位 **A3a −0.00302／A3b −0.00506**（看似更便宜），但**臂级差值口径为 +0.00218／+0.00091**（更贵）⇒ 与第 44 轮「种子对齐效应」、第 49 轮「时效维度符号翻转」**同族的第四例**（换到成本比维度也能造出符号翻转）⇒ **A3a／A3b 在决策轴上仍然只是对齐效应，不重开**。**只有 A4b 在 r=10 上两种口径同向**（配对 −0.01628／差值 −0.01567），但它正是通过率 2/3、种子 2 完全崩塌（`ΔPOD` −0.1919、POD 0.0163 vs 对照 0.2083）的那一臂家族 ⇒ **不得择优**。

6. **三条新登记的可复用纪律**：① **「该不该部署」必须连同成本比一起报**——CSI/BIAS 表无法回答这个问题，而 `r*` 才是决策相关的量；且 **`r*` 本身是带噪统计量，必须与零模型自身的 |r*| 尺度并列报**（否则「交叉点在 5.9」会被误读成机制结论）。② **FAR 有两个定义（标准 F/(H+F) 与条件 POFD = F/(F+CN)），送进任何概率/成本模型前必须先确认**（本轮因此得到一次假失败）。③ **成本比与阈值都是「口径依赖」**：逐阈值 `r*` 中位 **thr=0.5：A4b 2.984｜thr=2.0：5.913｜thr=4.0：9.186**，且 **A4 在 thr=4.0 处 `r*` 翻成 −8.823** ⇒ **「哪一臂更划算」随工作点改变，任何跨口径的「更好」表述都不成立**。

**附带读数**：对照臂自身的成本损失技巧 `CS(r)` 中位 **0.6039（r=0.1）→ 0.4002（r=1）→ 0.1965（r=10）**（随成本比单调衰减）⇒ **对照臂本身是有真实决策价值的，E45 六臂没有一项能超过它**。**场次级自助（8 场次，仅作内部一致性，不得当跨种子证据）**：A4b @ r=4 的三个种子 CI 全部不含零，A3b／A4 @ r≤2 的种子 0／1 不含零——但**复制单元仍是 seed（n=3）**，按第 46／48 轮纪律**不得**据此宣称任何一臂成立。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）、**合格臂集合仍为空**；⑤ **重建自校**：336 次窗表→已发布指标复算 max|diff| = 0.000e+00、七臂中位 CSI 复算 7/7 逐位一致、21 格 `window_keys` 一致同序、`DC(r*)=0` 代数自校通过；⑥ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
"""

MEM_BULLET = (
    "- **第 60 轮新增一条轴（决策价值轴 ＋ FAR 定义坑）**：① **决策价值轴（首次判读）**＝把 E45 放到成本损失/"
    "CNL 空间（Murphy 1977；Jolliffe & Stephenson 2008 第 9 章）：`cost(r) = r·(1−POD)·p + POFD·(1−p)`，"
    "`r = C_M/C_F`（漏报成本÷虚警成本），真值基率 **p = 0.014982**（21 格共享，离散度 0.000e+00），"
    "`DC(r)` 对 r 线性 ⇒ **盈亏平衡成本比 `r* = (1−p)·ΔPOFD/(p·ΔPOD)`**。**结论：r ≤ 3（漏报≤虚警 3 倍）"
    "的整个区间，六臂没有任何一臂比对照更划算**（零模型＝B0 换种子，地板 0.00485@r=0.1 … 0.00886@r=10）；"
    "**唯一越界格 A4b @ r=10**（配对中位 −0.01628）但**通过率仅 2/3**、效应仅地板 1.84 倍、"
    "且 `r*` 中位 **5.913** 只有零模型自身 |r*| 尺度 **3.605** 的 1.64 倍 ⇒ **连交叉点都与种子噪声不可分**。"
    "② **六臂 `r*` 中位 A1 2.666｜A2 3.124｜A3a 2.609｜A3b 2.845｜A4 3.878｜A4b 5.913** ⇒ "
    "**要靠这六臂挣回价值必须先假定「漏报比虚警贵 2.6–5.9 倍」**；而 **A4b 的 ΔPOD +0.34144 与 ΔPOFD +0.03178 "
    "同现**（买命中率＝多报）⇒ 与 S3、第 51 轮、第 54 轮**四台仪器同向，本轮是第五台（从成本比空间看同一件事）**。"
    "③ **A4 在决策空间里最不稳**（`r*` 逐 seed 3.878／24.206／**−3.779**，成本剖面一致性 1/3）；"
    "**A3a／A3b 在 r=10 配对为负（−0.00302／−0.00506）而臂级差值为正（+0.00218／+0.00091）** ⇒ "
    "配对/差值两口径再次反向 ⇒ **仍是种子对齐效应**（第 44/49 轮同族的第四例）。**裁决零变化、MASTER 不升版。**\n"
)

MEM_STATUS = (
    "- **E45 状态更新（2026-11-14 第 60 轮）**：远端 E45 全部 `COMPLETED`（提交器 273、训练 275/283、"
    "评测 274/282/290、汇总 291、诊断 297/298），21 份评测齐、13 份 json 齐全、队列空、远端↔本地 md5 13/13 全同 "
    "⇒ **无新活动、未改远端**。**裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力；MASTER 仍 v2.48**。"
    "本轮新增证据＝**决策价值轴**（`tools/_e45_verify_round60.py`，POS 18/18 VERDICT: PASS）："
    "① **成本损失/CNL 空间首次判读**——`cost(r)=r·(1−POD)·p+POFD·(1−p)`（`r=C_M/C_F`，基率 p=0.014982），"
    "**r ≤ 3 的整个区间六臂无一臂比对照更划算**（零模型地板已给），**唯一越界格 A4b @ r=10 通过率仅 2/3、"
    "效应仅地板 1.84 倍、`r*` 中位 5.913 只有零模型 |r*| 3.605 的 1.64 倍** ⇒ **连交叉点都与种子噪声不可分**；"
    "② **六臂 `r*` 中位 2.666–5.913** ⇒ 要靠这六臂挣钱必须先假定漏报贵 2.6–5.9 倍；**A4b ΔPOD +0.34144 "
    "与 ΔPOFD +0.03178 同现**（买命中率＝多报）⇒ 与 S3/第 51 轮/第 54 轮四台仪器同向；③ **A4 `r*` 连符号翻"
    "（3.878／24.206／−3.779）**，**A3a／A3b 配对与差值口径反向** ⇒ 仍是种子对齐效应。"
    "④ **本轮踩坑：已发布 FAR 是标准 F/(H+F)，成本模型要条件 POFD=F/(F+CN)，不可混用**（首版因此得到 "
    "max|diff| 0.924 的假失败）。**新登记三条纪律：决策价值必须连同成本比与零模型 |r*| 尺度一起报；"
    "FAR 定义必须先确认；成本比与阈值都是口径依赖**。待办仍是三件实现缺陷修复 ＋ 重提机制族。\n"
)

D14_BLOCK = """
### 2026-11-14｜第 60 轮（决策价值轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与第 50–59 轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮唯一远端动作是只读状态查询，未 sbatch、未下载、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。
- **新证据**——`tools/_e45_verify_round60.py`（只读本地，**POS 18/18 VERDICT: PASS**，产物 `tools/_evals/e45/round60_decision_value.json`）：前 59 轮问的全是「准不准」，**从未问过「该部署哪一臂」** ⇒ 本轮首次在**决策价值轴（成本损失/CNL）**判读，**裁决零变化**。① **模型与可复算性**：`cost(r)=r·(1−POD)·p+POFD·(1−p)`（`r=C_M/C_F`，基率 p=0.014982，21 格共享、离散度 0.000e+00）；`DC(r)` 对 r 线性 ⇒ **盈亏平衡成本比 `r*=(1−p)·ΔPOFD/(p·ΔPOD)`**；**自校**：336 次窗表→已发布指标复算 max|diff| **0.000e+00**、`DC(r*)=0` 代数自校 max **2.776e−17**。② **主结论：r ≤ 3 的整个区间，六臂无一臂比对照更划算**（零模型＝B0 换种子，地板 0.00485@r=0.1 … 0.00886@r=10，越界清单为空）；**唯一越界格 A4b @ r=10**（配对中位 −0.01628）但**通过率 2/3**、效应仅地板 **1.84 倍**、`r*` 中位 **5.913** 仅零模型 |r*| 尺度 **3.605** 的 **1.64 倍** ⇒ **连交叉点都与种子噪声不可分**。③ **六臂 `r*` 中位 A1 2.666｜A2 3.124｜A3a 2.609｜A3b 2.845｜A4 3.878｜A4b 5.913** ⇒ 要靠这六臂挣钱须先假定「漏报比虚警贵 2.6–5.9 倍」；**A4b 的 ΔPOD +0.34144 与 ΔPOFD +0.03178 同现**（买命中率＝多报）⇒ 与 S3、第 51 轮、第 54 轮**四台仪器同向，本轮是第五台**。④ **A4 决策空间最不稳**（`r*` 逐 seed 3.878／24.206／**−3.779**，成本剖面一致性 1/3）；**A1／A2 的 ΔPOD 符号一致性 1/3／0/3**，其低 r「更便宜」是少报的记账副产品。⑤ **A3a／A3b 在 r=10 配对为负（−0.00302／−0.00506）而臂级差值为正（+0.00218／+0.00091）** ⇒ 配对/差值两口径再次反向 ⇒ 仍是种子对齐效应（第 44/49 轮同族的第四例）；**只有 A4b 两种口径同向，但它正是通过率 2/3、种子 2 崩塌的那一臂家族** ⇒ 不得择优。⑥ **本轮踩坑：已发布 FAR 是标准 F/(H+F)，成本模型要条件 POFD=F/(F+CN)，不可混用**（首版得 max|diff| 0.924 的假失败）⇒ **新登记三条纪律：决策价值必须连同成本比与零模型 |r*| 尺度一起报；FAR 定义必须先确认；成本比与阈值皆口径依赖**（逐阈值 `r*`：A4b 2.984@0.5 → 5.913@2.0 → 9.186@4.0；A4 在 4.0 处翻成 −8.823）。
- **附带读数**：对照臂自身成本损失技巧 `CS(r)` 中位 **0.6039（r=0.1）→ 0.4002（r=1）→ 0.1965（r=10）** ⇒ 对照臂有真实决策价值，**E45 六臂没有一项能超过它**。**场次级自助（8 场次，仅内部一致性）**：A4b @ r=4 三种子 CI 全不含零、A3b／A4 @ r≤2 种子 0／1 不含零——复制单元仍是 seed（n=3），**不得**当跨种子证据。
- **守卫**：S3 的 12 个区间仍全部跨零、`crosses_zero` 12/12 为 true、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；由 538 窗原始表复算 21 格 `CSI@2.0` 与七臂中位 CSI **max|diff| = 0.000e+00**；21 格 `window_keys` 一致同序。
- **裁决零变化**：六臂判负、不投十 seed 算力；**MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**。
"""

D14_P_BLOCK = """
### 2026-11-14｜第 60 轮（决策价值轴的首次判读）：状态复核（无远端改动，裁决不变）

- **远端**：273 `COMPLETED`（05:04:04）、275/283 各 7/7、274/282/290、291、297/298 全部 `COMPLETED`；21 份评测齐、13 份 json 齐全、队列空、无 E46+ 目录；远端↔本地 **md5 13/13 全同**。**未下载、未 sbatch、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS`。
- **新证据**——`tools/_e45_verify_round60.py`（只读本地，**POS 18/18 VERDICT: PASS**，产物 `tools/_evals/e45/round60_decision_value.json`）：**决策价值轴（成本损失/CNL）首次判读** ⇒ **裁决零变化**。① 模型 `cost(r)=r·(1−POD)·p+POFD·(1−p)`（`r=C_M/C_F`，基率 p=0.014982，21 格共享）、盈亏平衡成本比 `r*=(1−p)·ΔPOFD/(p·ΔPOD)`；自校 336 次复算 max|diff| **0**、`DC(r*)=0` max **2.776e−17**。② **r ≤ 3 全区间六臂无一臂比对照更划算**（零模型地板已给）；唯一越界 **A4b @ r=10** 但**通过率 2/3**、效应仅地板 1.84 倍、`r*` 5.913 仅零模型 |r*| 3.605 的 1.64 倍 ⇒ **连交叉点都与种子噪声不可分**。③ 六臂 `r*` 中位 2.666–5.913 ⇒ 要靠它们挣钱须先假定漏报贵 2.6–5.9 倍；**A4b ΔPOD +0.34144 与 ΔPOFD +0.03178 同现**（买命中率＝多报）⇒ 与 S3/第 51/54 轮**四台仪器同向，本轮是第五台**。④ **A4 `r*` 连符号翻（3.878／24.206／−3.779）**，成本剖面一致性 1/3；**A3a／A3b 在 r=10 配对负、臂级差值正** ⇒ 仍是种子对齐效应。⑤ **坑：已发布 FAR = 标准 F/(H+F)，成本模型要 POFD = F/(F+CN)，不可混用**（首版假失败 max|diff| 0.924）。⑥ **新登记三条纪律**：决策价值须连同成本比与零模型 |r*| 尺度报；FAR 定义先确认；成本比/阈值皆口径依赖。
- **守卫**：S3 十二区间仍全部跨零、`crosses_zero` 12/12 为 true、六臂 `s3_verdict` 不变、**合格臂集合仍为空**；21 格 `CSI@2.0` 与七臂中位 CSI 复算 **max|diff| = 0.000e+00**。
- **裁决零变化**：六臂判负、不投十 seed 算力；**MASTER 不升版**。
"""

D11_LINE = ("- 2026-11-14 第 60 轮（决策价值轴，只读本地，POS 18/18 VERDICT: PASS，"
            "产物 `tools/_evals/e45/round60_decision_value.json`）：**裁决不变**（六臂判负、"
            "不投十 seed 算力、MASTER v2.48 不升版）。**决策价值轴（成本损失/CNL）首次判读**："
            "`cost(r)=r·(1−POD)·p+POFD·(1−p)`（`r=C_M/C_F`，基率 p=0.014982，21 格共享）、"
            "盈亏平衡成本比 `r*=(1−p)·ΔPOFD/(p·ΔPOD)`；自校 336 次复算 max|diff| 0、"
            "`DC(r*)=0` 代数自校 max 2.776e−17。**① r ≤ 3 全区间六臂无一臂比对照更划算**"
            "（零模型＝B0 换种子）；**唯一越界格 A4b @ r=10**（配对中位 −0.01628）但**通过率 2/3**、"
            "效应仅地板 1.84 倍、`r*` 中位 **5.913** 仅零模型自身 |r*| 尺度 **3.605** 的 1.64 倍 "
            "⇒ **连交叉点都与种子噪声不可分**。**② 六臂 `r*` 中位 A1 2.666｜A2 3.124｜A3a 2.609｜"
            "A3b 2.845｜A4 3.878｜A4b 5.913** ⇒ 要靠这六臂挣钱须先假定「漏报比虚警贵 2.6–5.9 倍」；"
            "**A4b 的 ΔPOD +0.34144 与 ΔPOFD +0.03178 同现**（买命中率＝多报）⇒ 与 S3、第 51 轮、"
            "第 54 轮**四台仪器同向，本轮是第五台**。**③ A4 在决策空间最不稳**（`r*` 逐 seed "
            "3.878／24.206／**−3.779**，成本剖面一致性 1/3）；**A3a／A3b 在 r=10 配对为负"
            "（−0.00302／−0.00506）而臂级差值为正（+0.00218／+0.00091）** ⇒ 配对/差值两口径再次反向 "
            "⇒ 仍是种子对齐效应（第 44/49 轮同族的第四例）。**④ 本轮踩坑：已发布 FAR 是标准 "
            "F/(H+F)，成本模型要条件 POFD=F/(F+CN)，不可混用**（首版得 max|diff| 0.924 的假失败）。"
            "**⑤ 新登记三条纪律**：决策价值必须连同成本比与零模型 |r*| 尺度一起报；FAR 定义必须先确认；"
            "成本比与阈值皆口径依赖。**未 sbatch、未改远端状态、未动 `docs/`。**\n")

AUTOM_BLOCK = """
### 2026-11-14｜第 60 轮（决策价值轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与第 50–59 轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮唯一远端动作是只读状态查询，未 sbatch、未下载、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；无 E46+ 目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。
- **新证据**——`tools/_e45_verify_round60.py`（只读本地，**POS 18/18 VERDICT: PASS**，产物 `tools/_evals/e45/round60_decision_value.json`）：前 59 轮问的全是「准不准」，**从未问过「该部署哪一臂」** ⇒ 本轮首次在**决策价值轴（成本损失/CNL）**判读，**裁决零变化**。① **模型**：`cost(r)=r·(1−POD)·p+POFD·(1−p)`（`r=C_M/C_F`，基率 p=0.014982，21 格共享、离散度 0.000e+00），`DC(r)` 对 r 线性 ⇒ **盈亏平衡成本比 `r*=(1−p)·ΔPOFD/(p·ΔPOD)`**；**自校**：336 次窗表→已发布指标复算 max|diff| **0.000e+00**、`DC(r*)=0` 代数自校 max **2.776e−17**。② **主结论：r ≤ 3 全区间六臂无一臂比对照更划算**（零模型＝B0 换种子，地板 0.00485@r=0.1 … 0.00886@r=10，越界清单为空）；**唯一越界格 A4b @ r=10**（配对中位 −0.01628）但**通过率 2/3**、效应仅地板 **1.84 倍**、`r*` 中位 **5.913** 仅零模型 |r*| 尺度 **3.605** 的 **1.64 倍** ⇒ **连交叉点都与种子噪声不可分**。③ **六臂 `r*` 中位 A1 2.666｜A2 3.124｜A3a 2.609｜A3b 2.845｜A4 3.878｜A4b 5.913** ⇒ 要靠这六臂挣钱须先假定漏报贵 2.6–5.9 倍；**A4b 的 ΔPOD +0.34144 与 ΔPOFD +0.03178 同现**（买命中率＝多报）⇒ 与 S3、第 51 轮、第 54 轮**四台仪器同向，本轮是第五台**。④ **A4 `r*` 连符号翻**（3.878／24.206／**−3.779**，成本剖面一致性 1/3）；**A3a／A3b 在 r=10 配对负、臂级差值正** ⇒ 仍是种子对齐效应；**只有 A4b 两种口径同向，但它正是通过率 2/3、种子 2 崩塌的那一臂家族** ⇒ 不得择优。⑤ **坑：已发布 FAR 是标准 F/(H+F)，成本模型要条件 POFD=F/(F+CN)，不可混用**（首版得 max|diff| 0.924 的假失败）⇒ **新登记三条纪律**：决策价值须连同成本比与零模型 |r*| 尺度一起报；FAR 定义先确认；成本比/阈值皆口径依赖。
- **附带读数**：对照臂 `CS(r)` 中位 **0.6039（r=0.1）→ 0.4002（r=1）→ 0.1965（r=10）** ⇒ 对照臂有真实决策价值，六臂无一超过它。**场次级自助（8 场次，仅内部一致性）**：A4b @ r=4 三种子 CI 全不含零、A3b／A4 @ r≤2 种子 0／1 不含零——复制单元仍是 seed（n=3），不得当跨种子证据。
- **守卫**：S3 的 12 个区间仍全部跨零、`crosses_zero` 12/12 为 true、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；由 538 窗原始表复算 21 格 `CSI@2.0` 与七臂中位 CSI **max|diff| = 0.000e+00**；21 格 `window_keys` 一致同序。
- **裁决零变化**：六臂判负、不投十 seed 算力；**MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**。
"""


def read(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def append_once(path, block, marker, label):
    try:
        txt = read(path)
    except Exception as exc:
        ERRORS.append("%s: cannot read (%s)" % (label, exc))
        return False
    if marker in txt:
        REPORT.write("  unchanged  %s (marker already present)\n" % label)
        return False
    write(path, txt.rstrip("\n") + "\n" + block)
    REPORT.write("  inserted   %s\n" % label)
    return True


def re_sub_once(path, pattern, repl, label):
    """Regex-based replace; the anchor may have been compacted away."""
    try:
        txt = read(path)
    except Exception as exc:
        ERRORS.append("%s: cannot read (%s)" % (label, exc))
        return False
    if repl in txt:
        REPORT.write("  unchanged  %s (already updated)\n" % label)
        return False
    new, n = re.subn(pattern, repl.replace("\\", "\\\\"), txt, count=1,
                     flags=re.MULTILINE)
    if n != 1:
        ERRORS.append("%s: pattern not found: %r" % (label, pattern))
        return False
    write(path, new)
    REPORT.write("  updated    %s\n" % label)
    return True


def insert_bullet(path, bullet, label):
    try:
        txt = read(path)
    except Exception as exc:
        ERRORS.append("%s: cannot read (%s)" % (label, exc))
        return False
    if bullet in txt:
        REPORT.write("  unchanged  %s (bullet already present)\n" % label)
        return False
    for anchor in ("- **第 58 轮新增两条轴（训练动力学轴 ＋ 契约/来源再验证）**",
                   "- **第 57 轮新增两条轴（聚合算子轴 ＋ 评测契约/来源审计）**",
                   "- **三条定位**",
                   "- **第 8–"):
        if anchor in txt:
            write(path, txt.replace(anchor, bullet + anchor, 1))
            REPORT.write("  inserted   %s (anchor %r)\n" % (label, anchor[:28]))
            return True
    ERRORS.append("%s: no usable anchor for the bullet" % label)
    return False


def main():
    changes = 0

    # 1) HANDOVER_E45.md -- new section appended (round 59 block is the last
    #    section in the file; appending keeps the newest round at the end).
    changes += append_once(HANDOVER, HANDOVER_BLOCK,
                           "**\u3244（2026-11-14 第 60 轮（决策价值轴的首次判读））**",
                           "HANDOVER_E45.md")

    # 2) upper MEMORY.md status section
    changes += re_sub_once(
        MEMORY,
        r"- \*\*第 8\u2013\d+ 轮独立复核结论完全一致、无新增偏离\*\*\uff1a",
        "- **第 8\u201360 轮独立复核结论完全一致、无新增偏离**：",
        "MEMORY.md (round range)")
    changes += re_sub_once(
        MEMORY,
        r"第 40\u2013\d+ 轮新增的均为\*\*事后诊断新轴\*\*",
        "第 40\u201360 轮新增的均为**事后诊断新轴**",
        "MEMORY.md (new-axis range)")
    changes += re_sub_once(
        MEMORY,
        r"第 8\u2013\d+ 轮复核结论一致",
        "第 8\u201360 轮复核结论一致",
        "MEMORY.md (version-line round count)")
    changes += re_sub_once(
        MEMORY,
        r"（逐 seed 分解、maxratio 判据判读、跨实验复现、派生布尔分支、多阈值口径、S1 结构量、窗级／场次级／时效／强度·难度／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样稳定性／聚合算子与评测契约／训练动力学）",
        "（逐 seed 分解、maxratio 判据判读、跨实验复现、派生布尔分支、多阈值口径、S1 结构量、窗级／场次级／时效／强度·难度／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样稳定性／聚合算子与评测契约／训练动力学／晋级规则标定／决策价值）",
        "MEMORY.md (axis list)")
    changes += re_sub_once(
        MEMORY,
        r"详见 `HANDOVER_E45\.md` §4\u2464[^\s]{0,14} 与逐日日志。",
        "详见 `HANDOVER_E45.md` §4⑤–㉔ 与逐日日志。",
        "MEMORY.md (handover pointer)")
    changes += insert_bullet(MEMORY, MEM_BULLET, "MEMORY.md (round-60 bullet)")

    # 3) the per-round status paragraph: replace the previous round's line
    changes += re_sub_once(
        MEMORY,
        r"^- \*\*E45 状态更新（2026-11-14 第 \d+ 轮）\*\*：.*$",
        MEM_STATUS.rstrip("\n"),
        "MEMORY.md (status paragraph)")

    # 4) daily logs
    changes += append_once(D14_U, D14_BLOCK,
                           "### 2026-11-14｜第 60 轮（决策价值轴的首次判读）",
                           "2026-11-14.md (upper)")
    changes += append_once(D14_P, D14_P_BLOCK,
                           "### 2026-11-14｜第 60 轮（决策价值轴的首次判读）",
                           "2026-11-14.md (project)")
    changes += append_once(D11_U, D11_LINE,
                           "- 2026-11-14 第 60 轮（决策价值轴，只读本地，POS 18/18 VERDICT: PASS，",
                           "2026-11-11.md (upper)")
    changes += append_once(AUTOM_F, AUTOM_BLOCK,
                           "### 2026-11-14｜第 60 轮（决策价值轴的首次判读）",
                           "automation memory")

    print(REPORT.getvalue())
    if ERRORS:
        print("ERRORS (%d):" % len(ERRORS))
        for e in ERRORS:
            print("  - %s" % e)
    print("writeback: %d file(s) changed" % changes)
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
