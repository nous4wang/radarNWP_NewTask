#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 58 -- writeback (idempotent).

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
**\u3252（2026-11-14 第 58 轮（训练动力学轴的首次判读））**——脚本 `tools/_e45_verify_round58.py`（只读本地，**POS 25/25 VERDICT: PASS**，产物 `tools/_evals/e45/round58_training_dynamics.json`）：前 57 轮判过聚合层／机制探针／S1 结构量／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样稳定性／聚合算子＋评测契约共十四层，**全部读的是评测产物**；而 **21 份训练日志（7 臂 × 3 seed 的 Slurm 作业 stdout）从未被任何一轮打开过** —— 它是唯一记录「模型被拟合时到底发生了什么」的产物，且**直接关系到两件未决的实现缺陷**（待办① 量通道 `1e-12` 护栏、待办② 幅度截断没作用到输出场）与**一条从未写进冻结协议的选 epoch 规则**。本轮首次判读，**裁决零变化**，得七条新事实：

1. **训练损失在 21/21 格严格单调下降（ep0 > ep1 > ep2），训练段零 NaN/inf，而它与技能几乎零相关**：`r(ep0 损失, CSI@2.0)` = **−0.0895**（21 格）／**−0.1138**（18 臂）；`r(ep2/ep0 损失比, CSI@2.0)` = +0.2944／+0.3063。⇒ **「last.pt 是崩塌端」这件事在训练损失上完全看不见**（崩塌是纯技能侧现象）；**最刺眼的一格**：**A4b_s2 的 `CSI@2.0 = 0.015169` 是全表最低（地板 0.0924 的 1/6），而它的训练损失 0.9740 → 0.9147 仍单调下降、量级与对照相同** ⇒ **训练损失不能当模型选择依据、也不能当训练健康度指标**。
2. **「ep00 ＝ 峰值端」这条规则在 E45 内部不可验证**：21 格的损失 argmin **全部**落在 ep02（损失最低），而 ep02 恰是技能崩塌端 ⇒ **损失曲线与技能曲线给出的「最佳」是相反的**；E45 每格**只评过 ep00 这一点**，没有任何一格评过 ep01/ep02 ⇒ **新登记待办④**：下一族机制要么把「选 epoch」规则写进预注册，要么用 sbatch 诊断作业显式评 ep00/ep01/ep02 三点（**需另开作业，不在本轮**）。
3. **拓扑两臂把训练损失系统性抬高、却把中位技能压到低于对照**：ep0 损失中位 **B0 0.9213｜A1 0.9231｜A2 0.9233｜A3a 1.0125（+9.9%）｜A3b 1.0148（+10.1%）｜A4 0.9392｜A4b 0.9525**；而中位 `CSI@2.0` **B0 0.121817 > A3b 0.116055 > A3a 0.110015** ⇒ **拓扑项让目标更难，却没有换来更高的技能**（与第 44 轮「种子对齐效应」、第 53 轮「机制生效 ≠ 分数受益」同向，本轮是**从损失侧**的独立佐证）。
4. **训练期首次看到机制自己的面积账（本轮最硬的一条新证据）**：A3a／A3b 的 `topo_pred_area ÷ topo_truth_area` 中位 **A3a 1.9094／1.5635／1.7709（s0/s1/s2）、A3b 2.0152／1.6961／1.8591** ⇒ **机制自己在训练期就在报告：预测面积是真值面积的 1.6–2.0 倍** ⇒ **过预报不是评测期才出现的，而是训练期就在发生**；与 S3 `GAIN_WAS_AREA_OR_AMPLITUDE`、第 51 轮 `ΔF/真值`、第 54 轮列联表域面积配平**四台仪器同向**。⚠️ **A3b（带 `--topo-area-normalize`）的比值并不比 A3a 小 ⇒ 预注册的「面积归一化」没有修掉面积失配**。口径边界：A3b 的面积量是归一化后的，**绝对值不可与 A3a 比，比值只在臂内可比**。
5. **参数量与新头的一手账（修正一处表述）**：`n_new_head_keys` 实测 **scaled_structure 6｜scaled_structure_own 2｜amp_monotone_gate 2｜amp_hard_support 2｜topo_simvp 0**，`n_shared_keys` 全部 **66**；`model params` 三元组 **simvp_v2 (3,67,393) ＝ topo_simvp (3,67,393)**（拓扑臂**零参数开销**）｜**amp_monotone_gate ＝ amp_hard_support ＝ scaled_structure_own (3,67,458)**｜**scaled_structure (3,71,683)**。⇒ **「A4 用新标量头、A4b 不新增头」在模块层面成立**（A4b 模块数 67 与基线相同，A4 多 4 个模块），**但 A4b 仍新增 2 个参数键（与两个门臂同级）** ⇒ **「own ＝ 零新增」只在「不新增独立头模块」这一层成立，在参数量层面不成立** ⇒ 交接文档与 MASTER 里「A4b 不新增头」的措辞应按此口径读（**本轮不升 MASTER 版本**，只在本节登记）。
6. **算力预算（下一族的输入）**：每 epoch 训练秒数中位 **B0 108.4｜A1 110.1｜A2 110.6｜A4 108.8｜A4b 108.9｜A3a 167.5（1.55×）｜A3b 197.2（1.82×）** ⇒ 下一族若带拓扑项，训练算力按 **×1.6–1.8** 计。
7. **契约与来源的第三份独立证据**：21 份日志头（arm/model/loss/topo flags）与臂表 **21/21 吻合**；**S0 等价门在 21 个作业里各跑 5 个机制模型 = 105 次检查全部 PASS**（`max_abs_diff = 0.000e+00`、`bitwise=True`，JSON 里 `pass`/`audit_ok` 均 true），**不新增键的 topo_simvp 状态迁移干净**（missing/unexpected 均为 0）；21 格训练集规模同为 **25175 样本**；每格 3 个 checkpoint（epoch_00/01/02）；**stage2 日志指向的 checkpoint 时间戳与训练日志写入的 artifacts 目录 21/21 吻合** ⇒ 评测评的就是训练产出的那一格 ⇒ 「七臂各只比基线多一处改动」现在有**第三份独立证据**（此前：臂表本身、第 57 轮的远端 config 审计；本轮：作业 stdout 里的 S0 报告 + 参数量账）。

**守卫与口径**：① **S3 的 12 个区间仍全部跨零**，已发布的 `crosses_zero` 标志 **12/12 为 true**，六臂 `s3_verdict` 逐臂不变（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）⇒ **合格臂集合仍为空**；② 已发布头条表的七臂中位 CSI 由逐 seed 产物复算 **7/7 逐位一致（≤1e−9）**；③ 仍按「**中位数 ＋ 通过率**」报，单点分数不作裁决；④ **事后诊断，不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；⑤ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）。

**新登记两条可复用纪律**：① **训练损失不能替代技能曲线**——凡使用「epXX 是峰值端」这类说法的实验，必须同时给出被选 epoch 与其余 epoch 的技能对照，否则该规则**不可验证**；② **机制项必须报「训练期内部读数」**——`topo_pred_area ÷ topo_truth_area` 这类量在训练期就把过预报写在脸上，只在评测期看 BIAS 会把因果方向讲反。
"""

MEM_BULLET = (
    "- **第 58 轮新增两条轴（训练动力学轴 ＋ 契约/来源再验证）**：① **训练动力学轴（首次判读 21 份训练日志）**：训练损失在 "
    "**21/21 格严格单调下降**、训练段零 NaN/inf，而 **r(ep0 损失, CSI@2.0) = −0.0895（21 格）／−0.1138（18 臂）** ⇒ "
    "**训练损失对技能几乎零信息量**、**「last.pt 是崩塌端」在损失上完全看不见**（最刺眼：**A4b_s2 的 `CSI@2.0 = 0.015169` "
    "是全表最低，训练损失却仍单调下降且与对照同量级**）⇒ **训练损失不得用于选 epoch、也不得当训练健康度指标**；"
    "**「ep00 ＝ 峰值端」在 E45 内部不可验证**（每格只评过 ep00 一点）⇒ **新登记待办④：选 epoch 规则要么写进预注册、"
    "要么显式评 ep00/ep01/ep02**。② **训练期首次看到机制内部读数**：A3a／A3b 的 `topo_pred_area ÷ topo_truth_area` 中位 "
    "**1.56–2.02** ⇒ **过预报在训练期就已存在**，且**预注册的「面积归一化」（A3b）没有修掉面积失配** ⇒ 与 S3、第 51 轮 "
    "`ΔF/真值`、第 54 轮列联表域配平**四台仪器同向**。③ **拓扑两臂把损失抬高 9.9%／10.1%，却把中位技能压到低于对照** ⇒ "
    "与第 44／53 轮同向（从损失侧独立佐证）。④ **参数量一手账修正一处表述**：`n_new_head_keys` = scaled_structure **6**／"
    "scaled_structure_own **2**／两门臂各 **2**／topo_simvp **0**；`params` = topo_simvp 与基线**逐位相同**（零参数开销）、"
    "A4b 与两门臂**同价**（(3,67,458)）、A4 (3,71,683) ⇒ **「A4b 不新增头」只在模块层面成立，在参数量层面不成立**。"
    "⑤ **算力预算**：拓扑两臂每 epoch 训练时间是对照的 **1.55×／1.82×**。⑥ **契约/来源第三份独立证据**：日志头与臂表 "
    "21/21 吻合、S0 门 105 次检查全 PASS、训练集 25175 样本一致、stage2 checkpoint 时间戳与训练 artifacts 21/21 吻合。"
    "**裁决零变化、MASTER 不升版。**\n"
)

D14_BLOCK = """
### 2026-11-14｜第 58 轮（训练动力学轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十七轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮唯一远端动作是「只读取回训练日志」，未 sbatch、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；无 E46+ 目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。**本轮新增取回**：21 份训练日志 + 3 份 stage2 日志 + 1 份 summary 日志打包为 `/tmp/e45_train_logs.tar.gz`（md5 `bcafa93243bd5649671a76c7d853d9c3`），下载到 `tools/_evals/e45/e45_train_logs.tar.gz` 并解包到 `tools/_evals/e45/_train_logs/`（只读，不写远端训练目录）。
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。
- **新证据**——`tools/_e45_verify_round58.py`（只读本地，**POS 25/25 VERDICT: PASS**，产物 `tools/_evals/e45/round58_training_dynamics.json`）：前 57 轮判过十四层、**全部读的是评测产物**，**21 份训练日志从未被打开过** ⇒ 本轮首次在**训练动力学轴**判读，**裁决零变化**。① **训练损失在 21/21 格严格单调下降、零 NaN/inf，而与技能几乎零相关**（`r(ep0 损失, CSI@2.0)` = **−0.0895**（21 格）／**−0.1138**（18 臂））⇒ **「last.pt 是崩塌端」在损失上完全看不见**；**A4b_s2 的 `CSI@2.0 = 0.015169` 是全表最低，训练损失却仍单调下降且与对照同量级** ⇒ **训练损失不得用于选 epoch、也不得当健康度指标**；**「ep00 ＝ 峰值端」在 E45 内部不可验证** ⇒ **新登记待办④**。② **训练期首次看到机制内部读数**：A3a／A3b 的 `topo_pred_area ÷ topo_truth_area` 中位 **1.56–2.02** ⇒ **过预报在训练期就已存在**，且 **A3b 的「面积归一化」没有修掉面积失配** ⇒ 与 S3、第 51 轮 `ΔF/真值`、第 54 轮列联表域配平**四台仪器同向**。③ **拓扑两臂把损失抬高 9.9%／10.1%，却把中位技能压到低于对照**。④ **参数量一手账**：`n_new_head_keys` = scaled_structure **6**／own **2**／两门臂各 **2**／topo_simvp **0**；`params` = topo_simvp 与基线逐位相同、A4b 与门臂同价、A4 (3,71,683) ⇒ **「A4b 不新增头」只在模块层面成立**。⑤ **算力**：拓扑两臂每 epoch 训练时间是对照的 **1.55×／1.82×**。⑥ **契约/来源第三份独立证据**：日志头与臂表 21/21 吻合、**S0 门 105 次检查全 PASS**、训练集 25175 样本一致、stage2 checkpoint 时间戳与训练 artifacts 21/21 吻合。
- **重建自校**：已发布头条表七臂中位 CSI 由逐 seed 产物复算 **7/7 逐位一致（≤1e−9）**；**S3 的 12 个区间仍全部跨零**、`crosses_zero` 标志 12/12 为 true、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**。
- **裁决零变化**：六臂判负、不投十 seed 算力；**MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；③ **新登记纪律**：训练损失不能替代技能曲线（用「epXX 是峰值端」必须附其余 epoch 的技能对照）；机制项必须报训练期内部读数。
"""

D14_P_BLOCK = """
### 2026-11-14｜第 58 轮（训练动力学轴的首次判读）：状态复核（无远端改动，裁决不变）

- **远端**：273 `COMPLETED`（05:04:04）、275/283 各 7/7、274/282/290、291、297/298 全部 `COMPLETED`；21 份评测齐、13 份 json 齐全、队列空、无 E46+ 目录；远端↔本地 **md5 13/13 全同**。**本轮唯一远端动作**：把 21 份训练日志 + 3 份 stage2 + 1 份 summary 打包下载（md5 `bcafa93243bd5649671a76c7d853d9c3`）到 `tools/_evals/e45/`。**未 sbatch、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS`。
- **新证据**——`tools/_e45_verify_round58.py`（只读本地，**POS 25/25 VERDICT: PASS**，产物 `tools/_evals/e45/round58_training_dynamics.json`）：**21 份训练日志首次被判读** ⇒ **裁决零变化**。① 训练损失 **21/21 格单调下降**、零 NaN/inf，与技能几乎零相关（r = **−0.0895**／**−0.1138**）⇒ **训练损失不能当选 epoch 或健康度依据**（**A4b_s2 CSI = 0.015169 全表最低，损失却正常**）；**「ep00 ＝ 峰值端」不可验证** ⇒ 待办④。② **训练期机制内部读数**：`topo_pred_area ÷ topo_truth_area` 中位 **1.56–2.02** ⇒ **过预报在训练期就存在**，**A3b 的面积归一化没修掉它** ⇒ 四台仪器同向。③ 拓扑两臂损失 **+9.9%／+10.1%** 而中位技能低于对照。④ `n_new_head_keys` = A4 **6**／A4b **2**／门臂各 **2**／topo **0**；A4b 与门臂同价 ⇒ **「不新增头」只在模块层成立**。⑤ 拓扑臂训练时间 **1.55×／1.82×**。⑥ 日志头与臂表 21/21 吻合、**S0 门 105 次全 PASS**、训练集 25175 样本一致、stage2 checkpoint 与训练 artifacts **21/21 吻合**。
- **裁决零变化**：S3 十二区间仍全部跨零、`crosses_zero` 12/12 为 true、六臂 `s3_verdict` 不变、合格臂集合为空 ⇒ 六臂判负、不投十 seed 算力；**MASTER 不升版**。
"""

D11_LINE = ("- 2026-11-14 第 58 轮（训练动力学轴，只读本地，POS 25/25 VERDICT: PASS，"
            "产物 `tools/_evals/e45/round58_training_dynamics.json`）：**裁决不变**（六臂判负、"
            "不投十 seed 算力、MASTER v2.48 不升版）。**21 份训练日志首次被判读**：① 训练损失在 "
            "**21/21 格严格单调下降**、零 NaN/inf，而 `r(ep0 损失, CSI@2.0)` = **−0.0895**（21 格）／"
            "**−0.1138**（18 臂）⇒ **训练损失对技能几乎零信息量**，**「last.pt 是崩塌端」在损失上完全看不见**"
            "（最刺眼：**A4b_s2 的 `CSI@2.0 = 0.015169` 全表最低，训练损失却单调下降且与对照同量级**）"
            "⇒ **训练损失不得用于选 epoch、也不得当健康度指标**；**「ep00 ＝ 峰值端」在 E45 内部不可验证** "
            "⇒ **新登记待办④**。② **训练期首次看到机制内部读数**：A3a／A3b 的 `topo_pred_area ÷ topo_truth_area` "
            "中位 **1.56–2.02** ⇒ **过预报在训练期就已存在**，且 **A3b 的「面积归一化」没有修掉面积失配** ⇒ 与 S3、"
            "第 51 轮 `ΔF/真值`、第 54 轮列联表域配平**四台仪器同向**。③ **拓扑两臂把损失抬高 9.9%／10.1%，"
            "却把中位技能压到低于对照**。④ **参数量一手账修正一处表述**：`n_new_head_keys` = scaled_structure **6**／"
            "own **2**／两门臂各 **2**／topo_simvp **0**；`params` = topo_simvp 与基线逐位相同、A4b 与门臂同价、"
            "A4 (3,71,683) ⇒ **「A4b 不新增头」只在模块层面成立、在参数量层面不成立**。⑤ **算力**：拓扑两臂每 epoch "
            "训练时间是对照的 **1.55×／1.82×**。⑥ **契约/来源第三份独立证据**：日志头与臂表 21/21 吻合、**S0 门 105 次"
            "检查全 PASS**、训练集 25175 样本一致、stage2 checkpoint 时间戳与训练 artifacts 21/21 吻合。"
            "**未 sbatch、未改远端状态、未动 `docs/`。**\n")

AUTOM_BLOCK = """
### 2026-11-14｜第 58 轮（训练动力学轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十七轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮唯一远端动作是「只读取回训练日志」，未 sbatch、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；无 E46+ 目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。**本轮新增取回**：21 份训练日志 + 3 份 stage2 + 1 份 summary 打包下载（tar md5 `bcafa93243bd5649671a76c7d853d9c3`）。
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。
- **新证据**——`tools/_e45_verify_round58.py`（只读本地，**POS 25/25 VERDICT: PASS**，产物 `tools/_evals/e45/round58_training_dynamics.json`）：前 57 轮判过十四层、**全部读评测产物**，**21 份训练日志从未被打开** ⇒ 本轮首次在**训练动力学轴**判读，**裁决零变化**。① **训练损失 21/21 格单调下降、零 NaN/inf、与技能几乎零相关**（r = −0.0895／−0.1138）⇒ **训练损失不能当选 epoch 或健康度依据**；**「ep00 ＝ 峰值端」在 E45 内部不可验证** ⇒ 新登记待办④。② **训练期机制内部读数**：`topo_pred_area ÷ topo_truth_area` 中位 **1.56–2.02** ⇒ **过预报在训练期就存在**，**A3b 的面积归一化没修掉它**。③ 拓扑两臂损失 +9.9%／+10.1% 而中位技能低于对照。④ `n_new_head_keys` = A4 6／A4b 2／门臂各 2／topo 0 ⇒ **「A4b 不新增头」只在模块层成立**。⑤ 拓扑臂训练时间 1.55×／1.82×。⑥ **契约/来源第三份独立证据**：日志头与臂表 21/21 吻合、S0 门 105 次全 PASS、训练集 25175 样本一致、stage2 checkpoint 与训练 artifacts 21/21 吻合。
- **重建自校**：头条表七臂中位 CSI 复算 7/7 逐位一致（≤1e−9）；**S3 的 12 个区间仍全部跨零**、`crosses_zero` 12/12 为 true、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**。
- **裁决零变化**：六臂判负、不投十 seed 算力；**MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；③ **新登记纪律**：训练损失不能替代技能曲线；机制项必须报训练期内部读数。
"""


def read(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def insert_before(path, anchor, block, marker, label):
    try:
        txt = read(path)
    except Exception as exc:
        ERRORS.append("%s: cannot read (%s)" % (label, exc))
        return False
    if marker in txt:
        REPORT.write("  unchanged  %s (marker already present)\n" % label)
        return False
    if anchor not in txt:
        ERRORS.append("%s: anchor not found: %r" % (label, anchor[:60]))
        return False
    write(path, txt.replace(anchor, block + anchor, 1))
    REPORT.write("  inserted   %s\n" % label)
    return True


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
    new, n = re.subn(pattern, repl.replace("\\", "\\\\"), txt, count=1)
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
    for anchor in ("- **三条定位**",
                   "- **第 57 轮新增两条轴（聚合算子轴 ＋ 评测契约/来源审计）**",
                   "- **第 8–"):
        if anchor in txt:
            write(path, txt.replace(anchor, bullet + anchor, 1))
            REPORT.write("  inserted   %s (anchor %r)\n" % (label, anchor[:28]))
            return True
    ERRORS.append("%s: no usable anchor for the bullet" % label)
    return False


def main():
    changes = 0

    # 1) HANDOVER_E45.md -- new section before the archive heading
    changes += insert_before(
        HANDOVER,
        "\n### 下面是结案前的执行记录（存档用，不必重跑）",
        HANDOVER_BLOCK,
        "**\u3252（2026-11-14 第 58 轮（训练动力学轴的首次判读））**",
        "HANDOVER_E45.md")

    # 2) upper MEMORY.md status section
    changes += re_sub_once(
        MEMORY,
        r"- \*\*第 8\u2013\d+ 轮独立复核结论完全一致、无新增偏离\*\*\uff1a",
        "- **第 8\u201358 轮独立复核结论完全一致、无新增偏离**：",
        "MEMORY.md (round range)")
    changes += re_sub_once(
        MEMORY,
        r"第 40\u2013\d+ 轮新增的均为\*\*事后诊断新轴\*\*",
        "第 40\u201358 轮新增的均为**事后诊断新轴**",
        "MEMORY.md (new-axis range)")
    changes += re_sub_once(
        MEMORY,
        r"第 8\u2013\d+ 轮复核结论一致",
        "第 8\u201358 轮复核结论一致",
        "MEMORY.md (version-line round count)")
    changes += re_sub_once(
        MEMORY,
        r"（逐 seed 分解、maxratio 判读、跨实验复现、派生布尔分支、多阈值口径、S1 结构量、窗级／场次级／时效／强度·难度／误差构成／臂间合并／剂量-反应／指标族／护栏标定）",
        "（逐 seed 分解、maxratio 判据判读、跨实验复现、派生布尔分支、多阈值口径、S1 结构量、窗级／场次级／时效／强度·难度／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样稳定性／聚合算子与评测契约／训练动力学）",
        "MEMORY.md (axis list)")
    changes += re_sub_once(
        MEMORY,
        r"详见 `HANDOVER_E45\.md` §4\u2464[^\s]{0,14} 与逐日日志。",
        "详见 `HANDOVER_E45.md` §4⑤–㉒ 与逐日日志。",
        "MEMORY.md (handover pointer)")
    changes += insert_bullet(MEMORY, MEM_BULLET, "MEMORY.md (round-58 bullet)")

    # 3) daily logs
    changes += append_once(D14_U, D14_BLOCK,
                           "### 2026-11-14｜第 58 轮（训练动力学轴的首次判读）",
                           "2026-11-14.md (upper)")
    changes += append_once(D14_P, D14_P_BLOCK,
                           "### 2026-11-14｜第 58 轮（训练动力学轴的首次判读）",
                           "2026-11-14.md (project)")
    changes += append_once(D11_U, D11_LINE,
                           "- 2026-11-14 第 58 轮（训练动力学轴，只读本地，POS 25/25 VERDICT: PASS，",
                           "2026-11-11.md (upper)")
    changes += append_once(AUTOM_F, AUTOM_BLOCK,
                           "### 2026-11-14｜第 58 轮（训练动力学轴的首次判读）",
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
