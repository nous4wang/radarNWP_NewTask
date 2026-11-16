#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 57 -- writeback (idempotent).

Idempotency rule (established in earlier rounds): the marker is the exact
text being inserted; if the marker is already present in the target file,
the file is left untouched ("unchanged").  A missing anchor is an ERROR --
never a silent skip.

This round produces NO new criterion and NO verdict change, so MASTER is
NOT bumped (stays v2.48) and docs/ is NOT touched.  New evidence goes to
HANDOVER_E45.md, the upper MEMORY.md status section, and the daily logs.
"""
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPPER = r"R:\1radarNWPprediction\.workbuddy\memory"
PROJ = os.path.join(REPO, ".workbuddy", "memory")
AUTOM = os.path.join(PROJ, "automations",
                     "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")

REPORT = io.StringIO()

HANDOVER = os.path.join(UPPER, "HANDOVER_E45.md")
MEMORY = os.path.join(UPPER, "MEMORY.md")
D14_U = os.path.join(UPPER, "2026-11-14.md")
D14_P = os.path.join(PROJ, "2026-11-14.md")
D11_U = os.path.join(UPPER, "2026-11-11.md")
AUTOM_F = AUTOM

# ----------------------------------------------------------------- texts
HANDOVER_BLOCK = """
**㉑（2026-11-14 第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读））**——脚本 `tools/_e45_verify_round57.py`（只读本地，**POS 22/22 VERDICT: PASS**，产物 `tools/_evals/e45/round57_operator_sensitivity.json`）：前 56 轮判过聚合层／机制探针／S1 结构量／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定／增益图样稳定性共十三层，**从未问过两件事**：① **发布头条表是用哪一个"聚合算子"算出来的**（默认恒为「合并 538 窗的列联表 → CSI → 3 seed 取中位数」＝`pooled`；但这个算子只是一个建模选择，不是定律）；② **21 格到底在什么"评测契约"下跑的**（前 56 轮的每一处配对比较都默认这 21 格可配对，却从未审计过）。本轮第一次同时判这两条轴，**裁决零变化**，得六条新事实：

1. **聚合算子轴（5 个算子：`pooled`／`macro_mean`／`macro_median`／`trimmed10`／`case_equal`）——定性裁决算子稳健，定量幅度不算子稳健**。**符号图样在 5 个算子下完全一致（`--++++`，即 A1／A2 负、A3a／A3b／A4／A4b 正），distinct patterns = 1** ⇒ 「哪一臂比对照好、哪一臂更差」不随算子改变。**但零模型地板随算子变化 5.5 倍**（`pooled` **0.038771** → `case_equal` **0.007008**）⇒ **「增益是否超过种子噪声」这个判断是算子依赖的**：`pooled` 下六臂的效应/地板比**无一超过 1**（最大 A3b +0.63），而 `case_equal` 下 **A4 +2.07、A3b +1.83、A3a +1.21、A4b +1.08 均超过 1**。⇒ **已发布的 `pooled` 算子是这五个里对「过噪声」最保守的一个**（地板最大）⇒ **头条表没有偏袒任何一臂；换一个更宽松的算子只会让各臂"看起来更好"**。**臂间排序对算子不稳**（与 `pooled` 的 Spearman：macro_mean 0.857／macro_median 0.750／trimmed10 0.750／**case_equal 0.500**）⇒ 与第 52 轮「臂效应只占方差 8.1%」同向。
2. **「增益＝面积/幅度效应」这条裁决在 5 个算子下全部存活**（每窗按第 54 轮口径做面积配平后与 S3 场域等偏置点估比符号）：**pooled 6/6、macro_mean 6/6、trimmed10 6/6、macro_median 5/6、case_equal 5/6**（两处不一致都在 A1／A2 的接近零侧）⇒ **连"放宽算子"这条最有利于各臂的路都救不回任何一个臂**。**面积配平不移动真值**（`T=h+m` 逐窗最大偏差 **0.000e+00**）。
3. **`case_equal` 之所以让各臂"过噪声"，机制是场次集中度**（逐场次中位 ΔCSI，臂 vs 对照同 seed）：**A3a／A3b 各 7/8 场次为正**（均值 +0.01053／+0.01330，顶部驱动场次 `CP_01_09282353` +0.0271／+0.0349，占 |和| 的 30–31%）；**A4 仅 5/8 为正**（顶部驱动 +0.04595、占 33%）；A4b 仅 4/8 为正；**A1 的顶部驱动是负的**（`CP_01_08180913` −0.05007、占 56%）。对照臂换种子的场次级零模型：**均值口径地板 0.00701、逐场次最大地板 0.05163** ⇒ **A4 的顶部驱动值 +0.04595 已逼近逐场次地板** ⇒ 「A4 在等场次权重下过噪声」仍与「增益集中在个别场次 + 过预报」同向，**不构成机制成立**。
4. **跨轮口径纪律（本轮踩到并自校）**：`case_equal`（**场次 CSI 的均值**）与第 47 轮的场次级统计量（**逐场次 ΔCSI 的中位数**）**不是同一个统计量**，两者的零模型地板（**0.00701 vs 0.01077**）**不可互相比较** ⇒ 本轮按第 47 轮口径重算一遍作对照：**地板 0.01077，六臂全部落在地板内**（A1 +0.00340(3/3)、A2 −0.00818(0/3)、A3a +0.00291(3/3)、A3b +0.00966(3/3)、A4 +0.00986(2/3)、A4b +0.00778(2/3)）⇒ **与第 47 轮结论逐项一致**。⇒ **新登记纪律：凡报「超过零模型地板」必须同时报该算子下的地板；跨轮比较统计量必须对齐口径**（与第 52 轮「零模型地板按量纲分别算」、第 54 轮「地板按指标族分别算」同族）。
5. **评测契约/来源审计（值级证据，前 56 轮从未做过）**：**21 格全部 `ablation.mode='full'`、输入变量集完全相同（27 个 NWP 变量 + 10 个雷达变量）、零丢帧（`missing_frame_steps` 全空）、每格 538 样本/538 窗、`window_keys` 跨 21 格完全一致且同序** ⇒ 第 46–56 轮一切配对比较的**前提条件被独立验证**。**checkpoint 来源与臂表逐格吻合 21/21**（`runs/e45/<ARM>_s<SEED>/train/<model>_<stamp>/epoch_00.pt`），**且三个 seed 是同步块**（s0 启动戳 1794359074、s1 1794365104、s2 1794371105/06，7 臂同戳）⇒ **seed 是共同区组**（与第 52 轮 seed 占方差 38.7% 一致）。
6. **远端训练 config 的值级审计（唯一能证伪"旗标没生效"的一手证据）**：**只有取值能区分，因为每份 config 都带着全部旗标及其默认值**。实测：**A3a 3/3 seed 的 `topo_clamp_mm=20.0` 且另两个拓扑旗标为默认**；**A3b 3/3 seed 的 `topo_truth_smooth_sigma=1.0` + `topo_area_normalize=True` 且 clamp 为 0**；**其余五臂三个拓扑旗标全部处于默认关闭**；**`model` 字段 21/21 与臂表一致**；**`loss` 字段与臂表一致**（五臂 `balanced`、A3a／A3b `topo`，且 `topo` 的 base 就是同一个 `_BalancedExtremeCriterion(alpha_csi=0.5)`，见 `scripts/train.py:449-452`）⇒ **「七臂各只比基线多一处改动」首次被一手配置证据证实**；**`alpha_csi=0.5` 21/21、NWP 21/21 存在（`no_nwp`/`zero_nwp` 全 false）、21/21 同一 `data_root`、每格恰好一份训练 config**。⇒ 这条审计**独立于全部分数**，不依赖任何判据。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；⑤ **重建自校**：`pooled` 算子复算已发布 `CSI@2.0` **21/21 max|diff| = 0.000e+00**、已发布配对增益符号复现（4 正 2 负）、面积配平不动真值（0.000e+00）；⑥ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
"""

MEM_OLD1 = "- **第 8–56 轮独立复核结论完全一致、无新增偏离**："
MEM_OLD2 = "第 40–56 轮新增的均为**事后诊断新轴**"
MEM_OLD3 = "详见 `HANDOVER_E45.md` §4⑤–⑳ 与逐日日志。"
MEM_NEW1 = "- **第 8–57 轮独立复核结论完全一致、无新增偏离**："
MEM_NEW2 = "第 40–57 轮新增的均为**事后诊断新轴**"
MEM_NEW3 = "详见 `HANDOVER_E45.md` §4⑤–㉑ 与逐日日志。"
MEM_BULLET = (
    "- **第 57 轮新增两条轴（聚合算子轴 ＋ 评测契约/来源审计）**：① **聚合算子轴**——发布头条表恒用 `pooled` 算子（合并 538 窗列联表 → CSI → 3 seed 中位数），本轮首次用 5 个算子（`pooled`／`macro_mean`／`macro_median`／`trimmed10`／`case_equal`）重算：**配对增益的符号图样在 5 个算子下完全一致（`--++++`）⇒ 定性裁决算子稳健**；**但零模型地板随算子变化 5.5 倍（0.038771 → 0.007008）⇒ 「是否超过种子噪声」是算子依赖的**（`pooled` 下六臂无一过噪声，`case_equal` 下 A4/A3b/A3a/A4b 过），**而已发布的 `pooled` 是其中最保守的一个 ⇒ 头条表未偏袒任何一臂**；**「增益＝面积/幅度效应」在 5 个算子下全部存活**（与 S3 符号一致 6/6、6/6、6/6、5/6、5/6）⇒ 放宽算子也救不回任何一臂；② **评测契约/来源审计**（一手证据）：21 格 `mode='full'`、输入变量集相同（27+10）、零丢帧、538 窗、`window_keys` 跨 21 格同序 ⇒ 此前所有配对比较的前提被独立验证；**远端训练 config 值级审计证实「七臂各只比基线多一处改动」**（A3a 仅 `topo_clamp_mm=20.0`、A3b 仅 `smooth=1.0`+`area_normalize=True`、其余五臂全默认；`model`／`loss`／`alpha_csi=0.5`／NWP／`data_root` 21/21 一致）。**裁决零变化、MASTER 不升版。**\n"
)

D14_BLOCK = """
### 2026-11-14｜第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十六轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未改远端状态、未动 `docs/`，MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；无 E46+ 目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。
- **新证据**——`tools/_e45_verify_round57.py`（只读本地，**POS 22/22 VERDICT: PASS**，产物 `tools/_evals/e45/round57_operator_sensitivity.json`）：前 56 轮判过十三层，**从未问过「发布头条表是用哪个聚合算子算出来的」与「21 格到底在什么评测契约下跑的」** ⇒ 本轮首次同时判这两条轴，**裁决零变化**。① **聚合算子轴**：5 个算子下**配对增益符号图样完全一致（`--++++`，distinct = 1）⇒ 定性裁决算子稳健**；**但零模型地板随算子变 5.5 倍（0.038771 → 0.007008）⇒ 「是否超过种子噪声」是算子依赖的**（`pooled` 下六臂效应/地板比无一 >1，最大 A3b +0.63；`case_equal` 下 A4 +2.07、A3b +1.83、A3a +1.21、A4b +1.08）⇒ **已发布的 `pooled` 是最保守的算子 ⇒ 头条表未偏袒任何一臂**；臂间排序不稳（与 `pooled` 的 Spearman 最低 0.500）。② **「增益＝面积/幅度效应」在 5 个算子下全部存活**（与 S3 等偏置点估符号一致 6/6、6/6、6/6、5/6、5/6），面积配平不动真值（0.000e+00）⇒ 放宽算子也救不回任何一臂。③ **`case_equal` 的机制＝场次集中度**：A3a／A3b 各 7/8 场次为正（顶部驱动 `CP_01_09282353` 占 |和| 30–31%）、A4 仅 5/8（顶部驱动 +0.04595，逼近对照换种子的逐场次地板 0.05163）⇒ 「过噪声」仍与「集中于个别场次＋过预报」同向，**不构成机制成立**。④ **跨轮口径自校**：`case_equal`（场次 CSI 均值）≠ 第 47 轮（逐场次 ΔCSI 中位数），地板 0.00701 vs 0.01077 **不可比**；按第 47 轮口径重算 ⇒ 地板 0.01077、**六臂全部落在地板内**、与第 47 轮逐项一致 ⇒ **新登记纪律：报「超过零模型地板」必须同时报该算子的地板；跨轮比较统计量必须对齐口径**。⑤ **评测契约审计**：21 格 `mode='full'`、输入变量集相同（27 NWP + 10 雷达）、零丢帧、538/538 窗、`window_keys` 跨 21 格完全同序 ⇒ 前 56 轮一切配对比较的前提被独立验证；**checkpoint 来源与臂表 21/21 吻合**，**三个 seed 是同步块**（7 臂同启动戳）⇒ seed 是共同区组。⑥ **远端训练 config 值级审计**（一手证据）：**只有取值能区分**，因为每份 config 都带全部旗标及默认值；实测 **A3a 仅 `topo_clamp_mm=20.0`、A3b 仅 `smooth=1.0`+`area_normalize=True`、其余五臂三个拓扑旗标全默认**；**`model`／`loss`（`topo` 的 base 就是同一个 `_BalancedExtremeCriterion(alpha_csi=0.5)`，`scripts/train.py:449-452`）／`alpha_csi=0.5`／NWP／`data_root` 21/21 一致，每格恰好一份 config** ⇒ **「七臂各只比基线多一处改动」首次被一手配置证据证实**。
- **重建自校**：`pooled` 算子复算已发布 `CSI@2.0` **21/21 max|diff| = 0.000e+00**；已发布配对增益符号复现（4 正 2 负）；S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 不变、**合格臂集合仍为空**。
- **裁决零变化**：六臂判负、不投十 seed 算力；**MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**。
"""

D14_P_BLOCK = """
### 2026-11-14｜第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读）：状态复核（无远端改动，裁决不变）

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；21 份评测齐、13 份 json 齐全、队列空、无 E46+ 目录；远端↔本地 **md5 13/13 全同**。**未下载、未 sbatch、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS`。
- **新证据**——`tools/_e45_verify_round57.py`（只读本地，**POS 22/22 VERDICT: PASS**，产物 `tools/_evals/e45/round57_operator_sensitivity.json`）：首次同时判**聚合算子轴**与**评测契约/来源审计**两条轴 ⇒ **裁决零变化**。① **算子轴**：5 个算子下配对增益**符号图样完全一致（`--++++`）⇒ 定性裁决算子稳健**；**零模型地板随算子变 5.5 倍（0.038771 → 0.007008）⇒ 「是否超过种子噪声」是算子依赖的**（`pooled` 下六臂无一过噪声；`case_equal` 下 A4 +2.07、A3b +1.83、A3a +1.21、A4b +1.08）⇒ **已发布的 `pooled` 是最保守的算子 ⇒ 头条表未偏袒任何一臂**；臂间排序对算子不稳（Spearman 最低 0.500）。② **「增益＝面积/幅度效应」在 5 个算子下全部存活**（与 S3 符号一致 6/6、6/6、6/6、5/6、5/6）⇒ 放宽算子也救不回任何一臂。③ **`case_equal` 的机制＝场次集中度**（A3a／A3b 7/8 场次为正、顶部驱动 `CP_01_09282353`；A4 仅 5/8，顶部驱动 +0.04595 逼近对照逐场次地板 0.05163）⇒ **不构成机制成立**。④ **跨轮口径自校**：`case_equal` ≠ 第 47 轮口径（地板 0.00701 vs 0.01077 不可比）；按第 47 轮口径重算 ⇒ **六臂全部落在地板 0.01077 内**、与第 47 轮逐项一致 ⇒ **新纪律：报「超过零模型地板」必须同时报该算子的地板；跨轮比较统计量必须对齐口径**。⑤ **契约审计**：21 格 `mode='full'`、输入变量集相同（27+10）、零丢帧、538/538 窗、`window_keys` 跨 21 格同序；**checkpoint 与臂表 21/21 吻合**、三个 seed 为同步块。⑥ **远端训练 config 值级审计**（一手证据）：**A3a 仅 `topo_clamp_mm=20.0`、A3b 仅 `smooth=1.0`+`area_normalize=True`、其余五臂全默认**；`model`／`loss`／`alpha_csi=0.5`／NWP／`data_root` 21/21 一致 ⇒ **「七臂各只比基线多一处改动」首次被一手配置证据证实**。
- **裁决零变化**：12/12 区间全跨零、六臂 `s3_verdict` 不变、合格臂集合为空 ⇒ 六臂判负、不投十 seed 算力；**MASTER 不升版**。
"""

D11_LINE = ("- 2026-11-14 第 57 轮（聚合算子轴 ＋ 评测契约/来源审计，只读本地，POS 22/22 VERDICT: PASS，"
            "产物 `tools/_evals/e45/round57_operator_sensitivity.json`）：**裁决不变**（六臂判负、"
            "不投十 seed 算力、MASTER v2.48 不升版）。① **聚合算子轴**（首次判读）：发布头条表恒用 "
            "`pooled` 算子（合并 538 窗列联表 → CSI → 3 seed 中位数），本轮用 5 个算子重算 ⇒ "
            "**配对增益符号图样完全一致（`--++++`，distinct = 1）⇒ 定性裁决算子稳健**；"
            "**零模型地板随算子变 5.5 倍（0.038771 → 0.007008）⇒ 「是否超过种子噪声」是算子依赖的**"
            "（`pooled` 下六臂效应/地板比无一 >1，最大 A3b +0.63；`case_equal` 下 A4 +2.07、"
            "A3b +1.83、A3a +1.21、A4b +1.08）⇒ **已发布的 `pooled` 是最保守的算子、头条表未偏袒任何一臂**；"
            "**「增益＝面积/幅度效应」在 5 个算子下全部存活**（与 S3 等偏置点估符号一致 6/6、6/6、6/6、5/6、5/6；"
            "面积配平不动真值 0.000e+00）⇒ 放宽算子也救不回任何一臂；`case_equal` 让各臂「过噪声」的机制＝"
            "场次集中度（A3a／A3b 7/8 场次为正、顶部驱动 `CP_01_09282353`；A4 仅 5/8，顶部驱动 +0.04595 "
            "逼近对照换种子的逐场次地板 0.05163）⇒ 不构成机制成立。② **跨轮口径自校**：`case_equal`"
            "（场次 CSI 均值）≠ 第 47 轮（逐场次 ΔCSI 中位数），地板 0.00701 vs 0.01077 不可比；"
            "按第 47 轮口径重算 ⇒ 六臂全部落在地板 0.01077 内、与第 47 轮逐项一致 ⇒ **新登记纪律："
            "报「超过零模型地板」必须同时报该算子的地板；跨轮比较统计量必须对齐口径**。"
            "③ **评测契约/来源审计（值级一手证据）**：21 格 `mode='full'`、输入变量集相同（27 NWP + 10 雷达）、"
            "零丢帧、538/538 窗、`window_keys` 跨 21 格完全同序 ⇒ 前 56 轮一切配对比较的前提被独立验证；"
            "**checkpoint 来源与臂表 21/21 吻合**、三个 seed 为同步块（7 臂同启动戳）⇒ seed 是共同区组；"
            "**远端训练 config 值级审计证实「七臂各只比基线多一处改动」**（A3a 仅 `topo_clamp_mm=20.0`、"
            "A3b 仅 `smooth=1.0`+`area_normalize=True`、其余五臂三个拓扑旗标全默认；`model`／`loss`／"
            "`alpha_csi=0.5`／NWP／`data_root` 21/21 一致；`topo` 的 base 就是同一个 "
            "`_BalancedExtremeCriterion(alpha_csi=0.5)`，`scripts/train.py:449-452`）。"
            "④ 重建自校：`pooled` 复算已发布 `CSI@2.0` 21/21 max|diff| = 0.000e+00；S3 的 12 个区间仍全部跨零、"
            "六臂 `s3_verdict` 不变、合格臂集合仍为空。**未 sbatch、未改远端状态、未动 `docs/`。**\n")

AUTOM_BLOCK = """
### 2026-11-14｜第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十六轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未改远端状态、未动 `docs/`，MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；无 E46+ 目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。
- **新证据**——`tools/_e45_verify_round57.py`（只读本地，**POS 22/22 VERDICT: PASS**，产物 `tools/_evals/e45/round57_operator_sensitivity.json`）：前 56 轮判过十三层，**从未问过「发布头条表是用哪个聚合算子算出来的」与「21 格到底在什么评测契约下跑的」** ⇒ 本轮首次同时判这两条轴，**裁决零变化**。① **聚合算子轴**（5 算子）：**配对增益符号图样完全一致（`--++++`，distinct = 1）⇒ 定性裁决算子稳健**；**零模型地板随算子变 5.5 倍（0.038771 → 0.007008）⇒ 「是否超过种子噪声」是算子依赖的**（`pooled` 下六臂效应/地板比无一 >1，最大 A3b +0.63；`case_equal` 下 A4 +2.07、A3b +1.83、A3a +1.21、A4b +1.08）⇒ **已发布的 `pooled` 是最保守的算子 ⇒ 头条表未偏袒任何一臂**；臂间排序对算子不稳（与 `pooled` 的 Spearman 最低 0.500）。② **「增益＝面积/幅度效应」在 5 个算子下全部存活**（与 S3 等偏置点估符号一致 6/6、6/6、6/6、5/6、5/6），面积配平不动真值（0.000e+00）⇒ 放宽算子也救不回任何一臂。③ **`case_equal` 的机制＝场次集中度**：A3a／A3b 各 7/8 场次为正（顶部驱动 `CP_01_09282353` 占 |和| 30–31%）、A4 仅 5/8（顶部驱动 +0.04595，逼近对照换种子的逐场次地板 0.05163）⇒ 「过噪声」仍与「集中于个别场次＋过预报」同向，**不构成机制成立**。④ **跨轮口径自校**：`case_equal`（场次 CSI 均值）≠ 第 47 轮（逐场次 ΔCSI 中位数），地板 0.00701 vs 0.01077 **不可比**；按第 47 轮口径重算 ⇒ 地板 0.01077、**六臂全部落在地板内**、与第 47 轮逐项一致 ⇒ **新登记纪律：报「超过零模型地板」必须同时报该算子的地板；跨轮比较统计量必须对齐口径**。⑤ **评测契约审计**：21 格 `mode='full'`、输入变量集相同（27 NWP + 10 雷达）、零丢帧、538/538 窗、`window_keys` 跨 21 格完全同序 ⇒ 前 56 轮一切配对比较的前提被独立验证；**checkpoint 来源与臂表 21/21 吻合**、**三个 seed 是同步块**（7 臂同启动戳）⇒ seed 是共同区组。⑥ **远端训练 config 值级审计**（一手证据，**只有取值能区分，因为每份 config 都带全部旗标及默认值**）：**A3a 仅 `topo_clamp_mm=20.0`、A3b 仅 `smooth=1.0`+`area_normalize=True`、其余五臂三个拓扑旗标全默认**；**`model`／`loss`（`topo` 的 base 就是同一个 `_BalancedExtremeCriterion(alpha_csi=0.5)`，`scripts/train.py:449-452`）／`alpha_csi=0.5`／NWP／`data_root` 21/21 一致，每格恰好一份 config** ⇒ **「七臂各只比基线多一处改动」首次被一手配置证据证实**。
- **重建自校**：`pooled` 算子复算已发布 `CSI@2.0` **21/21 max|diff| = 0.000e+00**；已发布配对增益符号复现（4 正 2 负）；S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 不变、**合格臂集合仍为空**。
- **裁决零变化**：六臂判负、不投十 seed 算力；**MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**。
"""


def read(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def insert_before(path, anchor, block, marker):
    """Insert block before anchor; if marker already present -> unchanged."""
    txt = read(path)
    if marker in txt:
        REPORT.write("  unchanged  %s (marker already present)\n" % os.path.basename(path))
        return False
    if anchor not in txt:
        raise SystemExit("ERROR: anchor not found in %s: %r" % (path, anchor[:60]))
    txt = txt.replace(anchor, block + anchor, 1)
    write(path, txt)
    REPORT.write("  inserted   %s\n" % os.path.basename(path))
    return True


def append_once(path, block, marker):
    txt = read(path)
    if marker in txt:
        REPORT.write("  unchanged  %s (marker already present)\n" % os.path.basename(path))
        return False
    write(path, txt.rstrip("\n") + "\n" + block)
    REPORT.write("  inserted   %s\n" % os.path.basename(path))
    return True


def replace_once(path, old, new):
    txt = read(path)
    if new in txt and old not in txt:
        REPORT.write("  unchanged  %s (already updated)\n" % os.path.basename(path))
        return False
    if old not in txt:
        raise SystemExit("ERROR: old text not found in %s: %r" % (path, old[:60]))
    write(path, txt.replace(old, new, 1))
    REPORT.write("  updated    %s\n" % os.path.basename(path))
    return True


def main():
    changes = 0

    # 1) HANDOVER_E45.md -- append section 20 before the archive heading
    changes += insert_before(
        HANDOVER,
        "\n### 下面是结案前的执行记录（存档用，不必重跑）",
        HANDOVER_BLOCK,
        "**㉑（2026-11-14 第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读））**")

    # 2) upper MEMORY.md status section
    changes += replace_once(MEMORY, MEM_OLD1, MEM_NEW1)
    changes += replace_once(MEMORY, MEM_OLD2, MEM_NEW2)
    changes += replace_once(MEMORY, MEM_OLD3, MEM_NEW3)
    txt = read(MEMORY)
    if MEM_BULLET not in txt:
        anchor = "- **三条定位**"
        if anchor not in txt:
            raise SystemExit("ERROR: anchor not found in MEMORY.md: %r" % anchor)
        write(MEMORY, txt.replace(anchor, MEM_BULLET + anchor, 1))
        REPORT.write("  inserted   MEMORY.md (round-57 bullet)\n")
        changes += 1
    else:
        REPORT.write("  unchanged  MEMORY.md (round-57 bullet already present)\n")

    # 3) daily logs
    changes += append_once(D14_U, D14_BLOCK, "### 2026-11-14｜第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读）")
    changes += append_once(D14_P, D14_P_BLOCK, "### 2026-11-14｜第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读）")
    changes += append_once(D11_U, D11_LINE, "- 2026-11-14 第 57 轮（聚合算子轴 ＋ 评测契约/来源审计，只读本地，POS 22/22 VERDICT: PASS，")
    changes += append_once(AUTOM_F, AUTOM_BLOCK, "### 2026-11-14｜第 57 轮（聚合算子轴 ＋ 评测契约/来源审计的首次判读）")

    print(REPORT.getvalue())
    print("writeback: %d file(s) changed" % changes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
