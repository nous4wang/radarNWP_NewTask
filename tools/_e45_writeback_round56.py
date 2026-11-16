"""E45 round-56 write-back (idempotent, marker-guarded).

Round 56 = new axis: STABILITY OF THE PER-WINDOW GAIN PATTERN ACROSS SEEDS.

Why this axis: rounds 40-55 judged E45's gains along aggregate CSI/BIAS,
mechanism read-outs, cross-experiment reproducibility, derived booleans,
multi-threshold calibers, S1 struct metrics, window level, storm-case level,
LOSO/ICC/MDE, lead time, severity/difficulty strata, error composition, the
omnibus arm effect, mechanism dose-response, metric families and the guard's
own calibration.  NONE asked whether the SET OF WINDOWS in which an arm beats
the control is the same set across the arm's three seeds -- i.e. whether the
arm has a reproducible skill signature at all, or merely a per-seed accident.

Round-56 findings (script tools/_e45_verify_round56.py, POS 6/6, VERDICT: PASS):
  * Control reference: the control arm's own cross-seed per-window CSI
    correlation is +0.8706 (median over 3 seed pairs) => the statistic IS
    sensitive enough to detect a reproducible pattern if one existed, so the
    arms' failures below are real, not a power artefact.
  * Axis A (cross-seed reproducibility of the per-window gain vector):
    A4 is the ONLY arm with a positive, seed-consistent pattern
    (median r = +0.2908, partial r = +0.4699, r>floor 3/3, sign agreement
    0.645, three-seed agreement 0.488 / all-positive 0.387).
    A3a/A3b are NEGATIVE and beyond the shuffle floor (median r = -0.3934 /
    -0.3839, |r|>floor 3/3, negative in 2/3 pairs) => their gain PATTERN
    FLIPS wholesale between seeds.  A1/A2 show no pattern (|r|>floor 1/3).
    A4b collapses under the partial correlation (+0.0757 -> -0.2545).
  * Axis B (arm-to-arm co-structure): same-seed median r = +0.4080 vs
    cross-seed -0.1167 (excess +0.5247) => arms co-move at the same seed,
    but same-family excess (+0.4850) vs cross-family (+0.2599) gap only
    +0.2251 => mostly a SEED effect, not a mechanism-family effect.
  * Axis C (truth-weighted seed-shared share of the window-level gain, with a
    sign-flip permutation null): NO arm exceeds the null p95
    (A1 0.0821/0.2876, A2 0.1670/0.2354, A3a 0.1332/0.2886,
     A3b 0.1437/0.3160, A4 0.2749/0.2749, A4b 0.1571/0.2876)
    => no arm has a statistically significant reproducible gain component;
    A4 is the closest (pseudo-p 0.2550).
  * Axis E (composition of the seed-consistent gain): A4's consistent-gain
    windows (n=207) carry dPOD +0.1088, dFAR +0.0815, dBIASratio +0.7926
    => the reproducible part is bought by over-forecast, matching S3's
    GAIN_WAS_AREA_OR_AMPLITUDE; A3a/A3b consistent-gain windows also carry
    dBIASratio +0.3340/+0.4161.
  * Verdict ZERO change: S3 12/12 intervals still cross zero, six arm
    verdicts unchanged, qualified-arm set empty, MASTER stays v2.48.
    Post-hoc diagnosis, no new criterion (S3 guard fixed at thr=2.0 mm,
    replication unit = seed).

Collateral repair in the SAME files (pre-existing, NOT caused by round 56):
the upper MEMORY.md was re-compressed at 2026-11-14 08:23 (after round 55's
07:40 write-back), and the rewrite dropped citations that the repo's own
gates require -> tools/_check_v246_row.py and tools/_check_v248_row.py went
FAIL on "MEMORY.md's status section carries v2.46" and on the missing
"3.0188" / "24.7" citations.  Round 56 restores them (the facts come from
HANDOVER section 4 round 43 / MASTER v2.48) and re-runs the gates.

Idempotency rules (rounds 41/43/45/47/48/49/51/52/54/55):
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged";
  * never slice a file from "my marker" to "the next heading";
  * this write-back is never invoked from a verification script.
"""

from pathlib import Path

MARK = "第 56 轮（增益图样跨种子稳定性轴的首次判读）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1114 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
PROJ_LOG_1114 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

# --------------------------------------------------------------------- blocks
HANDOVER_BLOCK = """
**⑳（2026-11-14 第 56 轮（增益图样跨种子稳定性轴的首次判读））**——脚本 `tools/_e45_verify_round56.py`（只读本地，**POS 6/6 VERDICT: PASS**，产物 `tools/_evals/e45/round56_gain_pattern_stability.json`）：前 55 轮判过聚合层／机制探针层／S1 结构量层／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定，**从未问过「一个臂超过对照的那批窗，在它自己的三个 seed 上是不是同一批」** ⇒ 本轮第一次在**增益图样的跨种子稳定性轴**上判读（复制单元仍是 seed，n=3；535 个全局有效窗；同种子的窗级配对 ΔCSI@2.0 向量），**裁决零变化**（事后诊断，**不构成新判据**：S3 护栏固定 thr=2.0 mm、复制单元固定为 seed），得六条新事实：

1. **先立参照：这套统计量是"有功效"的**。对照臂 B0 自己的跨种子逐窗 CSI 相关中位 **+0.8706**（三对 seed 0.8979/0.8706/0.7853）⇒ 若某个臂真有可复现的增益图样，本统计量**抓得到** ⇒ 下面各臂的"抓不到"是**真没有**，不是样本不够（对照臂三种子同号一致率 0.7626，作为三元统计量的噪声地板）。
2. **A4 是唯一有"正向且跨种子一致"增益图样的臂**：跨种子配对相关中位 **+0.2908**（控制掉对照窗型后的偏相关 **+0.4699**）、**3/3 对超过窗打乱零假设地板**、符号一致率 **0.645**、三种子同号一致率 **0.488**（全为正 0.387）。
3. **A3a／A3b 的增益图样在种子之间整体翻转**：跨种子相关中位 **−0.3934／−0.3839**、**3/3 对 |r| 超过地板**、**3 对里有 2 对为负** ⇒ 它们此前在配对口径上的"正增益"**没有稳定的窗级图样**（与第 40 轮「A4 增益 99.2% 来自 seed 0」、第 44 轮「种子对齐效应」同向，但更极端：这里翻的是**图样的符号**本身）。
4. **A1／A2 完全无图样**（|r| 超地板仅 1/3，中位 −0.0807／−0.0366）；**A4b 在偏相关下垮掉**（+0.0757 → −0.2545）⇒ A4b 也没有可复现图样。⇒ **六臂中只有 A4 留下了一个真实的、正向的、跨种子可复现的窗级增益图样**。
5. **但它不可与"机制增益"划等号**：新的**轴 C**（按真值事件量加权的「种子共有份额」，用**符号翻转置换零模型**标定）显示 **六臂全部未超过零假设 p95**（A1 0.0821/0.2876、A2 0.1670/0.2354、A3a 0.1332/0.2886、A3b 0.1437/0.3160、**A4 0.2749/0.2749**（伪 p=0.2550）、A4b 0.1571/0.2876）⇒ **没有任何一臂的增益含有统计上可辨的"跨种子共有"成分**；A4 最接近但仍未过线。
6. **而且即便那点"可复现"成分也全是过预报换来的**：新的**轴 E**（三种子同号窗的增益构成）显示 A4 的一致增益窗（n=207）**dPOD +0.1088、dFAR +0.0815、dBIASratio +0.7926** ⇒ 增益与过预报同时出现，**与 S3 的 `GAIN_WAS_AREA_OR_AMPLITUDE` 同向**；A3a／A3b 的一致增益窗同样带 dBIASratio +0.3340／+0.4161。⇒ **轴 A 的"可复现图样"与轴 C／轴 E 的"不可辨 + 过预报"合起来＝S3 裁决的第三台独立仪器印证**。

**旁证（轴 B，臂间共结构）**：同种子的臂间相关中位 **+0.4080**（偏相关 +0.4595）vs 跨种子 **−0.1167** ⇒ 同种子下各臂确实共动（共享种子/初始化/窗样本）；但**同族臂的超出量（+0.4850）与跨族臂（+0.2599）只差 +0.2251** ⇒ **这份共结构主要是"同种子"的效应，不是"同机制族"的效应**（最大同种子相关 0.8394 出现在 A3a+A3b，两者同属拓扑支）。**逐场次**：A4 在两个大场次（182 窗／137 窗）都为正（+0.194／+0.455），A3a／A3b 在两个场次都为负。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列（本轮配对 −0.002185～+0.012611、差值 −0.011802～+0.022079，与前 54 轮一致）；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变，**合格臂集合仍为空**；⑤ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）⇒ 新证据只进本节、上层 MEMORY.md 状态段与逐日日志。⑥ **新登记一条可复用的口径纪律**：**"臂的聚合增益为正"不等于"该臂有可复现的窗级增益图样"**——必须单独检验图样的跨种子稳定性（轴 A）与其中的种子共有份额（轴 C）；本轮 A3a／A3b 就是**图样整体翻转**、A4 则是**图样可复现但成分全为过预报**（同量级 0.15→0.17、0.21→0.23）。

"""

MEMORY_BULLET = ("- **版本沿革与派生布尔的订正**：v2.44 结案 → v2.45 分臂订正 → v2.46 三件实现待办的源码级根因定位 → "
                 "v2.47 数字订正 → **v2.48 订正 `scale_is_constant`（析取，触发的是相关性分支，不是「退化成常数」）**；"
                 "当前 **v2.48**（第 8–56 轮复核结论一致，不升版）。v2.48 的承重读数：**A4b 种子 2 的 `scale_cv` 3.0188 是六格最高**，"
                 "而**六格最小 `scale_cv` 1.2358 仍是底线 0.05 的 24.7 倍** ⇒ E45 全程没有任何一格真的退化成常数，"
                 "该格发生的是**量级塌缩**（`scale_mean` 0.0316）＋与真值脱钩（`scale_truth_corr` 0.2789，触发的是相关性分支）。\n")

LOG_BLOCK = """
### 2026-11-14｜第 56 轮（增益图样跨种子稳定性轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十五轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`，MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245_row`／`_e45_docs_numbers` `VERDICT: PASS`；`_check_v246_row`／`_check_v247_row`／`_check_v248_row` 曾 `FAIL`，**根因不在本轮**——上层 MEMORY.md 于 2026-11-14 08:23 被再次压缩，压缩掉的正是闸门要求的 `v2.46` / `3.0188` / `24.7` 三处引用 ⇒ 本轮在同一文件内**修复**（事实取自 HANDOVER §4 第 43 轮与 MASTER v2.48）并复跑闸门。
- **新证据**——`tools/_e45_verify_round56.py`（只读本地，**POS 6/6 VERDICT: PASS**，产物 `tools/_evals/e45/round56_gain_pattern_stability.json`）：前 55 轮判过聚合层／机制探针／S1 结构量／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并／剂量-反应／指标族／护栏标定，**从未问过「一个臂超过对照的那批窗，在它自己的三个 seed 上是不是同一批」** ⇒ 本轮首次在**增益图样跨种子稳定性轴**判读（535 个全局有效窗，同种子窗级配对 ΔCSI@2.0 向量），**裁决零变化**。六条新事实：① **先立参照**：对照臂 B0 自己的跨种子逐窗 CSI 相关中位 **+0.8706** ⇒ 本统计量有功效 ⇒ 各臂的"抓不到"是真没有；② **A4 是唯一有正向且跨种子一致的增益图样**（中位 r **+0.2908**、偏相关 **+0.4699**、3/3 对超地板、符号一致 0.645、三种子同号 0.488）；③ **A3a／A3b 的增益图样在种子之间整体翻转**（中位 r **−0.3934／−0.3839**，3/3 对 |r| 超地板、2/3 对为负）；④ A1／A2 无图样（超地板仅 1/3）、A4b 在偏相关下垮掉（+0.0757→−0.2545）；⑤ **轴 C**（真值加权的种子共有份额 ＋ 符号翻转置换零模型）**六臂全部未过零假设 p95**（A4 0.2749/0.2749、伪 p 0.2550 最接近）⇒ 无任何一臂含统计上可辨的跨种子共有成分；⑥ **轴 E**（三种子同号窗的构成）显示 A4 的一致增益窗（n=207）**dPOD +0.1088、dFAR +0.0815、dBIASratio +0.7926** ⇒ 那一丁点可复现成分仍是过预报换来的，**与 S3 `GAIN_WAS_AREA_OR_AMPLITUDE` 同向（第三台独立仪器印证）**。旁证（轴 B）：同种子臂间相关中位 +0.4080 vs 跨种子 −0.1167，但同族超出 +0.4850 与跨族 +0.2599 仅差 +0.2251 ⇒ 共结构主要是「同种子」效应而非「同机制族」效应。
- **新登记一条口径纪律**：**「臂的聚合增益为正」≠「该臂有可复现的窗级增益图样」**——须单独检验图样的跨种子稳定性（轴 A）与其中的种子共有份额（轴 C）。本轮 A3a／A3b 是**图样整体翻转**、A4 是**图样可复现但成分全为过预报**。
- **裁决零变化**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空** ⇒ 六臂判负、不投十 seed 算力；**MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**。
"""

PROJ_LOG_BLOCK = """
### 2026-11-14｜第 56 轮（增益图样跨种子稳定性轴的首次判读）：状态复核（无远端改动，裁决不变）

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；21 份评测齐、13 份 json 齐全、队列空、无 E46+ 目录；远端↔本地 **md5 13/13 全同**。**未下载、未 sbatch、未改远端状态、未动 `docs/`（MASTER 仍 v2.48）。**
- **闸门**：`_check_tables`／`_check_v245_row`／`_e45_docs_numbers` `VERDICT: PASS`；`_check_v246_row`／`_check_v247_row`／`_check_v248_row` 曾 `FAIL`，**根因不在本轮**——上层 MEMORY.md 于 08:23 被再次压缩，压缩掉了闸门要求的 `v2.46` / `3.0188` / `24.7` 引用 ⇒ 本轮在同文件内修复后复跑闸门。
- **新证据**——`tools/_e45_verify_round56.py`（只读本地，**POS 6/6 VERDICT: PASS**，产物 `tools/_evals/e45/round56_gain_pattern_stability.json`）：首次在**增益图样跨种子稳定性轴**判读（535 个全局有效窗；复制单元仍是 seed）⇒ **裁决零变化**。① 对照臂自身跨种子逐窗 CSI 相关中位 **+0.8706** ⇒ 统计量有功效；② **A4 是唯一正向且跨种子一致的增益图样**（中位 r +0.2908、偏相关 +0.4699、3/3 对超地板）；③ **A3a／A3b 图样在种子间整体翻转**（中位 r −0.3934／−0.3839，2/3 对为负）；④ A1／A2 无图样、A4b 偏相关下垮掉；⑤ **轴 C**（真值加权种子共有份额＋符号翻转置换零模型）**六臂全部未过零假设 p95** ⇒ 无可辨的跨种子共有成分；⑥ **轴 E**：A4 一致增益窗（n=207）dPOD +0.1088、dFAR +0.0815、dBIASratio +0.7926 ⇒ 可复现成分仍是过预报换来的，**与 S3 同向**。
- **新口径纪律**：**「聚合增益为正」≠「有可复现的窗级增益图样」**——须单独检验图样稳定性与种子共有份额。
- **裁决零变化**：12/12 区间全跨零、六臂 `s3_verdict` 不变、合格臂集合为空 ⇒ 六臂判负、不投十 seed 算力；**MASTER 不升版**。
"""

AUTOMATION_BLOCK = """
### 2026-11-14｜第 56 轮（增益图样跨种子稳定性轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十五轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`，MASTER 仍 v2.48）。**

- **远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；无 E46+ 目录。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245_row`／`_e45_docs_numbers` `VERDICT: PASS`；`_check_v246_row`／`_check_v247_row`／`_check_v248_row` 曾 `FAIL`，**根因不在本轮**——上层 MEMORY.md 于 2026-11-14 08:23 被再次压缩，压缩掉的正是闸门要求的 `v2.46` / `3.0188` / `24.7` 三处引用 ⇒ 本轮在同一文件内**修复**并复跑闸门。
- **新证据**——`tools/_e45_verify_round56.py`（只读本地，**POS 6/6 VERDICT: PASS**，产物 `tools/_evals/e45/round56_gain_pattern_stability.json`）：首次在**增益图样跨种子稳定性轴**判读（535 个全局有效窗，同种子窗级配对 ΔCSI@2.0 向量；复制单元仍是 seed）⇒ **裁决零变化**。① **先立参照**：对照臂 B0 自己的跨种子逐窗 CSI 相关中位 **+0.8706** ⇒ 统计量有功效 ⇒ 各臂"抓不到"是真没有；② **A4 是唯一有正向且跨种子一致的增益图样**（中位 r **+0.2908**、偏相关 **+0.4699**、3/3 对超窗打乱地板、符号一致 0.645、三种子同号 0.488）；③ **A3a／A3b 的增益图样在种子之间整体翻转**（中位 r **−0.3934／−0.3839**、3/3 对 |r| 超地板、2/3 对为负）；④ A1／A2 无图样（超地板仅 1/3）、A4b 偏相关下垮掉（+0.0757→−0.2545）；⑤ **轴 C**（真值事件量加权的种子共有份额 ＋ 符号翻转置换零模型）**六臂全部未超过零假设 p95**（A1 0.0821/0.2876、A2 0.1670/0.2354、A3a 0.1332/0.2886、A3b 0.1437/0.3160、**A4 0.2749/0.2749** 伪 p 0.2550、A4b 0.1571/0.2876）⇒ 无任何一臂含统计上可辨的跨种子共有成分；⑥ **轴 E**（三种子同号窗的增益构成）显示 A4 的一致增益窗（n=207）**dPOD +0.1088、dFAR +0.0815、dBIASratio +0.7926** ⇒ 那一丁点可复现成分仍是过预报换来的，**与 S3 `GAIN_WAS_AREA_OR_AMPLITUDE` 同向（第三台独立仪器印证）**；A3a／A3b 一致增益窗同样带 dBIASratio +0.3340／+0.4161。**旁证（轴 B）**：同种子臂间相关中位 +0.4080 vs 跨种子 −0.1167，但同族超出 +0.4850 与跨族 +0.2599 仅差 +0.2251 ⇒ 共结构主要是「同种子」效应而非「同机制族」效应（最大同种子相关 0.8394 出现在 A3a+A3b）。
- **新登记一条口径纪律**：**「臂的聚合增益为正」≠「该臂有可复现的窗级增益图样」**——须单独检验图样的跨种子稳定性（轴 A）与其中的种子共有份额（轴 C）。
- **裁决零变化**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空** ⇒ 六臂判负、不投十 seed 算力；**MASTER 不升版**。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**。
"""


def insert_before(path: Path, anchor: str, block: str, marker: str) -> str:
    """Insert block before anchor (once). Returns 'changed' / 'unchanged'."""
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return "unchanged"
    if anchor not in text:
        raise SystemExit("ANCHOR NOT FOUND in %s: %r" % (path, anchor[:60]))
    text = text.replace(anchor, block + "\n" + anchor, 1)
    path.write_text(text, encoding="utf-8")
    return "changed"


def append_once(path: Path, block: str, marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return "unchanged"
    if not text.endswith("\n"):
        text += "\n"
    text += block
    path.write_text(text, encoding="utf-8")
    return "changed"


def replace_once(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return "unchanged"
    if old not in text:
        raise SystemExit("OLD STRING NOT FOUND in %s" % path)
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    return "changed"


results = []

# 1. HANDOVER: new section before the archived execution log
results.append(("HANDOVER", insert_before(
    HANDOVER, "### 下面是结案前的执行记录（存档用，不必重跑）", HANDOVER_BLOCK, MARK)))

# 2. MEMORY.md status section: round counter + restored citations
mem_text = MEMORY.read_text(encoding="utf-8")
old_round = ("- **第 8–55 轮独立复核结论完全一致、无新增偏离**")
new_round = ("- **第 8–56 轮独立复核结论完全一致、无新增偏离**")
if "第 8–56 轮独立复核结论完全一致" in mem_text:
    results.append(("MEMORY-round-counter", "unchanged"))
else:
    if old_round not in mem_text:
        raise SystemExit("MEMORY round counter anchor not found")
    mem_text = mem_text.replace(old_round, new_round, 1)
    mem_text = mem_text.replace("第 40–55 轮新增的均为", "第 40–56 轮新增的均为", 1)
    mem_text = mem_text.replace("`HANDOVER_E45.md` §4⑤–⑲ 与逐日日志",
                                "`HANDOVER_E45.md` §4⑤–⑳ 与逐日日志", 1)
    MEMORY.write_text(mem_text, encoding="utf-8")
    results.append(("MEMORY-round-counter", "changed"))

results.append(("MEMORY-version-history", replace_once(
    MEMORY, "- **三条定位**：", MEMORY_BULLET + "- **三条定位**：")))

# 3. daily logs
results.append(("LOG-1114", append_once(LOG_1114, LOG_BLOCK, MARK)))
results.append(("LOG-1111", append_once(LOG_1111, LOG_BLOCK, MARK)))
results.append(("PROJ-LOG-1114", append_once(PROJ_LOG_1114, PROJ_LOG_BLOCK, MARK)))
results.append(("AUTOMATION-MEM", append_once(AUTOMATION_MEM, AUTOMATION_BLOCK, MARK)))

# ------------------------------------------------------------------- verify
print("=== write-back results ===")
for name, st in results:
    print("  %-24s %s" % (name, st))

print("\n=== marker counts (must be exactly 1 where inserted) ===")
for name, path in (("HANDOVER", HANDOVER), ("LOG-1114", LOG_1114), ("LOG-1111", LOG_1111),
                   ("PROJ-LOG-1114", PROJ_LOG_1114), ("AUTOMATION-MEM", AUTOMATION_MEM)):
    n = path.read_text(encoding="utf-8").count(MARK)
    print("  %-24s %d" % (name, n))
    if n != 1:
        raise SystemExit("marker count != 1 in %s" % name)

mem = MEMORY.read_text(encoding="utf-8")
for token, label in (("v2.46", "v2.46"), ("3.0188", "3.0188"), ("24.7", "24.7"),
                     ("相关性分支", "correlation branch"), ("量级塌缩", "magnitude collapse")):
    n = mem.count(token)
    print("  MEMORY carries %-20s %d occurrence(s)" % (label, n))
    if n < 1:
        raise SystemExit("MEMORY missing token %s" % label)

print("\nVERDICT: PASS (write-back)")
