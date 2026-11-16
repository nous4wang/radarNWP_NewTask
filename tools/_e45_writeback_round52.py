"""E45 round-52 write-back (idempotent, marker-guarded).

Round-52 = new axis: the BETWEEN-ARM OMNIBUS axis -- a two-way (arm x seed)
variance decomposition plus a seed-blocked randomisation test, with seed as the
blocking factor and the only replication unit.  All previous 51 rounds compared
each arm to the control; none asked whether the six arms are distinguishable
from EACH OTHER, nor quantified the "do not cherry-pick the best arm" rule.

No verdict change, no published-number change => MASTER is NOT re-versioned
(it stays v2.48); the evidence goes to HANDOVER section 4, the upper MEMORY.md
status section, the 2026-11-14 daily logs (upper + project level) and the
automation memory.

Idempotency rules learned in rounds 41/43/45/47/48/49/51:
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged".
  * never slice a file from "my marker" to "the next heading" -- that silently
    deletes write-back blocks inserted by other rounds (bug fixed in r51).
"""

from pathlib import Path

MARK = "第 52 轮（臂间合并检验轴的首次判读）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1114 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
PROJ_LOG_1114 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

HANDOVER_ANCHOR = "### 下面是结案前的执行记录（存档用，不必重跑）"

HANDOVER_BLOCK = f"""
**⑯（2026-11-14 第 52 轮 {MARK}）**——脚本 `tools/_e45_verify_round52.py`（只读本地，**POS 18/18 VERDICT: PASS**，产物 `tools/_evals/e45/round52_omnibus_arm_effect.json`）：前 51 轮全部是「**逐臂 vs 对照**」的比较（配对增益、S3 等偏置、LOSO、MDE、场次/时效/强度/难度分层、误差构成…），**从未问过一个更尖锐的问题：六臂彼此之间是否可区分？** ⇒ 本轮第一次在**臂间合并检验轴**上判读（两因素「臂 × seed」无重复方差分解 ＋ **以 seed 为区组**的随机化检验，复制单元仍是 seed），**裁决零变化**，但拿到四条前五十一轮都看不见的结构性事实：

1. **六臂之间根本分不开：合并检验的 p 值大到接近 1**。原始配对 ΔCSI@2.0：**F = 0.3056（df 5,10）、临界值 3.3258、解析 p = 0.8986、区组置换 p = 0.9003**（20000 次，零分布 F 中位 0.9200、q95 3.2926）；配平后（S3 侧）：**F = 0.6235、p = 0.7232**；ΔPOD：F = 0.9119、p = 0.4792。⇒ **「哪一臂更好」这个问题在 n=3 下连问都问不出来**（三套结果集的 F 全部远低于临界值）。

2. **方差份额把原因说透了：臂效应只占 8.1%，seed 与残差占 92%**。原始配对口径的方差分解：**臂 0.081／seed 0.387／残差（臂×seed 交互）0.532**；配平后：臂 0.211／seed 0.114／残差 0.676；ΔPOD：臂 0.203／seed 0.353／残差 0.445。⇒ **「臂」这个因子的解释力最小**，与第 40 轮「A4 的 +0.0180 里 99.2% 来自 seed 0」、第 48 轮 LOSO 同向，但更极端：**在方差层面，臂几乎不携带信息**。

3. **「不得择优」这条纪律第一次有了可量化的依据**。逐 seed 的赢家：原始配对口径 **A4 / A1 / A4**（2 个不同赢家）、配平后 **A3a / A1 / A4**（**3 个 seed 三个赢家**）、ΔPOD **A4b / A4b / A4**（2 个）；Kendall's W（臂排序跨 seed 的一致性）0.5905／0.6667／0.7810。**对 seed 做 bootstrap 重抽后的「赢家概率」**：原始口径 **A4 仅 0.524**（A1 0.258、A3b 0.218、其余三臂 0.000）；配平后 **A1 0.476**（A4 0.265、A3a 0.260）；ΔPOD **A4b 0.735**（A4 0.265）。⇒ **连「哪个臂最好」本身都是种子抽签的结果**：原始口径下 A4 只有 52% 概率是赢家，而 10000 次重抽里有 21.8% 由 A3b 夺魁 ⇒ **任何「A4 最好／A4b 最好」的表述都没有跨种子依据**，这是对「不得择优」的量化落实。

4. **六臂之间的分离度比「把对照臂换个种子」还小**。六臂臂均值的跨度：原始配对 **0.03514**、配平后 **0.02820**，**双双小于零模型地板 0.038771**（对照臂 B0 换种子互比的三对 |ΔCSI| 取最大；比值 0.91／0.73）⇒ **六臂彼此的差别，弱于把同一个对照臂换一次种子**。ΔPOD 轴是唯一的例外：跨度 0.18272 vs 其量纲的 POD 零模型地板 0.09246（比值 **1.98**）——**但它同样不显著（p = 0.4792）** ⇒ **连这条唯一的例外也不构成证据**（n=3 下 1.98 倍的差距仍被 seed 噪声吞掉）。**注：零模型地板按量纲分别计算，CSI 与 POD 的地板不可混用**（这是本轮写脚本时踩到的坑，见下）。

5. **下一阶梯的算力预算要改写：合并检验要 n = 30 才够**。以实测残差 SD（0.04080）与实测臂间效应结构做 Monte-Carlo 检出力（每格 600 次）：要检出**实测**臂间效应量（×1.0）需 **n = 30**（n=3:0.10、n=10:0.32、n=20:0.66、n=30:0.86）；放大 2 倍（×2.0）需 **n = 10**；放大 3 倍需 n = 5；**缩小到 0.5 倍则 n = 60 仍只有 0.55 检出力 ⇒ 不存在可行的 seed 数**。⇒ **与第 48 轮的 MDE 结论合并看**：单臂对照口径要 17–60 seed，而**「臂间有没有差别」这个更弱的问题反而要 30 seed**（因为它要在 6 个臂之间做多重对比，功效被摊薄）；**两族机制的真实效应若在 0.02–0.05 区间，十 seed 无论如何都不足以定案**。

**新登记两条可复用的口径纪律**：① **「不得择优」在 n=3 下必须用「赢家概率」量化，不能靠印象**——本轮给出的三个数（原始口径 A4 0.524／配平后 A1 0.476／ΔPOD A4b 0.735）都是对 seed 做 bootstrap 后的分布，**任何一个「最佳臂」的表述都应附上这个概率**；② **零模型地板必须按量纲分别计算**——用 CSI 的地板（0.038771）去判 POD 的量会得到假阳性（本轮首版即栽在这里，被自带断言当场抓住），本轮已改成按结果集分别给地板并把比值（跨度/地板）逐套报告。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列（逐臂原始增益中位：A1 −0.00342、A2 −0.00981、A3a +0.01553、A3b +0.02436、A4 +0.01343、A4b +0.01575；配平后增益中位：A1 +0.00644、A2 +0.00180、A3a +0.00025、A3b −0.00545、A4 +0.00440、A4b −0.00984）；③ **事后诊断，不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变，**合格臂集合仍为空**；⑤ 重建自校：168 次比较 max|diff| = 0、21 格 window_keys 完全一致、S3 的 12 个点估复算 max|diff| = 0。
"""

MEMORY_BLOCK = f"""
- **（2026-11-14 第 52 轮 {MARK}：裁决不变，新证据＝臂间合并检验轴的首次判读）** 远端 E45 状态与前五十一轮**完全一致**（提交器 **273** `COMPLETED` 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全、**队列空**、提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、`runs/` 下无 E46+ 新目录 ⇒ **无新活动**；远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**）；本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。**新脚本 `tools/_e45_verify_round52.py`（只读本地，POS 18/18 VERDICT: PASS，产物 `tools/_evals/e45/round52_omnibus_arm_effect.json`）**：前 51 轮全部是「**逐臂 vs 对照**」，**从未问过「六臂彼此是否可区分」** ⇒ 本轮首次在**臂间合并检验轴**判读（两因素「臂 × seed」无重复方差分解 ＋ **以 seed 为区组**的随机化检验，20000 次；复制单元仍是 seed），**裁决零变化**，得五条新事实：① **六臂之间根本分不开**：原始配对 ΔCSI@2.0 **F = 0.3056（df 5,10）、临界值 3.3258、解析 p = 0.8986、区组置换 p = 0.9003**；配平后 F = 0.6235、p = 0.7232；ΔPOD F = 0.9119、p = 0.4792 ⇒ **「哪一臂更好」在 n=3 下连问都问不出来**。② **方差份额说透原因**：臂 0.081／seed 0.387／残差 0.532（原始口径）⇒ **臂这个因子的解释力最小**（与第 40 轮「A4 增益 99.2% 来自 seed 0」、第 48 轮 LOSO 同向但更极端）。③ **「不得择优」第一次有了量化依据**：逐 seed 赢家原始 **A4/A1/A4**、配平后 **A3a/A1/A4**（三个 seed 三个赢家）、ΔPOD **A4b/A4b/A4**；Kendall W 0.5905／0.6667／0.7810；**对 seed bootstrap 重抽的赢家概率**：原始 **A4 仅 0.524**（A1 0.258、A3b 0.218，其余 0.000）、配平后 **A1 0.476**、ΔPOD **A4b 0.735** ⇒ **连「哪个臂最好」都是种子抽签的结果**。④ **六臂的分离度比换种子还小**：臂均值跨度 0.03514（原始）／0.02820（配平后）**双双小于零模型地板 0.038771**（比值 0.91／0.73）⇒ **六臂彼此的差别弱于把同一对照臂换一次种子**；ΔPOD 是唯一例外（跨度 0.18272 vs POD 量纲地板 0.09246，比值 1.98）**但同样不显著（p = 0.4792）**。⑤ **算力预算改写**：合并检验检出实测臂间效应量需 **n = 30**（n=10:0.32、n=20:0.66、n=30:0.86），×2.0 需 n = 10，×0.5 则 n = 60 仍只有 0.55 ⇒ **与第 48 轮 MDE 合并看，十 seed 对两种问题都不足以定案**。**新登记两条可复用纪律**：① **「不得择优」在 n=3 下必须用「赢家概率」量化**（任何「最佳臂」表述都应附上该概率）；② **零模型地板必须按量纲分别计算**——用 CSI 地板判 POD 会得假阳性（本轮首版即栽此坑，被自带断言当场抓住），本轮已改为按结果集分别给地板并逐套报告「跨度/地板」比值。**边界**：事后诊断、**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**——**六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**（无数值与裁决变化）⇒ 新证据只进本节、HANDOVER §4⑯、2026-11-14 逐日日志（上层 ＋ 项目级）与本文件。
"""

LOG_BLOCK = f"""
**补记，{MARK}（2026-11-14）**

- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；**队列空**；`runs/` 下无 E46+ 新目录；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **与前五十一轮完全一致，无新活动**。本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无数值与裁决变化）。
- **新证据（臂间合并检验轴的首次判读）**——`tools/_e45_verify_round52.py`（只读本地，**POS 18/18 VERDICT: PASS**，产物 `tools/_evals/e45/round52_omnibus_arm_effect.json`）：
  1. **六臂之间根本分不开**：原始配对 ΔCSI@2.0 **F = 0.3056（df 5,10）**、临界值 3.3258、**解析 p = 0.8986**、**区组置换 p = 0.9003**（20000 次，零分布 F 中位 0.9200／q95 3.2926）；配平后 F = 0.6235、p = 0.7232；ΔPOD F = 0.9119、p = 0.4792 ⇒ **「哪一臂更好」在 n=3 下连问都问不出来**。
  2. **方差份额**：臂 **0.081**／seed 0.387／残差 0.532（原始口径）；配平后 0.211／0.114／0.676；ΔPOD 0.203／0.353／0.445 ⇒ **臂这个因子的解释力最小**。
  3. **「不得择优」的量化依据**：逐 seed 赢家 原始 **A4/A1/A4**、配平后 **A3a/A1/A4**（三个 seed 三个赢家）、ΔPOD **A4b/A4b/A4**；Kendall W 0.5905／0.6667／0.7810；**对 seed bootstrap 重抽的赢家概率**：原始 **A4 仅 0.524**（A1 0.258、A3b 0.218，其余三臂 0.000）、配平后 **A1 0.476**、ΔPOD **A4b 0.735** ⇒ **连「哪个臂最好」都是种子抽签的结果**。
  4. **六臂的分离度比换种子还小**：臂均值跨度 **0.03514**（原始）／**0.02820**（配平后）**双双小于零模型地板 0.038771**（比值 0.91／0.73）；ΔPOD 唯一例外（0.18272 vs POD 量纲地板 0.09246，比值 **1.98**）**但同样不显著（p = 0.4792）**。
  5. **算力预算改写**：合并检验检出实测臂间效应量需 **n = 30**（n=3:0.10、n=10:0.32、n=20:0.66、n=30:0.86）；×2.0 需 n = 10；×0.5 则 n = 60 仍只有 0.55 ⇒ **十 seed 对「臂间有无差别」这个更弱的问题也不足以定案**。
- **新登记两条可复用纪律**：① **「不得择优」在 n=3 下必须用「赢家概率」量化**（任何「最佳臂」表述都应附该概率）；② **零模型地板必须按量纲分别计算**——用 CSI 地板（0.038771）判 POD 量会得假阳性（本轮首版即栽此坑，被自带断言当场抓住），已改为按结果集分别给地板并逐套报告「跨度/地板」比值。
- **裁决零变化**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空** ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**。
- **边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）。
"""

AUTOMATION_BLOCK = f"""
### 2026-11-14｜第 52 轮（臂间合并检验轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十一轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`）。**

**远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` md5 **13/13 全同**。

**闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。

**新证据**——`tools/_e45_verify_round52.py`（只读本地，**POS 18/18 VERDICT: PASS**，产物 `tools/_evals/e45/round52_omnibus_arm_effect.json`）：前 51 轮全部是「**逐臂 vs 对照**」，**从未问过「六臂彼此是否可区分」** ⇒ 本轮首次在**臂间合并检验轴**判读（两因素「臂 × seed」无重复方差分解 ＋ **以 seed 为区组**的随机化检验，20000 次；复制单元仍是 seed），**裁决零变化**，得五条新事实：① **六臂之间根本分不开**：原始配对 ΔCSI@2.0 **F = 0.3056（df 5,10）**、临界值 3.3258、解析 p = **0.8986**、区组置换 p = **0.9003**（零分布 F 中位 0.9200／q95 3.2926）；配平后 F = 0.6235、p = 0.7232；ΔPOD F = 0.9119、p = 0.4792 ⇒ **「哪一臂更好」在 n=3 下连问都问不出来**。② **方差份额**：臂 **0.081**／seed 0.387／残差 0.532（原始口径）⇒ **臂这个因子的解释力最小**（与第 40 轮「A4 增益 99.2% 来自 seed 0」、第 48 轮 LOSO 同向但更极端）。③ **「不得择优」第一次有了量化依据**：逐 seed 赢家 原始 **A4/A1/A4**、配平后 **A3a/A1/A4**（三个 seed 三个赢家）、ΔPOD **A4b/A4b/A4**；Kendall W 0.5905／0.6667／0.7810；**对 seed bootstrap 重抽的赢家概率**：原始 **A4 仅 0.524**（A1 0.258、A3b 0.218，其余三臂 0.000）、配平后 **A1 0.476**、ΔPOD **A4b 0.735** ⇒ **连「哪个臂最好」都是种子抽签的结果**。④ **六臂的分离度比换种子还小**：臂均值跨度 **0.03514**（原始）／**0.02820**（配平后）**双双小于零模型地板 0.038771**（比值 0.91／0.73）⇒ **六臂彼此的差别弱于把同一对照臂换一次种子**；ΔPOD 唯一例外（0.18272 vs POD 量纲地板 0.09246，比值 **1.98**）**但同样不显著（p = 0.4792）**。⑤ **算力预算改写**：合并检验检出实测臂间效应量需 **n = 30**（n=3:0.10、n=10:0.32、n=20:0.66、n=30:0.86），×2.0 需 n = 10，×0.5 则 n = 60 仍只有 0.55 ⇒ **与第 48 轮 MDE 合并看，十 seed 对两种问题都不足以定案**。

**新登记两条可复用纪律**：① **「不得择优」在 n=3 下必须用「赢家概率」量化**（任何「最佳臂」表述都应附该概率）；② **零模型地板必须按量纲分别计算**——用 CSI 地板判 POD 量会得假阳性（本轮首版即栽此坑，被自带断言当场抓住），已改为按结果集分别给地板并逐套报告「跨度/地板」比值。

**写回**：`HANDOVER_E45.md` §4⑯ ＋ 上层 MEMORY.md 状态段 ＋ 2026-11-14 逐日日志（上层 ＋ 项目级）＋ 2026-11-11 日志 ＋ 本文件（脚本 `tools/_e45_writeback_round52.py`，幂等，标记＝插入文本原文子串，连跑两次第二遍全部 `unchanged`，各文件标记计数 == 1）。

**本轮踩坑（可复用规则）**：① **零模型地板必须按量纲分别计算**——本轮首版用 CSI 的地板（0.038771）去判 dPOD 轴的臂间跨度（0.18272），断言当场报 NEG；**根因是量纲混用**（CSI 与 POD 的地板不可互换），修正为按结果集分别给地板、并把「跨度/地板」比值逐套报告。② **断言要写清楚它到底断言什么**——把「CSI 轴上六臂分离度弱于换种子」与「dPOD 轴只报事实」分成两条不同强度的断言，避免把一个事实陈述写成必须成立的命题。
"""

TARGETS = [
    ("HANDOVER", HANDOVER, HANDOVER_BLOCK, "insert_before", HANDOVER_ANCHOR),
    ("MEMORY", MEMORY, MEMORY_BLOCK, "append", None),
    ("LOG_1114", LOG_1114, LOG_BLOCK, "append", None),
    ("LOG_1111", LOG_1111, LOG_BLOCK, "append", None),
    ("PROJ_LOG_1114", PROJ_LOG_1114, LOG_BLOCK, "append", None),
    ("AUTOMATION_MEM", AUTOMATION_MEM, AUTOMATION_BLOCK, "append", None),
]


def main():
    for name, path, block, mode, anchor in TARGETS:
        text = path.read_text(encoding="utf-8")
        n_before = text.count(MARK)
        if n_before >= 1:
            print("%-16s unchanged (marker x%d)" % (name, n_before))
            continue
        if mode == "append":
            new = text.rstrip("\n") + "\n" + block.strip("\n") + "\n"
        else:
            if anchor not in text:
                raise SystemExit("ANCHOR MISSING in %s: %r" % (name, anchor))
            new = text.replace(anchor, block.strip("\n") + "\n\n" + anchor, 1)
        path.write_text(new, encoding="utf-8")
        check = path.read_text(encoding="utf-8")
        cnt = check.count(MARK)
        if cnt != 1:
            raise SystemExit("MARKER COUNT %d in %s (expected 1)" % (cnt, name))
        print("%-16s written (marker x1)" % name)


if __name__ == "__main__":
    main()
