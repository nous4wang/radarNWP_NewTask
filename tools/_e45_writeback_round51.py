"""E45 round-51 write-back (idempotent, marker-guarded).

Round-51 = new axis: the ERROR-COMPOSITION axis -- decompose each arm's dCSI
into a POD (hit-rate) contribution and a FAR (false-alarm) contribution from
the contingency tables, with a zero-model floor (control arm B0 with two
different seeds compared to each other), plus a cross-instrument check against
the S3 equal-bias result.

No verdict change, no published-number change => MASTER is NOT re-versioned
(it stays v2.48); the evidence goes to HANDOVER section 4, the upper MEMORY.md
status section, the 2026-11-14 daily logs (upper + project level) and the
automation memory.

Idempotency rules learned in rounds 41/43/45/47/48/49:
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged".
"""

from pathlib import Path

MARK = "第 51 轮（误差构成轴的首次判读）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1114 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
PROJ_LOG_1114 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

HANDOVER_ANCHOR = "### 下面是结案前的执行记录（存档用，不必重跑）"

HANDOVER_BLOCK = f"""
**⑮（2026-11-14 第 51 轮 {MARK}）**——脚本 `tools/_e45_verify_round51.py`（只读本地，**POS 7/7 VERDICT: PASS**，产物 `tools/_evals/e45/round51_error_composition.json`）：前 50 轮判过聚合层／机制探针层／S1 结构量层／窗级／风暴场次级／时效轴／强度·难度轴共七层，**从未把任何一臂的 ΔCSI 拆成「命中率(POD) 贡献」与「虚警率(FAR) 贡献」** ⇒ 本轮第一次在**误差构成轴**上判读，**裁决零变化**，但拿到四条前七层都看不见的结构性事实：

1. **先修代数再落笔：`CSI ≠ POD·(1−FAR)`**。正确恒等式是 **`CSI = POD(1−FAR) / (1−FAR + FAR·POD)`**（与 `h/(h+m+f)` 恒等）。本轮首版误用了 `POD·(1−FAR)`，被自带的代数恒等式断言当场抓住（max dev **0.064**）⇒ **分解必须用两因子 Shapley 拆（两条路径取平均，无路径依赖）**，本轮修正后 `contrib_POD + contrib_FAR ≡ ΔCSI` 逐格 **max|dev| < 1e−12**。

2. **六臂的「表面增益」几乎全部来自 POD 一侧**。跨 18 个（臂×seed）格：**`r(POD 贡献, ΔCSI) = +0.9141`**、`r(FAR 贡献, ΔCSI) = −0.1769`、`r(ΔPOD, ΔCSI) = +0.7673`、`r(ΔFAR, ΔCSI) = +0.0132` ⇒ **增益不是「少报虚警」换来的，而是「多命中」换来的**；而多命中在阈值 2.0 mm 上必须靠**同时多报虚警**来实现（A4b 的 FAR 贡献中位 **−0.05280**、ΔF/真值中位 **+2.08925**，即虚警事件数涨到真值事件数的两倍以上）。

3. **零模型地板（对照臂 B0 换种子互比，纯 seed 噪声，三对取最大 |值|）**：ΔCSI **0.038771**、POD 贡献 **0.046590**、FAR 贡献 **0.012257**、ΔF/真值 **0.333289**。逐臂通过率（配对口径，中位数）：**A4b 是唯一在 POD 贡献（3/3）、FAR 贡献（3/3）、ΔF/真值（3/3）三项上都越过地板的臂**（POD 贡献中位 +0.08631）；**A4** 的 FAR 贡献与 ΔF/真值各 2/3；**A3a 0/3、A3b 1/3、A1 的 POD 贡献 2/3、A2 全部 0/3**。⇒ **A3a／A3b 那些「配对口径为正」的增益在本轴上落在噪声地板之内**，与第 44 轮「种子对齐效应」、第 47／49／50 轮的场次·时效·强度分层结论同向。

4. **跨仪器交叉核对（两台不同的仪器给出同一结论）**：组成轴只用**列联表计数**，而 S3 等偏置用的是**预报场的顺序统计**（全局标量配平）。**过预报越厉害的格，等偏置后掉掉的 CSI 越多**：**`r(ΔF/真值, 等偏置掉掉的 CSI) = +0.6415`**、**`r(配平前偏置比 − 1, 掉掉的 CSI) = +0.6420`**（n=18）；逐臂等偏置掉掉的 CSI 中位：**A3b +0.02981、A4 +0.02539、A4b +0.02559、A3a +0.01528**，而 A1（−0.00986）／A2（−0.00887）为负（它们本就欠预报）。⇒ **S3「仍是面积／幅度效应」的裁决，在一台从未参与过 S3 的仪器上独立复现**；同时 **S3 的 12 个点估由逐 seed 原始／配平增益重算，max|diff| = 0**（12/12）。

**A4b 逐 seed 读数（配对口径）**：原始增益 **+0.01575／+0.04825／−0.11530**、配平后增益 **−0.00984／+0.00371／−0.06620** ⇒ **符号在两个 seed 上翻负** ⇒ A4b 唯一「三项全过地板」的那一格**仍是臂不稳，不是机制成立**；它同时是 ΔF/真值最大（中位 +2.09）、FAR 贡献最负（中位 −0.0528）的一格 ⇒ **与第 40 轮「A4b 有一个 seed 是 −0.1153」、第 45 轮「S1 场级塌缩」、第 49 轮「长时效过预报最重」同向**。

**新登记两条可复用的口径纪律**：① **`s3_equalbias.json` 的 `achieved_bias` 是「配平后」的偏置**（实测其与 `control_bias` 的比值恒为 1，最大偏离 **8.06e−07**）⇒ **该比值按构造恒等于 1，永远不能当协变量做相关**；配平**前**的偏置比必须由窗表自算（本轮用 `bias_ratio_pre`）。② **任何「把一个指标拆成两项贡献」的分解，都必须先写代数恒等式断言**，否则拆分的和与原量对不上（本轮首版即栽在这里，断言当场抓到 0.064 的偏差）。

**本轮修掉一个会静默毁数据的工具缺陷（`tools/_e45_verify_round50.py` 的内嵌写回）**：该脚本内嵌了一个 `upsert()`，旧实现把文件从**自己的标记**一路切到**下一个 `### ` 标题**为止 ⇒ **夹在中间的、更新一轮的写回块会被无声删除**（本轮写回连续两次成功、随即被第 50 轮脚本抹掉，5 个目标里 4 个被清空）。已改为：① 块文本已存在则**原样不动**（`already-present`）；② 只在标记存在时**就地替换**旧块、保留其位置，且旧块边界改按**外来边界**（其他轮的标题／项目符号）判定，**不再用「下一个 `### `」做无差别切割**。修后重跑第 50 轮脚本：仍 `VERDICT: PASS (POS 6/6)`，且第 51 轮的写回块**未被抹除**（5 个目标 r51 标记计数均 == 1）。**教训（可复用）**：**任何做「从标记切到下一标题」式删除的写回函数，都会杀掉并行新增的块**；写回必须改成「就地替换」或「文本已存在则跳过」，否则每轮新证据都会被上一轮的脚本悄悄吃掉。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列（差值口径下 A3a −0.01180、A3b −0.00576 为负，与配对口径反向 ⇒ **仍有种子对齐效应**；差值口径真正高于对照的仍只有 **A4 +0.02208、A4b +0.01575**，正是 POD／BIAS 爆表的两臂）；③ **事后诊断，不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变，**合格臂集合仍为空**。
"""

MEMORY_BLOCK = f"""
- **（2026-11-14 第 51 轮 {MARK}：裁决不变，新证据＝误差构成轴（hits/misses/虚警）的首次判读）** 远端 E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298）、`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全、**队列空**、提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、`runs/` 下无 E46+ 新目录 ⇒ **与前五十轮完全一致，无新活动**；远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**；本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）＋逐轮复核（`_e45_verify_round43…50`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。**新脚本 `tools/_e45_verify_round51.py`（只读本地，POS 7/7 VERDICT: PASS，产物 `tools/_evals/e45/round51_error_composition.json`）**：前 50 轮判过聚合／机制探针／S1 结构量／窗级／场次级／时效／强度·难度共七层，**从未把 ΔCSI 拆成「POD 贡献」与「FAR 贡献」** ⇒ 本轮首次在**误差构成轴**判读，**裁决零变化**，得四条新事实：① **先修代数**——`CSI ≠ POD·(1−FAR)`，正确恒等式是 **`CSI = POD(1−FAR)/(1−FAR+FAR·POD)`**；首版误用被自带恒等式断言当场抓住（dev 0.064），改用**两因子 Shapley 拆**后 `contrib_POD + contrib_FAR ≡ ΔCSI` 逐格 max|dev| < 1e−12。② **六臂的表面增益几乎全来自 POD 一侧**：跨 18 格 **`r(POD 贡献, ΔCSI) = +0.9141`**、`r(FAR 贡献, ΔCSI) = −0.1769` ⇒ **不是「少报虚警」换来的，是「多命中」换来的**，而多命中在 2.0 mm 上必须靠同时多报虚警实现。③ **零模型地板**（对照臂 B0 换种子互比，三对取最大）ΔCSI **0.038771**／POD 贡献 **0.046590**／FAR 贡献 **0.012257**／ΔF/真值 **0.333289**；逐臂通过率：**A4b 是唯一在 POD 贡献（3/3）、FAR 贡献（3/3）、ΔF/真值（3/3）三项都过地板的臂**（POD 贡献中位 +0.08631），A4 的 FAR 贡献与 ΔF/真值各 2/3，**A3a 0/3、A3b 1/3、A2 全 0/3** ⇒ A3a／A3b 的「配对口径为正」落在噪声地板之内。④ **跨仪器交叉核对**（组成轴＝列联表计数；S3＝场顺序统计的全局标量配平，**两台不同的仪器**）：**`r(ΔF/真值, 等偏置掉掉的 CSI) = +0.6415`**、**`r(配平前偏置比−1, 掉掉的 CSI) = +0.6420`**（n=18）⇒ **S3「仍是面积／幅度效应」在一台未参与过 S3 的仪器上独立复现**；S3 的 12 个点估由逐 seed 重算 **max|diff| = 0**。**A4b 逐 seed**：原始增益 +0.01575／+0.04825／**−0.11530**、配平后 −0.00984／+0.00371／**−0.06620** ⇒ **符号两个 seed 翻负 ⇒ 仍是臂不稳，不是机制成立**。**新登记两条可复用纪律**：① **`s3_equalbias.json` 的 `achieved_bias` 是配平后的偏置，其与 `control_bias` 的比值按构造恒为 1（实测最大偏离 8.06e−07）⇒ 永远不能当协变量做相关**，配平前的偏置比必须由窗表自算；② **任何「把一个指标拆成两项贡献」的分解，都必须先写代数恒等式断言**，否则拆分的和与原量对不上。**边界**：事后诊断、**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**。
"""

LOG_BLOCK = f"""
**补记，{MARK}（2026-11-14）**

- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；**队列空**；`runs/` 下无 E46+ 新目录；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **与前五十轮完全一致，无新活动**。本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）＋逐轮复核（`_e45_verify_round43…50`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无数值与裁决变化）。
- **新证据（误差构成轴的首次判读）**——`tools/_e45_verify_round51.py`（只读本地，**POS 7/7 VERDICT: PASS**，产物 `tools/_evals/e45/round51_error_composition.json`）：
  1. **代数先修**：`CSI ≠ POD·(1−FAR)`；正确恒等式 **`CSI = POD(1−FAR)/(1−FAR+FAR·POD)`**。首版误用被自带恒等式断言当场抓住（dev **0.064**）⇒ 改用**两因子 Shapley 拆**（无路径依赖），修正后 `contrib_POD + contrib_FAR ≡ ΔCSI` 逐格 max|dev| < 1e−12。
  2. **增益几乎全来自 POD 一侧**：跨 18 格 **`r(POD 贡献, ΔCSI) = +0.9141`**、`r(FAR 贡献, ΔCSI) = −0.1769` ⇒ **不是少报虚警换来的，是多命中换来的**；而多命中在 2.0 mm 上必须靠同时多报虚警实现（A4b FAR 贡献中位 **−0.05280**、ΔF/真值中位 **+2.08925**）。
  3. **零模型地板**（对照臂 B0 换种子互比，三对取最大 |值|）：ΔCSI **0.038771**／POD 贡献 **0.046590**／FAR 贡献 **0.012257**／ΔF/真值 **0.333289**。**通过率**：**A4b 唯一三项全过**（POD 贡献 3/3、POD 中位 +0.08631；FAR 贡献 3/3；ΔF/真值 3/3），A4 的 FAR 贡献与 ΔF/真值各 2/3，**A3a 0/3、A3b 1/3、A2 全 0/3** ⇒ A3a／A3b 的「配对口径为正」落在噪声地板内。
  4. **跨仪器交叉核对（组成轴＝列联表计数；S3＝场顺序统计的全局标量配平）**：**`r(ΔF/真值, 等偏置掉掉的 CSI) = +0.6415`**、**`r(配平前偏置比−1, 掉掉的 CSI) = +0.6420`**（n=18）⇒ **S3 裁决在另一台仪器上独立复现**；S3 的 12 个点估由逐 seed 重算 **max|diff| = 0**。逐臂等偏置掉掉的 CSI 中位：A3b +0.02981、A4b +0.02559、A4 +0.02539、A3a +0.01528，A1 −0.00986、A2 −0.00887（欠预报侧为负）。
  5. **A4b 逐 seed**：原始增益 +0.01575／+0.04825／**−0.11530**、配平后 −0.00984／+0.00371／**−0.06620** ⇒ **符号两个 seed 翻负 ⇒ 仍是臂不稳**（与第 40／45／49 轮同向）。
- **新登记两条可复用纪律**：① **`s3_equalbias.json` 的 `achieved_bias` 是配平后的偏置，其与 `control_bias` 的比值按构造恒为 1（实测最大偏离 **8.06e−07**）⇒ 永远不能当协变量做相关**；配平前的偏置比必须由窗表自算。② **任何「把一个指标拆成两项贡献」的分解，都必须先写代数恒等式断言**。
- **裁决零变化**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空** ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**（无数值与裁决变化）。
- **修掉一个会静默毁数据的工具缺陷**：`tools/_e45_verify_round50.py` 的内嵌写回 `upsert()` 旧实现把文件从**自己的标记**切到**下一个 `### ` 标题**为止 ⇒ **夹在中间的更新一轮的写回块会被无声删除**（本轮写回两次成功后即被它抹掉，5 个目标里 4 个被清空）。已改为「块文本已存在则原样不动」＋「就地替换保留位置」＋「旧块边界按外来边界判定」。修后重跑第 50 轮仍 `VERDICT: PASS (POS 6/6)`，第 51 轮写回块未被抹除（5 目标 r51 标记计数均 == 1）。**教训**：**做「从标记切到下一标题」式删除的写回函数会杀掉并行新增的块**。
- **边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）；窗级「贡献」即便局部超出噪声，也**不能**当作跨种子证据。

"""

AUTOMATION_BLOCK = f"""
### 2026-11-14｜第 51 轮（误差构成轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`）。**

**远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；队列空；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ 无新活动。远端 13 份 json 与本地 `tools/_evals/e45/` md5 **13/13 全同**。

**闸门**：既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）＋逐轮复核（`_e45_verify_round43…50`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。

**新证据**——`tools/_e45_verify_round51.py`（只读本地，**POS 7/7 VERDICT: PASS**，产物 `tools/_evals/e45/round51_error_composition.json`）：前 50 轮判过七层，**从未把 ΔCSI 拆成 POD 贡献与 FAR 贡献** ⇒ 本轮首次在**误差构成轴**判读，**裁决零变化**，得四条新事实：① **代数先修**——`CSI ≠ POD·(1−FAR)`，正确恒等式 `CSI = POD(1−FAR)/(1−FAR+FAR·POD)`；首版误用被自带恒等式断言当场抓住（dev 0.064），改用**两因子 Shapley 拆**后逐格 max|dev| < 1e−12。② **增益几乎全来自 POD 一侧**：`r(POD 贡献, ΔCSI) = +0.9141`、`r(FAR 贡献, ΔCSI) = −0.1769`（跨 18 格）⇒ 不是少报虚警换来的，是多命中换来的。③ **零模型地板**（B0 换 seed）ΔCSI 0.038771／POD 贡献 0.046590／FAR 贡献 0.012257／ΔF 真值 0.333289；**A4b 是唯一三项全过地板的臂**（POD 中位 +0.08631、ΔF/真值 +2.08925），A3a 0/3、A3b 1/3、A2 全 0/3。④ **跨仪器交叉核对**（组成轴＝列联表计数；S3＝场顺序统计的全局标量配平）：**`r(ΔF/真值, 等偏置掉掉的 CSI) = +0.6415`**、**`r(配平前偏置比−1, 掉掉的 CSI) = +0.6420`** ⇒ S3 裁决在另一台仪器上独立复现；S3 的 12 个点估重算 max|diff| = 0。**A4b 逐 seed 原始增益 +0.01575／+0.04825／−0.11530、配平后 −0.00984／+0.00371／−0.06620 ⇒ 符号两个 seed 翻负 ⇒ 仍是臂不稳**。

**本轮修掉一个会静默毁数据的工具缺陷**：`tools/_e45_verify_round50.py` 的内嵌写回 `upsert()` 旧实现把文件从**自己的标记**一路切到**下一个 `### ` 标题**为止 ⇒ **夹在中间的、更新一轮的写回块会被无声删除**（本轮写回连续两次成功、随即被第 50 轮脚本抹掉，5 个目标里 4 个被清空）。已改为：① 块文本已存在则**原样不动**（`already-present`）；② 只在标记存在时**就地替换**旧块、保留其位置，旧块边界改按**外来边界**（其他轮的标题／项目符号）判定，**不再用「下一个 `### `」做无差别切割**。修后重跑第 50 轮脚本仍 `VERDICT: PASS (POS 6/6)`，第 51 轮的写回块**未被抹除**（5 个目标 r51 标记计数均 == 1，且 r50 块仍在、顺序仍为 r50 → r51）。**教训（可复用）**：**任何做「从标记切到下一标题」式删除的写回函数，都会杀掉并行新增的块**；写回必须改成「就地替换」或「文本已存在则跳过」，否则每轮新证据都会被上一轮的脚本悄悄吃掉。

**写回**：`HANDOVER_E45.md` §4⑮ ＋ 上层 MEMORY.md 状态段 ＋ 2026-11-14 逐日日志（上层 ＋ 项目级）＋ 本文件（脚本 `tools/_e45_writeback_round51.py`，幂等，标记＝插入文本原文子串，连跑两次第二遍全部 `unchanged`，各文件标记计数 == 1）。

**本轮踩坑（可复用规则）**：① **`CSI ≠ POD·(1−FAR)`**——正确式是 `CSI = POD(1−FAR)/(1−FAR+FAR·POD)`；本轮首版用了错的式子，**靠自带代数恒等式断言当场抓住 0.064 的偏差**（⇒ 任何「把一个指标拆成两项贡献」的分解都必须先写恒等式断言）。② **Shapley 拆才无路径依赖**——两因子分解要取两条路径的平均，否则拆分的和与原量对不上。③ **`s3_equalbias.json` 的 `achieved_bias` 是配平后的偏置**，其与 `control_bias` 的比值按构造恒为 1（实测偏离 8.06e−07）⇒ **永远不能当协变量做相关**；配平前的偏置比必须由窗表自算。④ **断言要写实际次数**——首版把「重建比较次数」写成 84，实际是 168（21 格 × 8 个已发布指标）⇒ 空/错断言会空洞地报 FAIL 或 PASS。
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
