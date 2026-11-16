"""E45 round-55 write-back (idempotent, marker-guarded).

Round-55 = new axis: CALIBRATION OF THE S3 GUARD ACROSS THE E41->E45 LINEAGE.
All 54 previous rounds judged E45's CONTENT (mechanism read-outs, S1 struct
metrics, window level, storm-case level, lead-time axis, severity/difficulty
strata, error composition, omnibus arm effect, mechanism dose-response, metric
families).  None ever looked at the GUARD itself.

The guard (`tools/_e41_s3_equalbias.py::two_level_bootstrap`) builds a
TWO-LEVEL interval: it resamples SEEDS, then windows inside each drawn seed.
The published lineage was NOT run at a uniform seed-replication depth:
E41=3, E42=3, E43=2, E44=1, E45=3.  At n_seeds=1 the seed level is DEGENERATE
(the draw `rng.integers(0,1,1)` is always [0]), so E44's intervals are
WINDOW-ONLY -- exactly the pseudo-replication shape round 46 flagged.

This round:
  * reproduces all 12 published S3 point estimates exactly from local artifacts;
  * re-implements the guard in pure stdlib and reproduces the published raw
    intervals (crossing decisions 6/6, width ratios 0.985-1.007);
  * applies the depth-1 (E44-style) protocol to E45's own 18 (arm,seed) cells
    -> 17/18 "significant", 10/18 declared GAIN;
  * calibrates both protocols on a MECHANISM-FREE null (B0 seed-derangement)
    -> depth-1 false-positive rate 1.000, depth-3 0.000;
  * censuses the lineage and tabulates which published verdicts are
    calibration-sensitive.

No verdict change, no published-number change => MASTER is NOT re-versioned
(it stays v2.48); the evidence goes to HANDOVER section 4, the upper MEMORY.md
status section, the 2026-11-14 daily logs (upper + project level) and the
automation memory.

Idempotency rules learned in rounds 41/43/45/47/48/49/51/52/54:
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged";
  * never slice a file from "my marker" to "the next heading";
  * NOTE (round-54 lesson): the marker must NOT also be a substring that the
    round-55 VERIFICATION script prints, and this write-back must never be
    invoked from a verification script (round 53's embedded write-back caused
    triple-appends).
"""

from pathlib import Path

MARK = "第 55 轮（S3 护栏标定轴的首次判读）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1114 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
PROJ_LOG_1114 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

HANDOVER_ANCHOR = "### 下面是结案前的执行记录（存档用，不必重跑）"

HANDOVER_BLOCK = """
**⑲（2026-11-14 @@MARK@@）**——脚本 `tools/_e45_verify_round55.py`（只读本地，**POS 9/9 VERDICT: PASS**，产物 `tools/_evals/e45/round55_guard_calibration.json`）：前 54 轮的**全部**判读都只针对 E45 的**内容**（机制读数／S1 结构量／窗级／场次级／时效轴／强度·难度轴／误差构成／臂间合并检验／剂量-反应／指标族），**从未审过「护栏本身」**。本轮第一次审**判据工具的标定性**，得五条新事实：

1. **护栏是「两层自助」：先重抽 seed、再在抽中的 seed 内重抽窗**（`tools/_e41_s3_equalbias.py::two_level_bootstrap`：`si = rng.integers(0, n_seeds, n_seeds)` → `for k in si: idx = rng.integers(0, n_win, n_win)` → `diffs[i] = np.mean(obs)`）。⇒ **区间宽度直接由「种子复制深度」决定**，而**已发布的整条血脉并没有在统一深度上跑过**：

   | 实验 | `n_seeds` | 种子层是否退化 | 中位 raw CI 宽度 |
   |---|---|---|---|
   | E41（fixed） | 3 | 否 | 0.022459 |
   | E42（fixed） | 3 | 否 | 0.001114 |
   | E43（fixed） | 2 | 否 | 0.006915 |
   | **E44** | **1** | **是** | 0.006539 |
   | E45 | 3 | 否 | 0.070882 |

   ⇒ **`n_seeds=1` 时种子层完全退化**（`rng.integers(0,1,1)` 恒为 `[0]`），E44 的区间退化成**纯窗级区间**——**正是第 46 轮已点名「窗级 CI 是伪重复」的那一种形状**。⇒ **「跨零与否」在血脉内不可通约**：E44 的「显著」与 E45 的「无显著」是在**不同复制深度**上取到的。

2. **E44 式 depth-1 协议套到 E45 自己的 18 个（臂×seed）格上，会「制造」出显著性**：**17/18 格判显著，其中 10 格点估为正（会被判成「获得增益」）**，六臂**每一臂都至少有 1 格被判显著、至少 1 格被判成增益**（A1 s1 +0.0660／A2 s2 +0.0037／A3a s0 +0.0277／A3b s0 +0.0478／A4 s0 +0.0537／A4b s1 +0.0483）⇒ **同一个协议、同一批模型，换个深度结论就翻**。

3. **决定性的一条：拿「无机制」的零假设去标定两个协议**。零假设用 E45 **自己的对照臂换种子**（`B0_s{k}` vs `B0_s{π(k)}`，π 为种子错排 ⇒ 只含种子噪声、不含任何机制）：**depth-1 协议的假阳性率 = 6/6 = 1.000；depth-3 协议 = 0/2 = 0.000**。按 E44 自己的深度再问一次（B0 vs B0、每 seed 一个窗级区间）：**3/3 全部判「显著」**。⇒ **depth-1 协议在「同一臂换种子」这种最平凡的零假设上都 100% 报显著 ⇒ 它报出的「显著」不含任何机制信息**；而 **E45 用的 depth-3 协议在同一个零假设上完全校准**。⇒ **血脉里「E44 有显著、E45 没有」的那条叙事线，混淆的是护栏的复制深度，不是机制**。⚠️ 边界：本条**不重新裁决 E41/E42/E43/E44**（它们已结案、已封口），只说明**那些「显著」标签的强度受护栏深度限制**。

4. **深度 3 的区间宽，是护栏在「正确地」计入种子层，不是缺陷**。逐臂 depth3 CI 宽度 ÷ 同臂三个 depth-1（纯窗级）宽度的均值：**A1 10.14×、A4b 9.73×、A3b 7.65×、A4 5.48×、A3a 5.43×、A2 2.95×**（中位 **5.46×**）⇒ 与第 48 轮「seed 层离散度是窗级 CI 半宽的 9.1～30.8 倍」**同族但更直接**（本轮是同臂同数据、只换层数的直接比值）。⇒ **E45 之所以「什么都没判出来」，部分原因是它的区间诚实地把种子层算进去了**；E44 的区间之所以「什么都显著」，部分原因是它压根没算种子层。

5. **14 个已发布臂级裁决里有 4 个是「标定敏感」的——它们的裁决由「配平后区间的跨零判定」而不是由点估承载**：**E41/A1、E43/A3、E45/A1、E45/A2**（若配平后区间改为不跨零，E45/A2 会从 `NO_GAIN_TO_BEGIN_WITH` 翻成 `GAIN_SURVIVES_EQUALISATION`）。⇒ 新登记一条口径纪律：**引用任何 S3 裁决时必须同时报该臂的 `n_seeds` 与区间宽度**，否则「显著」这个词在不同深度间不可比。
   ⇒ 同时得到一个**血脉级的正面结论**：**整条 E41→E45 血脉（14 个臂）中，到达 `GAIN_SURVIVES_EQUALISATION` 的有 0 个**（E41 2 臂全 `GAIN_WAS_AREA_OR_AMPLITUDE`；E42 1 臂 `NO_GAIN_TO_BEGIN_WITH`；E43 1 臂 `GAIN_WAS_AREA_OR_AMPLITUDE`；E44 4 臂 1 `GAIN_WAS…`＋3 `NO_GAIN…`；E45 6 臂 4 `GAIN_WAS…`＋2 `NO_GAIN…`）⇒ **「配平后仍存活」这条路在整条血脉上从未被任何一臂走到过**——这与 E45 的六臂判负**方向一致、互为印证**，且是**独立于 E45 的**证据。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ **事后诊断，不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；⑤ **重建自校**：**12 个已发布 S3 点估由本地产物逐位复现，max|diff| = 0.000e+00**；stdlib 重实现的两层自助与已发布 raw 区间**跨零判定 6/6 一致、宽度比 0.985～1.007**（不同 RNG 流 ⇒ 仅差蒙特卡洛误差）；⑥ **不重新裁决 E41/E42/E43/E44**，只披露护栏深度对其「显著」标签的限制；⑦ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
"""

MEMORY_BLOCK = """
- **（2026-11-14 @@MARK@@：裁决不变，新证据＝S3 护栏本身的标定性首次判读）** 远端 E45 状态与前五十四轮**完全一致**（提交器 **273** `COMPLETED` 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291**、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全、**队列空**、日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、`runs/` 下无 E46+ 新目录 ⇒ **无新活动**；远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**）；本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。**新脚本 `tools/_e45_verify_round55.py`（只读本地，POS 9/9 VERDICT: PASS，产物 `tools/_evals/e45/round55_guard_calibration.json`）**：前 54 轮只判 E45 的**内容**，**从未审过护栏本身** ⇒ 本轮首次审**判据工具的标定性**，**裁决零变化**，得五条新事实：① **护栏是两层自助（先重抽 seed、再在 seed 内重抽窗），区间宽度由种子复制深度决定，而已发布血脉的深度并不统一**：E41=3／E42=3／E43=2／**E44=1（种子层完全退化 ⇒ 区间退化为纯窗级，正是第 46 轮点名的伪重复形状）**／E45=3（中位 raw CI 宽度 0.001114～0.070882）⇒ **「跨零与否」在血脉内不可通约**。② **E44 式 depth-1 协议套到 E45 自己的 18 个（臂×seed）格上，17/18 判显著、10/18 会被判成增益，六臂每臂都至少 1 格被判显著** ⇒ 换深度即翻结论。③ **决定性**：用 **E45 自己的对照臂换种子**（`B0_s{k}` vs `B0_s{π(k)}`，纯种子噪声、无机制）标定两个协议 ⇒ **depth-1 假阳性率 1.000（6/6）、depth-3 假阳性率 0.000（0/2）**；按 E44 自己的深度再问（B0 vs B0）**3/3 全判显著** ⇒ **depth-1 报出的「显著」不含机制信息，E45 的 depth-3 在同一零假设上完全校准** ⇒ **血脉里「E44 有显著、E45 没有」混淆的是护栏深度、不是机制**（⚠️ **不重新裁决 E41/E42/E43/E44**，只披露深度对其「显著」标签的限制）。④ **depth3 区间宽是护栏在正确计入种子层**：逐臂 depth3/depth1 宽度比 **A1 10.14×／A4b 9.73×／A3b 7.65×／A4 5.48×／A3a 5.43×／A2 2.95×**（中位 5.46×）⇒ E45「什么都没判出来」部分因为区间诚实计入了种子层，E44「什么都显著」部分因为压根没计。⑤ **14 个已发布臂级裁决中 4 个是「标定敏感」的**（E41/A1、E43/A3、E45/A1、E45/A2——裁决由「配平后区间的跨零判定」而非点估承载）⇒ 新登记纪律：**引用任何 S3 裁决必须同时报该臂的 `n_seeds` 与区间宽度**；同时得到**血脉级正面结论**：**整条 E41→E45 血脉 14 个臂中到达 `GAIN_SURVIVES_EQUALISATION` 的有 0 个** ⇒ 「配平后仍存活」这条路在整条血脉上从未被任何一臂走到，**独立于 E45 印证六臂判负**。**边界**：事后诊断、**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**——**六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版** ⇒ 新证据只进本节、HANDOVER §4⑲、2026-11-14 逐日日志（上层 ＋ 项目级）＋ 本文件。
"""

LOG_BLOCK = """
**补记，@@MARK@@（2026-11-14）**

- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291**、只读诊断 297/298）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；**队列空**；`runs/` 下无 E46+ 新目录；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **与前五十四轮完全一致，无新活动**。本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无数值与裁决变化）。
- **新证据（S3 护栏本身的标定性首次判读）**——`tools/_e45_verify_round55.py`（只读本地，**POS 9/9 VERDICT: PASS**，产物 `tools/_evals/e45/round55_guard_calibration.json`）：前 54 轮只判 E45 的**内容**，**从未审过护栏本身** ⇒ 本轮首次审**判据工具的标定性**：
  1. **护栏＝两层自助（先重抽 seed、再在 seed 内重抽窗），区间宽度由种子复制深度决定，而已发布血脉深度不统一**：E41=3／E42=3／E43=2／**E44=1（种子层完全退化 ⇒ 区间退化为纯窗级＝第 46 轮点名的伪重复形状）**／E45=3 ⇒ **「跨零与否」在血脉内不可通约**。
  2. **E44 式 depth-1 协议套到 E45 自己的 18 个（臂×seed）格：17/18 判显著、10/18 会被判成增益，六臂每臂至少 1 格被判显著** ⇒ 换深度即翻结论。
  3. **决定性（零假设标定）**：用 **E45 自己的对照臂换种子**（`B0_s{k}` vs `B0_s{π(k)}`，纯种子噪声、无机制）⇒ **depth-1 假阳性率 1.000（6/6）、depth-3 假阳性率 0.000（0/2）**；按 E44 自己的深度（B0 vs B0）**3/3 全判显著** ⇒ **depth-1 的「显著」不含机制信息；E45 的 depth-3 在同一零假设上完全校准** ⇒ **血脉里「E44 有显著、E45 没有」混淆的是护栏深度、不是机制**（⚠️ **不重新裁决 E41/E42/E43/E44**）。
  4. **depth3 区间宽是护栏在正确计入种子层**：逐臂 depth3/depth1 宽度比 A1 10.14×／A4b 9.73×／A3b 7.65×／A4 5.48×／A3a 5.43×／A2 2.95×（中位 5.46×）。
  5. **14 个已发布臂级裁决中 4 个「标定敏感」**（E41/A1、E43/A3、E45/A1、E45/A2——裁决由配平后区间的跨零判定承载）⇒ 新纪律：**引用任何 S3 裁决必须同时报该臂 `n_seeds` 与区间宽度**；**血脉级正面结论：E41→E45 共 14 臂中到达 `GAIN_SURVIVES_EQUALISATION` 的有 0 个** ⇒ 「配平后仍存活」这条路整条血脉从未被任何一臂走到，**独立于 E45 印证六臂判负**。
- **裁决零变化**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空** ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**。
- **重建自校**：**12 个已发布 S3 点估由本地产物逐位复现（max|diff| = 0）**；stdlib 重实现的两层自助与已发布 raw 区间**跨零判定 6/6 一致、宽度比 0.985～1.007**。
- **边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）。
"""

AUTOMATION_BLOCK = """
### 2026-11-14｜第 55 轮（S3 护栏标定轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十四轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`）。**

**远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` md5 **13/13 全同**。

**闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。

**新证据**——`tools/_e45_verify_round55.py`（只读本地，**POS 9/9 VERDICT: PASS**，产物 `tools/_evals/e45/round55_guard_calibration.json`）：前 54 轮只判 E45 的**内容**，**从未审过护栏本身** ⇒ 本轮首次审**判据工具的标定性**，**裁决零变化**。五条新事实：① **护栏＝两层自助（先重抽 seed、再在 seed 内重抽窗），区间宽度由种子复制深度决定，而已发布血脉深度不统一**（E41=3／E42=3／E43=2／**E44=1 ⇒ 种子层完全退化、区间退化为纯窗级＝第 46 轮点名的伪重复形状**／E45=3）⇒ **「跨零与否」在血脉内不可通约**。② **E44 式 depth-1 协议套到 E45 自己的 18 个（臂×seed）格：17/18 判显著、10/18 会被判成增益，六臂每臂至少 1 格被判显著**。③ **决定性（零假设标定）**：用 **E45 自己的对照臂换种子**（`B0_s{k}` vs `B0_s{π(k)}`，纯种子噪声、无机制）⇒ **depth-1 假阳性率 1.000（6/6）、depth-3 假阳性率 0.000（0/2）**；按 E44 自己的深度（B0 vs B0）**3/3 全判显著** ⇒ **depth-1 的「显著」不含机制信息，E45 的 depth-3 完全校准** ⇒ **血脉里「E44 有显著、E45 没有」混淆的是护栏深度、不是机制**（⚠️ **不重新裁决 E41/E42/E43/E44**）。④ **depth3 区间宽是护栏在正确计入种子层**：逐臂 depth3/depth1 宽度比 **A1 10.14×／A4b 9.73×／A3b 7.65×／A4 5.48×／A3a 5.43×／A2 2.95×**（中位 5.46×）。⑤ **14 个已发布臂级裁决中 4 个「标定敏感」**（E41/A1、E43/A3、E45/A1、E45/A2）⇒ 新纪律：**引用任何 S3 裁决必须同时报该臂 `n_seeds` 与区间宽度**；**血脉级正面结论：E41→E45 共 14 臂中到达 `GAIN_SURVIVES_EQUALISATION` 的有 0 个** ⇒ 「配平后仍存活」整条血脉从未被任何一臂走到，**独立于 E45 印证六臂判负**。

**重建自校**：12 个已发布 S3 点估由本地产物逐位复现（max|diff| = 0）；stdlib 重实现的两层自助与已发布 raw 区间跨零判定 6/6 一致、宽度比 0.985～1.007。

**写回**：`HANDOVER_E45.md` §4⑲ ＋ 上层 `MEMORY.md` 状态段 ＋ 2026-11-14 逐日日志（上层 ＋ 项目级）＋ 本文件（脚本 `tools/_e45_writeback_round55.py`，幂等，标记＝插入文本原文子串，连跑两次第二遍全部 `unchanged`，各文件标记计数 == 1）。
"""

def _m(block):
    return block.replace("@@MARK@@", MARK)


TARGETS = [
    ("HANDOVER", HANDOVER, _m(HANDOVER_BLOCK), "insert_before", HANDOVER_ANCHOR),
    ("MEMORY", MEMORY, _m(MEMORY_BLOCK), "append", None),
    ("LOG_1114", LOG_1114, _m(LOG_BLOCK), "append", None),
    ("LOG_1111", LOG_1111, _m(LOG_BLOCK), "append", None),
    ("PROJ_LOG_1114", PROJ_LOG_1114, _m(LOG_BLOCK), "append", None),
    ("AUTOMATION_MEM", AUTOMATION_MEM, _m(AUTOMATION_BLOCK), "append", None),
]


def main():
    changed, unchanged = [], []
    for name, path, block, mode, anchor in TARGETS:
        if not path.exists():
            raise SystemExit("TARGET MISSING: %s (%s)" % (name, path))
        text = path.read_text(encoding="utf-8")
        n_before = text.count(MARK)
        if n_before >= 1:
            unchanged.append((name, n_before))
            continue
        if mode == "insert_before":
            if anchor not in text:
                raise SystemExit("ANCHOR MISSING in %s: %r" % (name, anchor))
            new = text.replace(anchor, block.strip("\n") + "\n\n" + anchor, 1)
        else:
            new = text.rstrip("\n") + "\n" + block.strip("\n") + "\n"
        path.write_text(new, encoding="utf-8")
        check = path.read_text(encoding="utf-8")
        cnt = check.count(MARK)
        if cnt != 1:
            raise SystemExit("MARKER COUNT %d in %s (expected 1)" % (cnt, name))
        changed.append(name)
    for name, n in unchanged:
        print("unchanged  %-14s (marker x%d)" % (name, n))
    for name in changed:
        print("written    %-14s" % name)
    print("changed=%d unchanged=%d" % (len(changed), len(unchanged)))


if __name__ == "__main__":
    main()
