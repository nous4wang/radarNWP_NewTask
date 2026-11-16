"""E45 round-54 write-back (idempotent, marker-guarded).

Round-54 = new axis: the METRIC-FAMILY axis.  All 53 previous rounds judged
E45 on CSI / BIAS / POD / FAR only -- yet E45's own failure mode is exactly
that CSI+BIAS cannot separate "real skill" from "over-forecast that CSI
rewards" (the S3 guard concluded GAIN_WAS_AREA_OR_AMPLITUDE).  The stage-2
artifacts carry four skill families (CSI / ETS / HSS / FSS); ETS and HSS carry
a climatological null and penalise false alarms, FSS is displacement-tolerant.
This round judges all six arms on all four families for the first time, and
builds a contingency-domain counterpart of the S3 equal-bias guard.

No verdict change, no published-number change => MASTER is NOT re-versioned
(it stays v2.48); the evidence goes to HANDOVER section 4, the upper MEMORY.md
status section, the 2026-11-14 daily logs (upper + project level) and the
automation memory.

Idempotency rules learned in rounds 41/43/45/47/48/49/51/52:
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged";
  * never slice a file from "my marker" to "the next heading".
"""

from pathlib import Path

MARK = "第 54 轮（指标族轴的首次判读）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1114 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
PROJ_LOG_1114 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

HANDOVER_ANCHOR = "### 下面是结案前的执行记录（存档用，不必重跑）"

HANDOVER_BLOCK = f"""
**⑱（2026-11-14 第 54 轮 {MARK}）**——脚本 `tools/_e45_verify_round54.py`（只读本地，**POS 6/6 VERDICT: PASS**，产物 `tools/_evals/e45/round54_metric_family.json`）：前 53 轮的**全部**判读都只用了 **CSI／BIAS／POD／FAR** 四个量，而 E45 自己的失效模式恰恰是「**CSI+BIAS 分不开「真本事」与「过预报换来的分数」**」（S3 裁决 `GAIN_WAS_AREA_OR_AMPLITUDE`）⇒ 本轮第一次把 E45 六臂放到**四个指标族**上判读（`CSI` 无随机基准、`ETS` 带气候学随机基准、`HSS` 命中加权且惩罚虚警、`FSS` 邻域位移容错），**并新建了一个 S3 的「列联表域对应物」**，**裁决零变化**（事后诊断，**不构成新判据**：S3 护栏固定 thr=2.0 mm、复制单元固定为 seed），得五条新事实：

1. **判负不是「CSI 选错了」造成的——四个指标族几乎完全共线**。18 个（臂×seed）格的配对增益跨族相关：**csi–ets r=+0.9990（符号一致 18/18）、csi–hss r=+0.9984（18/18）、csi–fss r=+0.9738（17/18）、ets–hss r=+0.9988、ets–fss r=+0.9753、hss–fss r=+0.9792**；逐臂符号表 **A1／A2 四族全负，A3a／A3b／A4／A4b 四族全正**（16 正 8 负）⇒ **换任何一个连续型技巧评分都不会翻案**。**这条是本轮最有价值的否定性结论**：E45 判负**不是**指标选择偏差造成的。

2. **但「赢家」在族之间会换人，而它本来就只是种子抽签**。对 seed 做 bootstrap 重抽（10000 次）的「赢家概率」：**csi／ets／hss 三族完全一致（A4=0.744、A1=0.256、其余 0.000），只有 fss 族不同（A4=0.744、A4b=0.256）** ⇒ 与第 52 轮的「赢家概率」（原始口径 A4 仅 0.524）**同向但更强**：在连续型技巧评分上，A4 有 74% 概率是赢家、A1 26%——**但没有一个族给出超过 0.75 的确定性**，且 FSS 族直接换人 ⇒ **「A4 最好／A4b 最好」依然无跨种子依据，不得择优**。

3. **零模型地板必须按指标族分别计算**（第 52 轮纪律的再次验证）。thr=2.0 mm 下对照臂 B0 换种子互比的 |Δ| 地板：**csi 0.038771／ets 0.036256／hss 0.058711／fss 0.037227** ⇒ **用 CSI 的地板去判 HSS 会把噪声判成信号**（HSS 的地板是 CSI 的 1.51 倍）。逐臂配对中位增益与通过率（**中位数＋通过率，单点分数不作裁决**）：**A1** csi −0.003420（2/3）、ets −0.002968（2/3）、hss −0.004610（2/3）、fss −0.000606（2/3）；**A2** −0.009814／−0.009412／−0.015110／−0.012247（**四族全 0/3**）；**A3a** +0.015529（0/3）／+0.014771（0/3）／+0.024462（0/3）／+0.025583（1/3）；**A3b** +0.024357（1/3）／+0.022771（1/3）／+0.037299（1/3）／+0.039695（2/3）；**A4** +0.013426（1/3）／+0.013631（1/3）／+0.021368（1/3）／+0.041871（2/3）；**A4b** +0.015750（2/3）／+0.009432（2/3）／+0.018466（2/3）／+0.004163（2/3）⇒ **六臂 × 四族 = 24 格中，没有一格能稳定越过本族地板**（通过率最高 2/3，且无任何臂在四族同时 ≥2/3）。

4. **差值口径与配对口径在 FSS 族上第一次反向**。差值口径（两中位数相减）thr=2.0：**A3a −0.011802／−0.012602／−0.020716／+0.006513**、**A3b −0.005762／−0.006438／−0.010618／+0.020624** ⇒ **A3a／A3b 的「正增益」在 CSI/ETS/HSS 三族上于差值口径为负、只在 FSS 族转正** ⇒ 与第 44 轮（种子对齐效应）、第 49 轮（时效维度符号翻转）**同族的第三例：换指标族也能造出符号翻转** ⇒ **新登记一条口径纪律：凡比较族间一致性，配对口径与差值口径必须并列报，且必须声明指标族**。真正在差值口径上高于对照的仍只有 **A4（csi +0.022079／ets +0.020678／hss +0.032162／fss +0.060027）与 A4b（+0.015750／+0.009432／+0.018466／+0.004163）**——正是 POD／BIAS 爆表的两臂。

5. **新建 S3 的「列联表域对应物」，独立复现面积/幅度裁决**。S3 是**场域**做法（对预报场做一次全局标量配平使偏置相等，再算 CSI）；本轮从已发布的窗表出发做**列联表域**的对应物：把臂的**预报事件面积**缩放到对照的面积（保持臂的命中率），再重算各指标族。**三 seed 逐 seed 复算后与已发布 S3 的 12 个点估符号一致 6/6**（A1 +0.001464 vs S3 +0.000179；A2 +0.007809 vs +0.004086；A3a −0.001591 vs −0.000222；A3b −0.006432 vs −0.002844；**A4 −0.020177 vs −0.006161**；**A4b −0.051746 vs −0.024113**），中位 |diff| 0.003655 ⇒ **两台独立仪器给出同一结论：配平面积后增益不但消失、还转负，A4b 最甚**。窗级同法（逐窗配平面积）配以族内零模型地板：**A4 −0.0223196（1/3 过地板）、A4b −0.0246868（2/3）、A3b −0.0071345（0/3）、A3a −0.0049144（0/3）**，而 A1／A2 配平后恰为 0.0000000（0/3）⇒ **「面积/幅度效应」的裁决在第三台仪器上独立复现**（与第 45 轮 S1 结构量、第 51 轮误差构成轴同族）。
   ⚠️ **口径边界**：列联表域配平**只平面积**（不重建场的强度分布），与 S3 的场域标量配平**不是同一个操作**，两者只作**符号级交叉验证**，不构成新的等价性主张。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ **事后诊断，不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变（**A1／A3a／A3b／A4 = GAIN_WAS_AREA_OR_AMPLITUDE；A2／A4b = NO_GAIN_TO_BEGIN_WITH**），**合格臂集合仍为空**；⑤ **重建自校**：由窗表复算已发布 CSI/BIAS/POD/FAR **504 次比较 max|diff| = 0.000e+00**、头条表 56 项复算 max|diff| = 0.000e+00、21 格 `window_keys` 完全一致、**七臂每窗真值事件数 T=h+m 的跨臂离散度为 0.000e+00**（真值域共享 ⇒ 面积配平构造成立）；⑥ **MASTER 不升版**（无数值与裁决变化，仍 v2.48）。
"""

MEMORY_BLOCK = f"""
- **（2026-11-14 第 54 轮 {MARK}：裁决不变，新证据＝指标族轴的首次判读）** 远端 E45 状态与前五十三轮**完全一致**（提交器 **273** `COMPLETED` 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291**、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全、**队列空**、日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、`runs/` 下无 E46+ 新目录 ⇒ **无新活动**；远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**）；本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。**新脚本 `tools/_e45_verify_round54.py`（只读本地，POS 6/6 VERDICT: PASS，产物 `tools/_evals/e45/round54_metric_family.json`）**：前 53 轮的**全部**判读只用 **CSI／BIAS／POD／FAR**，而 E45 的失效模式恰恰是「CSI+BIAS 分不开真本事与过预报」⇒ 本轮首次在**四个指标族**（CSI 无随机基准／ETS 气候学随机基准／HSS 命中加权惩罚虚警／FSS 位移容错）上判读，**并新建 S3 的列联表域对应物**，**裁决零变化**，得五条新事实：① **判负不是「CSI 选错了」造成的**——四族增益跨族相关 **r = +0.9738～+0.9990**、符号一致 17–18/18、逐臂符号表 A1／A2 四族全负、A3a／A3b／A4／A4b 四族全正 ⇒ **换任何连续型技巧评分都不翻案**。② **赢家在族间会换人，且本就是种子抽签**：seed bootstrap 赢家概率 csi／ets／hss 三族一致（**A4=0.744、A1=0.256**），fss 族换人（**A4=0.744、A4b=0.256**）⇒ 与第 52 轮同向但更强，**不得择优**。③ **零模型地板必须按指标族分别算**（thr=2.0：**csi 0.038771／ets 0.036256／hss 0.058711／fss 0.037227**；用 CSI 地板判 HSS 会把噪声判成信号）⇒ **六臂 × 四族 = 24 格无一稳定越过本族地板**（通过率最高 2/3，无任何臂四族同时 ≥2/3）。④ **差值口径与配对口径在 FSS 族上第一次反向**（A3a/A3b 在 CSI/ETS/HSS 差值口径为负、仅 FSS 转正）⇒ **新登记纪律：凡比较族间一致性，配对口径与差值口径必须并列报、且必须声明指标族**；差值口径上真正高于对照的仍只有 A4／A4b（POD/BIAS 爆表两臂）。⑤ **新建 S3 的列联表域对应物独立复现面积/幅度裁决**：把臂的预报事件面积缩放到对照面积后重算，**与已发布 S3 的 12 个点估符号一致 6/6**（A1 +0.001464/A2 +0.007809/A3a −0.001591/A3b −0.006432/**A4 −0.020177**/**A4b −0.051746**，中位 |diff| 0.003655）⇒ **增益不但消失还转负**；窗级同法配族内地板：**A4 −0.0223196（1/3）、A4b −0.0246868（2/3）、A3b −0.0071345（0/3）、A3a −0.0049144（0/3）**，A1／A2 恰为 0.0000000（0/3）⇒ **面积/幅度裁决在第三台仪器上独立复现**（⚠️ 列联表域只平面积、不重建场强度分布，与 S3 场域标量配平**不是同一操作**，只作符号级交叉验证）。**边界**：事后诊断、**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**——**六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**（无数值与裁决变化）⇒ 新证据只进本节、HANDOVER §4⑱、2026-11-14 逐日日志（上层 ＋ 项目级）与本文件。
"""

LOG_BLOCK = f"""
**补记，{MARK}（2026-11-14）**

- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291**、只读诊断 297/298）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；**队列空**；`runs/` 下无 E46+ 新目录；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **与前五十三轮完全一致，无新活动**。本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无数值与裁决变化）。
- **新证据（指标族轴的首次判读）**——`tools/_e45_verify_round54.py`（只读本地，**POS 6/6 VERDICT: PASS**，产物 `tools/_evals/e45/round54_metric_family.json`）：前 53 轮的**全部**判读只用 CSI/BIAS/POD/FAR，而 E45 的失效模式恰恰是「CSI+BIAS 分不开真本事与过预报」⇒ 本轮首次在**四个指标族**（CSI／ETS／HSS／FSS）上判读，**并新建 S3 的列联表域对应物**：
  1. **判负不是「CSI 选错了」**：四族增益跨族相关 **r = +0.9738～+0.9990**、符号一致 17–18/18；逐臂符号表 A1／A2 四族全负，A3a／A3b／A4／A4b 四族全正（16 正 8 负）⇒ **换任何连续型技巧评分都不翻案**。
  2. **赢家在族间会换人，且本就是种子抽签**：seed bootstrap（10000 次）赢家概率 csi／ets／hss 一致（**A4=0.744、A1=0.256**），fss 族换人（**A4=0.744、A4b=0.256**）⇒ 与第 52 轮同向但更强 ⇒ **不得择优**。
  3. **零模型地板必须按指标族分别算**（thr=2.0：**csi 0.038771／ets 0.036256／hss 0.058711／fss 0.037227**）⇒ **24 格（6 臂×4 族）无一稳定越过本族地板**（通过率最高 2/3，无任何臂四族同时 ≥2/3）。
  4. **差值口径与配对口径在 FSS 族第一次反向**（A3a/A3b 仅 FSS 转正）⇒ **新登记纪律：族间一致性比较必须配对口径与差值口径并列报、并声明指标族**。
  5. **S3 列联表域对应物独立复现面积/幅度裁决**：把臂的预报事件面积缩放到对照面积后重算，**与已发布 S3 点估符号一致 6/6**（A1 +0.001464／A2 +0.007809／A3a −0.001591／A3b −0.006432／**A4 −0.020177**／**A4b −0.051746**，中位 |diff| 0.003655）⇒ **增益不但消失还转负**；窗级同法：**A4 −0.0223196（1/3）、A4b −0.0246868（2/3）、A3b −0.0071345（0/3）、A3a −0.0049144（0/3）**，A1／A2 恰为 0.0000000（0/3）。⚠️ 列联表域只平面积、不重建场强度分布，与 S3 场域标量配平不是同一操作，只作符号级交叉验证。
- **裁决零变化**：S3 的 12 个区间仍全部跨零、六臂 `s3_verdict` 逐臂不变（A1／A3a／A3b／A4 = GAIN_WAS_AREA_OR_AMPLITUDE；A2／A4b = NO_GAIN_TO_BEGIN_WITH）、**合格臂集合仍为空** ⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**。
- **重建自校**：由窗表复算已发布 CSI/BIAS/POD/FAR **504 次比较 max|diff| = 0**；头条表 56 项 max|diff| = 0；21 格 `window_keys` 一致；七臂每窗真值事件数跨臂离散度 0。
- **边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）。
"""

AUTOMATION_BLOCK = f"""
### 2026-11-14｜第 54 轮（指标族轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前五十三轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`）。**

**远端**：提交器 273 `COMPLETED`（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐、`runs/e45/` 13 份 json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` md5 **13/13 全同**。

**闸门**：`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。

**新证据**——`tools/_e45_verify_round54.py`（只读本地，**POS 6/6 VERDICT: PASS**，产物 `tools/_evals/e45/round54_metric_family.json`）：前 53 轮的**全部**判读只用 CSI/BIAS/POD/FAR，而 E45 的失效模式恰恰是「CSI+BIAS 分不开真本事与过预报」⇒ 本轮首次在**四个指标族**（CSI／ETS／HSS／FSS）上判读，**并新建 S3 的列联表域对应物**；**裁决零变化**。五条新事实：① **判负不是「CSI 选错了」**——四族增益跨族相关 r = +0.9738～+0.9990、符号一致 17–18/18 ⇒ 换任何连续型技巧评分都不翻案；② **赢家在族间会换人且本就是种子抽签**（csi/ets/hss 一致 A4=0.744／A1=0.256，fss 换人 A4=0.744／A4b=0.256）⇒ 不得择优；③ **零模型地板必须按指标族分别算**（csi 0.038771／ets 0.036256／hss 0.058711／fss 0.037227）⇒ 24 格无一稳定越过本族地板；④ **差值口径与配对口径在 FSS 族第一次反向** ⇒ 新登记「族间一致性比较必须并列两种口径并声明指标族」；⑤ **S3 列联表域对应物独立复现面积/幅度裁决**（与已发布 S3 点估符号一致 6/6，A4 −0.020177／A4b −0.051746，窗级 A4 −0.0223196／A4b −0.0246868）。

**写回**：`HANDOVER_E45.md` §4⑱ ＋ 上层 `MEMORY.md` 状态段 ＋ 2026-11-14 逐日日志（上层 ＋ 项目级）＋ 本文件（脚本 `tools/_e45_writeback_round54.py`，幂等，标记＝插入文本原文子串，连跑两次第二遍全部 `unchanged`，各文件标记计数 == 1）。
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
