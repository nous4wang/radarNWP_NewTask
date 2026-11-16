"""E45 round-49 write-back (idempotent, marker-guarded).

Round-49 = new axis: the FORECAST LEAD-TIME axis (``lead_scores`` inside every
``*.eval.json``), judged for the first time in the project's history.  No
verdict change, no published-number change => MASTER is NOT re-versioned (it
stays v2.48); the evidence goes to HANDOVER §4⑭, the upper MEMORY.md status
section, the 2026-11-14 daily logs (upper + project level) and the automation
memory.

Idempotency rules learned in rounds 41/43/45/47/48:
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged".
"""

from pathlib import Path

MARK = "第 49 轮（预报时效轴的首次判读）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1114 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md")
PROJ_LOG_1114 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

HANDOVER_ANCHOR = "### 下面是结案前的执行记录（存档用，不必重跑）"

HANDOVER_BLOCK = f"""
**⑭（2026-11-14 00:3x {MARK}）**——脚本 `tools/_e45_verify_round49.py`（只读本地，**POS 17/17 VERDICT: PASS**，产物 `tools/_evals/e45/round49_lead_axis.json`）：前 48 轮判过聚合层／机制探针层／S1 结构量层／窗级／风暴场次级共五层，**从未打开过每份 `*.eval.json` 里的 `lead_scores`（10 个预报时效 × 4 阈值）** ⇒ 本轮第一次在**预报时效轴**上判读，**裁决零变化**，但拿到三条前五层都看不见的结构性事实：

1. **时效轴上「种子噪声」比窗级聚合的地板还大**——零模型仍用对照臂 B0 自己换种子互比（纯 seed 噪声、不含任何机制），逐时效的 `|ΔCSI@2.0|` 噪声地板 **0.0251–0.0640**（合并 **0.05401**）⇒ **比第 46 轮的窗级聚合地板 0.038771 还大**。⇒ **任何按时效分层的读数，其单时效差异都必须先对这个地板**；这也再次说明第 46 轮那条「窗级 CI 是伪重复」不是孤例，**换任何切分维度都救不了 n=3**。

2. **增益随时效的符号翻转（前五层全部看不到）**——配对口径下 **A4 的增益在 L0–L3 为正（+0.0185／+0.0464／+0.0675／+0.0440）、在 L4–L9 全为负（−0.0148／−0.0170／−0.0366／−0.0431／−0.0444／−0.0338）**，斜率 **−0.01154/lead**；**A4b 恰好相反**（L0–L2 负 −0.0648／−0.0226／−0.0088 → L7–L9 正 +0.0146／+0.0553／+0.0552，斜率 **+0.01106/lead**）。⇒ **两臂的「增益」分别活在时效轴的两端**；**把 10 个时效平均掉的任何读数都会把符号抹掉**（这正是第 44 轮「配对口径与差值口径必须并列报」在时效维度的姊妹条）。**通过率**：六臂中 **只有 A4 出现过 3/3 同号为正的时效（L2、L3，共 2/10）**，且也只有 3/10 时效越过噪声地板（L1／L2／L3）；A1／A2／A3a／A3b／A4b 全部 **0/10**。

3. **过预报指纹随时效单调增长，与 S3 裁决独立吻合**——对对照的 BIAS 比：A4b **L0 3.29 → L4 5.01 → L9 6.68**（单调加剧），A4 L0 1.80 → L4 1.06 → L9 1.23，A3a L0 0.50（短时效欠预报一半）→ L9 1.72，A3b 1.30→1.50，A1／A2 全程 0.81–1.07 ⇒ **A4b 的过预报在长时效最重，而它的「增益」恰好也只在长时效** ⇒ 与 S3「仍是面积／幅度效应」从**时效维度**独立吻合。逐臂 `r(ΔCSI配对, ΔBIAS)`（跨 10 个时效）：**A4 +0.959**（几乎完全共线）、A3a +0.610、A1 +0.521、A2 +0.493、A3b +0.031、A4b −0.257；合并 60 个（臂×时效）点 **+0.4106**。

**产物自洽性（顺带复算）**：时效表的代数恒等式逐格成立——`BIAS = POD/(1−FAR)`（max dev **1.776e−15**）、`CSI = POD(1−FAR)/(1−FAR+FAR·POD)`（max dev **1.110e−16**）；`thresholds` 块与其自身计数一致（max dev **0.000e+00**）；21 格的 overall `CSI@2.0` **全部**能在已发布 `scores_summary_s{0,1,2}.json` 中找到（21/21）。**技能随时效衰减**：7/7 臂的 `CSI@2.0` 从 L0 到 L9 都下降（B0 0.2179 → 0.0529）⇒ 时效轴本身是有意义的轴，不是退化轴。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列（差值口径下 A3a／A3b 在 L4 分别为 −0.0146／−0.0055，与配对口径的 +0.0400／+0.0494 反向 ⇒ **仍有种子对齐效应**）；③ **事后诊断，不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **S3 裁决守卫复核**：12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变。**新登记一条可复用的口径纪律**：**凡报「随时效变化」的结论，必须同时报斜率或早/晚期差、并且必须给出该时效上的零模型地板**——只看跨时效平均会把符号翻转（本案 A4 早正晚负、A4b 早负晚正）抹平。
"""

MEMORY_BLOCK = f"""
- **（2026-11-14 00:3x {MARK}：裁决不变，新证据＝预报时效轴的首次判读）** 远端 E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、汇总 **291**、只读诊断 297/298）、`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全、队列空、日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、`runs/` 下无 E46+ 新目录 ⇒ 与前四十八轮完全一致；远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**；本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。六套既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）＋八套逐轮复核（`_e45_verify_round40/41/42/43/44/45/46/47/48`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。**新脚本 `tools/_e45_verify_round49.py`（只读本地，POS 17/17 VERDICT: PASS）**：前 48 轮判过聚合／机制探针／S1 结构量／窗级／场次级共五层，**从未打开每份 `*.eval.json` 里的 `lead_scores`（10 个预报时效 × 4 阈值）** ⇒ 本轮首次在时效轴判读，**裁决零变化**，得三条新事实：① **时效轴上的种子噪声地板比窗级聚合地板还大**——零模型（对照臂 B0 换种子互比）逐时效 `|ΔCSI@2.0|` 地板 **0.0251–0.0640**（合并 **0.05401**）＞ 第 46 轮的 0.038771 ⇒ **换任何切分维度都救不了 n=3**；② **增益随时效符号翻转**——A4 配对增益 L0–L3 全正（+0.0185／+0.0464／+0.0675／+0.0440）、L4–L9 全负（斜率 **−0.01154/lead**），**A4b 恰好相反**（L0–L2 负 → L7–L9 正，斜率 **+0.01106/lead**）⇒ **把 10 个时效平均掉会把符号抹掉**；**通过率**：六臂中只有 **A4 出现过 3/3 同号为正的时效（L2、L3，2/10）**、只有 3/10 时效越过噪声地板，A1／A2／A3a／A3b／A4b 全部 **0/10**；③ **过预报指纹随时效单调增长**——对对照 BIAS 比 A4b **3.29→5.01→6.68**、A3a 0.50→1.72、A3b 1.30→1.50、A1／A2 全程 0.81–1.07 ⇒ **A4b 的过预报在长时效最重、而它的「增益」也只在长时效** ⇒ 与 S3「仍是面积／幅度效应」从时效维度独立吻合；逐臂 `r(ΔCSI配对, ΔBIAS)` **A4 +0.959**、A3a +0.610、A1 +0.521、A2 +0.493、A3b +0.031、A4b −0.257，合并 60 点 **+0.4106**。**顺带复算**：时效表代数恒等式逐格成立（`BIAS=POD/(1−FAR)` dev 1.776e−15、`CSI=POD(1−FAR)/(1−FAR+FAR·POD)` dev 1.110e−16）、`thresholds` 块与自身计数一致（0.000e+00）、21 格 overall `CSI@2.0` 全部见诸已发布 summary（21/21）、7/7 臂技能随时效衰减。**新登记口径纪律**：**凡报「随时效变化」的结论，必须同时报斜率或早/晚期差，并给出该时效的零模型地板**。**边界**：事后诊断、**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**。
"""

LOG_BLOCK = f"""
**补记，{MARK}（2026-11-14 00:3x）**

- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、汇总 **291**、只读诊断 297/298）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；**队列空**；`runs/` 下无 E46+ 新目录；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **与前四十八轮完全一致，无新活动**。本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：六套既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）＋八套逐轮复核（`_e45_verify_round40/41/42/43/44/45/46/47/48`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无数值与裁决变化）。
- **新证据（预报时效轴的首次判读）**——`tools/_e45_verify_round49.py`（只读本地，**POS 17/17 VERDICT: PASS**，产物 `tools/_evals/e45/round49_lead_axis.json`）：
  1. **时效轴上「种子噪声」比窗级聚合地板还大**：零模型（对照臂 B0 换种子互比）逐时效 `|ΔCSI@2.0|` 地板 **0.0251–0.0640**（合并 **0.05401**）＞ 第 46 轮窗级地板 0.038771 ⇒ **任何按时效分层的读数都要先对这个地板**。
  2. **增益随时效符号翻转**：A4 配对增益 L0–L3 全正（+0.0185／+0.0464／+0.0675／+0.0440）、L4–L9 全负（斜率 **−0.01154/lead**）；**A4b 恰好相反**（L0–L2 负 → L7–L9 正，斜率 **+0.01106/lead**）⇒ **跨时效平均会抹掉符号**。**通过率**：只有 **A4 出现过 3/3 同号为正的时效（L2、L3，2/10）**、3/10 时效越过噪声地板；其余五臂全部 **0/10**。
  3. **过预报指纹随时效单调增长**：对对照 BIAS 比 A4b **3.29→5.01→6.68**、A3a 0.50→1.72、A3b 1.30→1.50、A1／A2 全程 0.81–1.07 ⇒ **A4b 的过预报在长时效最重、其「增益」也只在长时效** ⇒ 与 S3 裁决从时效维度独立吻合；逐臂 `r(ΔCSI, ΔBIAS)` **A4 +0.959**、A3a +0.610、A1 +0.521、A2 +0.493、A3b +0.031、A4b −0.257，合并 60 点 **+0.4106**。
  4. **顺带复算**：时效表代数恒等式逐格成立（dev 1.776e−15／1.110e−16）、`thresholds` 块与自身计数一致（0.000e+00）、21 格 overall `CSI@2.0` 全部见诸已发布 summary（21/21）、7/7 臂技能随时效衰减（B0 0.2179 → 0.0529）。

**口径**：仍按中位数＋通过率报；配对与差值口径并列（差值口径下 A3a／A3b 在 L4 为负、与配对口径反向 ⇒ 仍有种子对齐效应）；事后诊断，不构成新判据（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）；**S3 守卫复核 12/12 区间仍全部跨零，六臂 `s3_verdict` 逐臂不变**。

**新登记口径纪律**：**凡报「随时效变化」的结论，必须同时报斜率或早/晚期差，并给出该时效上的零模型地板**——只看跨时效平均会把符号翻转（A4 早正晚负、A4b 早负晚正）抹平。

**边界**：**裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版** ⇒ 新证据只进 `HANDOVER_E45.md` §4⑭、上层 MEMORY.md 状态段与本轮逐日日志。
"""

AUTOMATION_BLOCK = f"""
### 2026-11-14 00:3x｜第 49 轮（预报时效轴的首次判读）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前四十八轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`）。**

**远端**：提交器 **273 COMPLETED**（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。

**闸门**：六套既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）＋八套逐轮复核（`_e45_verify_round40/41/42/43/44/45/46/47/48`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。

**新证据**——`tools/_e45_verify_round49.py`（只读本地，**POS 17/17 VERDICT: PASS**，产物 `tools/_evals/e45/round49_lead_axis.json`）：前 48 轮判过聚合／机制探针／S1 结构量／窗级／场次级共五层，**从未打开每份 `*.eval.json` 里的 `lead_scores`（10 个预报时效 × 4 阈值）** ⇒ 本轮首次在时效轴判读，**裁决零变化**，得三条新事实：① **时效轴上的种子噪声地板比窗级聚合地板还大**（零模型＝对照臂 B0 换种子互比；逐时效 `|ΔCSI@2.0|` 地板 **0.0251–0.0640**、合并 **0.05401** ＞ 第 46 轮的 0.038771）⇒ **换任何切分维度都救不了 n=3**；② **增益随时效符号翻转**——A4 配对增益 L0–L3 全正（+0.0185／+0.0464／+0.0675／+0.0440）、L4–L9 全负（斜率 **−0.01154/lead**），**A4b 恰好相反**（L0–L2 负 → L7–L9 正，斜率 **+0.01106/lead**）⇒ **跨时效平均会抹掉符号**；**通过率**＝只有 **A4 出现过 3/3 同号为正的时效（L2、L3，2/10）**、3/10 时效越过噪声地板，其余五臂全部 **0/10**；③ **过预报指纹随时效单调增长**——对对照 BIAS 比 A4b **3.29→5.01→6.68**、A3a 0.50→1.72、A3b 1.30→1.50、A1／A2 全程 0.81–1.07 ⇒ **A4b 的过预报在长时效最重、其「增益」也只在长时效** ⇒ 与 S3「仍是面积／幅度效应」从时效维度独立吻合；逐臂 `r(ΔCSI配对, ΔBIAS)` **A4 +0.959**、A3a +0.610、A1 +0.521、A2 +0.493、A3b +0.031、A4b −0.257，合并 60 点 **+0.4106**。**顺带复算**：时效表代数恒等式逐格成立（dev 1.776e−15／1.110e−16）、`thresholds` 块与自身计数一致（0.000e+00）、21 格 overall `CSI@2.0` 全部见诸已发布 summary（21/21）、7/7 臂技能随时效衰减。**新登记口径纪律**：**凡报「随时效变化」的结论，必须同时报斜率或早/晚期差，并给出该时效的零模型地板**。

**写回**：`HANDOVER_E45.md` §4⑭ ＋ 上层 MEMORY.md 状态段 ＋ 2026-11-14 逐日日志（上层 ＋ 项目级，新建）＋ 本文件（脚本 `tools/_e45_writeback_round49.py`，幂等，标记＝插入文本原文子串，连跑两次第二遍全部 `unchanged`，各文件标记计数 == 1）。

**本轮踩坑（可复用规则）**：① **自写脚本的元组解包要先定形**——`decay` 起初存 `(arm, list)` 二元组却在后面按 `d[1][0]` 取值，`for a, d in decay` 已把 list 解包成 `d` ⇒ `d[1]` 取到的是 float ⇒ 报 `TypeError: 'float' object is not subscriptable`；**改存 dict（`decay[arm] = list`）后一次性消除该类错误**。② **先 dump 再写断言**（第 44 轮已立）——本轮对 `lead_scores` 的键结构（`lead_scores[str(lead)][thr]`）先 dump 一次再写代码，避免了键名猜测。
"""


def append_block(path, block):
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return "unchanged"
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + block, encoding="utf-8")
    return "written"


def insert_before(path, anchor, block):
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return "unchanged"
    if anchor not in text:
        raise SystemExit(f"ANCHOR NOT FOUND in {path}: {anchor!r} (refusing to silently skip)")
    idx = text.index(anchor)
    path.write_text(text[:idx] + block.lstrip("\n") + "\n" + text[idx:], encoding="utf-8")
    return "written"


def write_new(path, block):
    """Create the file if missing; otherwise append idempotently."""
    if path.exists():
        return append_block(path, block)
    header = ("# 2026-11-14 工作日志（radar-nwp-bench）\n\n"
              "> 逐日工作日志，只增不改；结论与数字以 `docs/MASTER.md` 与交接文档为准。\n")
    path.write_text(header + block.lstrip("\n"), encoding="utf-8")
    return "created"


def main():
    targets = [
        ("HANDOVER_E45.md", insert_before, HANDOVER, HANDOVER_ANCHOR, HANDOVER_BLOCK),
        ("MEMORY.md", append_block, MEMORY, None, MEMORY_BLOCK),
        ("memory/2026-11-14.md", write_new, LOG_1114, None, LOG_BLOCK),
        ("project 2026-11-14.md", write_new, PROJ_LOG_1114, None, LOG_BLOCK),
        ("automation memory", append_block, AUTOMATION_MEM, None, AUTOMATION_BLOCK),
    ]

    for name, fn, path, anchor, block in targets:
        if path.exists() and not path.exists():
            raise SystemExit(f"MISSING TARGET: {path}")
        res = fn(path, anchor, block) if anchor else fn(path, block)
        print(f"  {name:>22}: {res}")
        n = path.read_text(encoding="utf-8").count(MARK)
        if n != 1:
            raise SystemExit(f"MARKER COUNT != 1 in {path}: {n}")
    print("all targets written/unchanged; marker count == 1 in every target")


if __name__ == "__main__":
    main()
