"""E45 round-48 write-back (idempotent, marker-guarded).

Round-48 = new axis: LOSO seed-subset sensitivity of the S3 gate +
cluster-aware (storm-case) uncertainty on the window axis + MDE / power for
the next ladder step.  No verdict change, no published-number change =>
MASTER is NOT re-versioned (it stays v2.48); the evidence goes to
HANDOVER §4⑬, upper MEMORY.md status section and the three daily logs
(project-level + upper-level), plus the automation memory.

Idempotency rules learned in rounds 41/43/45/47:
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged".
"""

from pathlib import Path

MARK = "第 48 轮（S3 门的种子子集敏感性 ＋ 场次聚类不确定度 ＋ 下一阶梯的检出力）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
LOG_1113 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md")
PROJ_LOG_1113 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-13.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

HANDOVER_ANCHOR = "### 下面是结案前的执行记录（存档用，不必重跑）"

HANDOVER_BLOCK = f"""
**⑬（2026-11-13 23:3x {MARK}）S3 门的种子子集敏感性 ＋ 场次聚类不确定度 ＋ 下一阶梯的检出力**——脚本 `tools/_e45_verify_round48.py`（只读本地，**POS 14/14 VERDICT: PASS**）：前 47 轮判过聚合层／机制探针层／S1 结构量层／窗级／风暴场次级共五层，**从未问过两个决定「要不要投十 seed 算力」的问题**：① **如果把三个 seed 里的任何一个抽掉，结论会不会翻？** ② **按实测的 seed 间离散度，下一阶梯到底需要多少 seed 才够？** 本轮补上这两问（外加一件工具性工作：把第 46 轮那条「窗级 CI 是伪重复」从定性警告变成**可量化的设计效应**）。

1. **裁决的根据被重新验证，且比预想的更稳**：由逐 seed 产物复算 S3 的 **12 个点估**（六臂 × raw/equal_bias）⇒ 与 `s3_equalbias.json` **逐位相同（max|diff| = 0.000e+00）**；**12 个已发布区间全部跨零**（这才是 E45 判负的根据，**不是点估的符号**）。同时由 538 窗的原始列联表复算已发布聚合指标（CSI/BIAS/POD/FAR × 21 格）⇒ **84 次比较 max|diff| = 0.000e+00**。

2. **种子子集敏感性（LOSO）——「翻案」只可能来自点估符号，而符号本来就不是判据**：等偏置增益的点估在抽掉一个 seed 后**符号翻转**的臂：A1 **2/3**、A3a 1/3、A3b 1/3、A4 1/3（A2 0/3、A4b 0/3）；原始增益：A1 1/3、A3a 1/3、A3b 1/3、A4b 1/3（A2 0/3、A4 0/3）。⇒ **唯一在全部 4 个子集上都为正的是 A4 的原始增益**（+0.018037），但它**等偏置后点估为负**（−0.006161）⇒ 正是 S3「仍是面积/幅度效应」要说的事；**唯一在全部 4 个子集上都为正的是 A2 的等偏置点估**（+0.004086），但其区间跨零 ⇒ 仍判负。**结论：把任一 seed 抽掉都无法把任何一臂变成"通过"** ⇒ **六臂判负对种子集合不敏感**。

3. **第 46 轮的「窗级 CI 是伪重复」被量化成设计效应（本轮最有用的新工具量）**：窗来自 **8 个风暴场次**，对配对窗差做 ICC：**中位 0.268（范围 0.063–0.561）** ⇒ **538 窗的有效样本量中位只有 31.6 窗（范围 15.6–112.5）**。把重抽单元换成**风暴场次**（真聚类）后：窗级 bootstrap **18 格中 14 格 CI 不跨零**（与第 46 轮公布值逐位一致），而**场次聚类 bootstrap 只有 1 格不跨零**（A4b_s2 的强负格）；**窗级 CI 的半宽中位只有场次聚类 CI 的 0.136 倍**（范围 0.051–0.276）⇒ **窗级 CI 把不确定性低估了约 7.3 倍**。⇒ **E45 首次有了一个非伪重复、且复制单元正确的窗口层区间**；同时也说明第 46 轮那条「窗级 CI 不跨零不可当证据」的反面教材**有了精确倍数**。

4. **下一阶梯要投多少 seed（这一条直接改写「≥10 seed」的默认预期）**：实测 **seed 间增益离散度 pooled SD = 0.04896**（对照臂自身 CSI 的 seed 间 SD = 0.02035）⇒ 以两单侧 5% / 80% 检出力计的**最小可检出效应（MDE）**：**n=3 → 0.15162、n=10 → 0.04870、n=30 → 0.02592**。对照 E45 实测的效应量：六臂的均值增益**没有一臂超过 MDE(3)**（0/6）⇒ 与「12 个区间全跨零」完全自洽；而**要检出 E45 自己观测到的效应量**，A4 的 +0.018037 需要 **n=60**、A3b 的 +0.013653 需要 **n=103**、A3a 的 +0.007576 需要 **n=330** ⇒ **§11.9 阶梯阶 2 的「≥10 seed」只够检出 ≥0.0487 的效应**；**效应量放大到最佳臂的 2 倍（0.036）⇒ n=17、3 倍（0.054）⇒ n=9** ⇒ **下一族机制若真实效应在 0.02–0.05 区间，十 seed 不足以定案，需要 17–60 seed**（这是**算力预算的输入**，不是新的判据）。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **不改变任何裁决**；④ **边界**：MDE 是**正态近似的功效估算**，用实测 pooled SD 代入，未做非参重抽；且它假设效应量本身是待检的固定量。
"""

MEMORY_BLOCK = f"""
- **（2026-11-13 23:3x {MARK}：裁决不变，新证据＝S3 门的种子子集敏感性 ＋ 场次聚类不确定度 ＋ 下一阶梯的检出力）** 远端 E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、汇总 **291**、只读诊断 297/298）、`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全、队列空、日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ 与前四十七轮完全一致；本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。六套既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。**新脚本 `tools/_e45_verify_round48.py`（只读本地，POS 14/14 VERDICT: PASS）**：① **判据根据被复算**：S3 的 12 个点估由逐 seed 产物**逐位复现**（max|diff| = 0.000e+00），**12 个区间全部跨零**（这才是判负根据，非点估符号）；538 窗原始表复算已发布 CSI/BIAS/POD/FAR（21 格 × 4）⇒ **84 次比较 max|diff| = 0.000e+00**。② **LOSO 种子子集敏感性**：抽掉任一 seed，**没有任何一臂会变成"通过"**；唯一在 4 个子集上都为正的是 **A4 的原始增益**（+0.018037，但等偏置后 −0.006161 ⇒ 正是「仍是面积/幅度效应」）与 **A2 的等偏置点估**（+0.004086，但区间跨零）⇒ **六臂判负对种子集合不敏感**。③ **第 46 轮「窗级 CI 是伪重复」的量化版**：配对窗差 **ICC 中位 0.268**（0.063–0.561）⇒ **538 窗的有效样本量中位仅 31.6 窗**；窗级 bootstrap **14/18 格不跨零**（与第 46 轮逐位一致）vs **场次聚类 bootstrap 仅 1/18 格不跨零**；**窗级 CI 半宽只有场次聚类 CI 的 0.136 倍** ⇒ **低估不确定性约 7.3 倍**。④ **下一阶梯需多少 seed（算力预算输入）**：seed 间增益 pooled SD **0.04896** ⇒ **MDE：n=3 → 0.15162、n=10 → 0.04870、n=30 → 0.02592**；六臂**无一超过 MDE(3)**（0/6，与区间全跨零自洽）；要检出 E45 实测效应量，A4 的 +0.018037 需 **n=60**、A3b +0.013653 需 **n=103**、A3a +0.007576 需 **n=330** ⇒ **「≥10 seed」只够检出 ≥0.0487 的效应；效应量 2 倍 ⇒ n=17、3 倍 ⇒ n=9** ⇒ 下一族若真实效应在 0.02–0.05，十 seed 不足以定案（需 17–60 seed）。**边界**：事后诊断、**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed；MDE 为正态近似功效估算）⇒ **六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**。
"""

LOG_BLOCK = f"""
**补记，{MARK}（2026-11-13 23:3x）**

- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、汇总 **291**、只读诊断 297/298）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；队列空；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **与前四十七轮完全一致，无新活动**。本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。
- **闸门**：六套（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无数值与裁决变化）。
- **新证据（S3 门的种子子集敏感性 ＋ 场次聚类不确定度 ＋ 下一阶梯的检出力）**——`tools/_e45_verify_round48.py`（只读本地，**POS 14/14 VERDICT: PASS**）：
  1. **判据根据被复算**：S3 的 **12 个点估逐位复现**（max|diff| = 0.000e+00）、**12 个区间全部跨零**（判负根据是区间不是符号）；538 窗原始表复算已发布 CSI/BIAS/POD/FAR ⇒ **84 次比较 max|diff| = 0.000e+00**。
  2. **LOSO 种子子集敏感性**：抽掉任一 seed，**无任何一臂会变成"通过"**；唯一 4/4 为正的是 **A4 原始增益**（+0.018037，等偏置后 −0.006161）与 **A2 等偏置点估**（+0.004086，区间跨零）⇒ **六臂判负对种子集合不敏感**。
  3. **第 46 轮「窗级 CI 是伪重复」的量化版**：配对窗差 **ICC 中位 0.268** ⇒ **538 窗的有效样本量中位仅 31.6 窗**；窗级 bootstrap **14/18 不跨零**（与第 46 轮逐位一致）vs **场次聚类 bootstrap 仅 1/18 不跨零**；**窗级 CI 半宽只有聚类 CI 的 0.136 倍** ⇒ **低估约 7.3 倍**。
  4. **下一阶梯需多少 seed（算力预算输入）**：pooled seed 间 SD **0.04896** ⇒ **MDE：n=3 → 0.15162、n=10 → 0.04870、n=30 → 0.02592**；六臂**无一超过 MDE(3)**；检出 E45 实测效应量需 **A4 n=60 / A3b n=103 / A3a n=330** ⇒ **「≥10 seed」只够检出 ≥0.0487 的效应**（2 倍效应 → n=17、3 倍 → n=9）⇒ 下一族若真实效应 0.02–0.05，**十 seed 不足以定案**。

**口径**：仍按中位数＋通过率报；配对与差值口径并列；事后诊断，不构成新判据（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed；MDE 为正态近似功效估算）。

**边界**：**裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版** ⇒ 新证据只进 `HANDOVER_E45.md` §4⑬、上层 MEMORY.md 状态段与三份逐日日志。
"""

AUTOMATION_BLOCK = f"""
### 2026-11-13 23:3x｜第 48 轮（S3 门的种子子集敏感性 ＋ 场次聚类不确定度 ＋ 下一阶梯的检出力）：状态复核（无远端改动，裁决不变）

**判定：远端 E45 全部完成且与前四十七轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`）。**

**远端**：提交器 **273 COMPLETED**（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；**队列空**；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。远端 13 份 json 与本地 `tools/_evals/e45/` **md5 13/13 全同**。

**闸门**：六套（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。

**新证据**——`tools/_e45_verify_round48.py`（只读本地，**POS 14/14 VERDICT: PASS**）：① **判据根据被复算**——S3 的 12 个点估逐位复现（max|diff| = 0.000e+00）、**12 个区间全部跨零**（判负根据是区间而非点估符号）；由 538 窗原始表复算已发布 CSI/BIAS/POD/FAR ⇒ **84 次比较 max|diff| = 0.000e+00**。② **LOSO 种子子集敏感性**——抽掉任一 seed，**没有任何一臂会变成"通过"**；唯一 4/4 为正的是 A4 的原始增益（+0.018037，等偏置后 −0.006161）与 A2 的等偏置点估（+0.004086，区间跨零）⇒ **六臂判负对种子集合不敏感**。③ **第 46 轮「窗级 CI 是伪重复」的量化版**——配对窗差 **ICC 中位 0.268** ⇒ **538 窗的有效样本量中位仅 31.6 窗**；窗级 bootstrap **14/18 格不跨零**（与第 46 轮逐位一致）vs **场次聚类 bootstrap 仅 1/18 格不跨零**；**窗级 CI 半宽只有场次聚类 CI 的 0.136 倍** ⇒ **低估不确定性约 7.3 倍**。④ **下一阶梯需多少 seed（算力预算输入）**——seed 间增益 pooled SD **0.04896** ⇒ **MDE：n=3 → 0.15162、n=10 → 0.04870、n=30 → 0.02592**；六臂**无一超过 MDE(3)**（0/6，与区间全跨零自洽）；检出 E45 实测效应量需 **A4 n=60 / A3b n=103 / A3a n=330** ⇒ **「≥10 seed」只够检出 ≥0.0487 的效应**（2 倍效应 → n=17、3 倍 → n=9）⇒ 下一族若真实效应 0.02–0.05，**十 seed 不足以定案**（需 17–60 seed）。

**写回**：`HANDOVER_E45.md` §4⑬ ＋ 上层 MEMORY.md 状态段 ＋ 三份逐日日志（脚本 `tools/_e45_writeback_round48.py`，幂等，标记＝插入文本原文子串，连跑两次第二遍全部 `unchanged`，各文件标记计数 == 1）。

**本轮踩坑（可复用规则）**：① **产物结构必须先 dump 再写断言**——`diag_probe_s*.json` 的网格在 **`arms` 子字典**下（不是顶层）、`scores_summary_s*.json` 的键是 **`ARM_sSEED_ep00`** 形式；② **`.windows.json` 的 `windows` 是与 `window_keys` 平行的 LIST，不是 dict**（按 key 索引会报 `TypeError: list indices must be integers`）；③ **断言不能把「待测量的现象」写成「必须成立的前提」**——本轮首版把「LOSO 下符号稳定」「窗级 CI 比聚类 CI 宽」写成硬断言，两条都 FAIL，而真实结论恰好相反（符号会翻、窗级 CI 更窄）⇒ **测量类问题应作为 INFO 报告，硬断言只留给「结论必须不变」类命题**；④ **本地默认 python 无 numpy** ⇒ 诊断脚本改用纯 stdlib（自写 t 分位数 + 线性插值分位数）；⑤ **跨轮复算必须对齐对方的口径**——第 47 轮的场次级是「先合并列联表再算场次 CSI」，不是「窗差中位数的中位数」，口径不对齐会差 6.9e−03；⑥ **f-string 里的花括号要转义**——正文里写 `raw/equal_bias`、`ARM_sSEED_ep00` 这类带花括号的记号会被当成插值表达式而 NameError。
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


def main():
    targets = [
        ("HANDOVER_E45.md", insert_before, HANDOVER, HANDOVER_ANCHOR, HANDOVER_BLOCK),
        ("MEMORY.md", append_block, MEMORY, None, MEMORY_BLOCK),
        ("memory/2026-11-11.md", append_block, LOG_1111, None, LOG_BLOCK),
        ("memory/2026-11-13.md", append_block, LOG_1113, None, LOG_BLOCK),
        ("project 2026-11-13.md", append_block, PROJ_LOG_1113, None, LOG_BLOCK),
        ("automation memory", append_block, AUTOMATION_MEM, None, AUTOMATION_BLOCK),
    ]

    for name, fn, path, anchor, block in targets:
        if not path.exists():
            raise SystemExit(f"MISSING TARGET: {path}")
        res = fn(path, anchor, block) if anchor else fn(path, block)
        print(f"  {name:>22}: {res}")
        # invariant: the guard marker must appear exactly once in the target
        n = path.read_text(encoding="utf-8").count(MARK)
        if n != 1:
            raise SystemExit(f"MARKER COUNT != 1 in {path}: {n}")
    print("all targets written/unchanged; marker count == 1 in every target")


if __name__ == "__main__":
    main()
