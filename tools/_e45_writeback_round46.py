"""Round 46 write-back: window-level (per-window) axis audit of E45.

Idempotent, marker-matched write-back in the style of ``tools/_sync_v243.py``
and ``tools/_e45_writeback_round45.py``:

* every target is searched for an idempotency marker; if the marker is already
  present the target is reported ``unchanged`` and nothing is written;
* if an anchor cannot be found the script **raises** -- it never silently skips;
* no markdown table row is hand-spliced; there is no table in this round's
  write-back.  (If a table row is ever added, it must be built from a list of
  cells with the separator inserted by code -- hand-spliced pipes break the
  column count and corrupt the whole table.)

Targets:
  1. R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md   (new section 4-11)
  2. R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md          (status section)
  3. R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md      (append)
  4. R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md      (append)
  5. R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-13.md (append)

Run:  python tools/_e45_writeback_round46.py
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Per-target idempotency markers.  Each marker MUST be a substring of the text
#: actually inserted into that target (rounds 41 and 43 both shipped the bug of
#: using one shared marker, which made the guard fire for one target only and
#: duplicated the block in the others).
MARKER_HANDOVER = "第 46 轮新增证据"
MARKER_MEMORY = "第 46 次独立复核：裁决不变，新证据＝窗级（window-level）轴的首次判读"
MARKER_LOG = "第 46 轮：状态复核（无远端改动，裁决不变）＋ 新证据＝窗级轴的首次判读"

UPPER = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJECT = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory")

HANDOVER = UPPER / "HANDOVER_E45.md"
MEMORY = UPPER / "MEMORY.md"
LOG_1111 = UPPER / "2026-11-11.md"
LOG_1113 = UPPER / "2026-11-13.md"
LOG_1113_PROJ = PROJECT / "2026-11-13.md"

# ---------------------------------------------------------------- handover 4-11
HANDOVER_BLOCK = """
**⑪（2026-11-13 20:5x 第 46 轮新增证据）窗级（per-window）轴的首次判读 ＋ 一个决定性的零模型对照**——脚本 `tools/_e45_verify_round46.py`（只读本地，**POS 7/7 VERDICT: PASS**）：前 45 轮的判读只用过三层产物——**聚合层**（`scores_summary*`／`s3_equalbias` 的 CSI/BIAS）、**机制探针层**（`diag_probe_s*`）、**S1 结构量层**（`struct_metrics_all.json`，第 45 轮首次）——**从未打开 E45 stage-2 产出的第四类产物 `runs/e45/evals/*.windows.json`（21 格 × 538 窗 × 4 阈值的逐窗列联表：hits/misses/false_alarms/correct_negatives）**。本轮首次在**窗级**判读，**裁决零变化**（事后诊断，**不构成新判据**：S3 护栏固定 thr=2.0 mm、复制单元固定为 seed），但拿到两条**独立于既有三条轴**、且**互相印证**的新证据：

1. **端到端复算：已发布的聚合指标可由原始窗表逐位重建**。把 21 份 `*.windows.json` 的 538 张窗级列联表按 `scripts/evaluate.py` 的口径汇总，与已发布的 `*.eval.json` 比对：**21 格 × 4 阈值 × {CSI, BIAS, POD, FAR} 共 336 次比较，全部 max|diff| = 0.000e+00**（逐位相同）。同时 **21 格的 `window_keys` 完全一致且顺序一致** ⇒ **配对比较在窗级是合法的**（此前 S3 的配对假设从未被独立验证过）。
2. **决定性的一条：给「臂 vs 对照」配了零模型对照，结论是「增益与噪声不可区分」**。用**同一个对照臂 B0 的两个不同 seed 互比**（纯 seed 噪声，不含任何机制）作零模型：**B0_s0 vs B0_s1 的窗级配对中位差 +0.0059～+0.0335（0/8 场次剔除后转负）；B0_s0 vs B0_s2 与 B0_s1 vs B0_s2 全部 8/8 场次剔除后转负、幅度 −0.0433～−0.0022**；**三对控制合并后的聚合 |ΔCSI| 噪声地板 = 0.038771**。对照六臂的**窗级配对中位差**：**A1 +0.000832、A2 −0.002185、A3a +0.003105、A3b +0.009133、A4 +0.012611、A4b +0.000333**——**六臂全部落在零模型噪声地板（0.0388）之内，且**没有一个臂的窗级效应量超过**「把对照臂换个种子」的幅度。⇒ **六臂判负在窗级独立复现**，且失效模式被进一步钉死为**「臂不稳（符号随种子翻转）」而非「普遍不足」**：A1 逐 seed 聚合增益 +0.0660/−0.0500、A4b +0.0483/**−0.1153**、A4 −0.0130、A3a −0.0205 反复变号。
3. **窗级 bootstrap 的 CI「不跨零」是伪重复，必须显式登记为反面教材**：在 18 个（臂×seed）格里，**窗级配对 bootstrap（10 000 次重抽、rng 20261113）的 CI 有 14/18 格不跨零**。但**复制单元是 seed（n=3），不是窗**——窗只是同一次采样的重复观测。实测**seed 层面的增益离散度是窗级 CI 半宽的 9.1～30.8 倍**（A2 9.1×、A4 9.8×、A4b 16.9×、A3a 21.9×、A3b 27.0×、A1 30.8×）⇒ **窗级 CI 的「显著」不构成效应显著的证据**，若照搬会把已判负的臂误判为通过。**新登记一条口径纪律：任何在窗/帧/像素级重抽得到的置信区间，只能描述该次采样的内部一致性，不得当作跨种子的证据**（与 §4④「过地板通过率不能单独用于晋级」、第 44 轮「配对口径与差值口径必须并列报」同族）。
4. **留一风暴场次（LOCO）稳健性检验**：538 窗来自 **8 个风暴场次**（真值 2 mm 事件量从 3.6 万到 510 万不等，`CP_01_08121100` 单场占 182 窗、510 万事件）。逐场剔除后重算聚合增益：**A3a/A3b/A4/A4b 四臂 8/8 场次剔除后中位数仍全为正**（A3a 最低 +0.000477、A3b +0.008865、A4 +0.003778、A4b +0.010334）⇒ **这四臂的表面增益不是单场次驱动**；**A1 7/8 场次转负、A2 8/8 场次转负**（与两臂本就在对照之下相符）。**但零模型同样给出「8/8 不转负」的形态**（B0_s0 vs B0_s1 即 0/8 转负）⇒ **LOCO 稳健性在对照臂自身也能出现，不能单独作为机制成立的证据**。
5. **增益高度集中在单一风暴场次**：`CP_01_09282353` 对 **A3a 46.9%、A3b 41.2%、A4 67.3%** 的总增益有贡献（该场次 B0 的 CSI 仅 0.0296、真值事件量仅 15 万，为八场次中第二弱）。⇒ 与第 40 轮「A4 的 +0.0180 平均原始增益里 99.2% 来自 seed 0」**在另一条轴上同向**：E45 的「增益」既不稳定于种子、也高度集中于个别场次。

**口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ **窗级轴只作诊断**，S3 护栏固定 thr=2.0 mm、复制单元固定为 seed ⇒ **不构成新判据**；④ **边界**：事后诊断，**不改变任何裁决**——**六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**（无数值与裁决变化）⇒ 新证据只进本节、上层 MEMORY.md 状态段与三份逐日日志。

**附带产物**：远端 `runs/e45/evals/` 的 42 份原始表已打包为 `tools/_evals/e45/e45_raw_audit.tar.gz`（md5 **9d8c1468d47c75df17600414fac28d2b**，与远端一致）取回本地；审计脚本 `tools/_e45_verify_round46.py` 可从该 tarball **自解压复现**（首次运行自动 unpack，POS 7/7 VERDICT: PASS），结果落 `tools/_evals/e45/round46_window_level.json`。
"""

# ------------------------------------------------------------------- memory bullet
MEMORY_BULLET = (
    "- **（2026-11-13 20:5x 第 46 次独立复核：裁决不变，新证据＝窗级（window-level）轴的首次判读）** "
    "远端 E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、"
    "汇总 **291**、只读诊断 297/298）、`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；"
    "远端与本地 md5 **13/13 全同**；六套既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／"
    "`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无裁决与数值变化）；"
    "本轮**未 sbatch、未改远端状态**（只下载了原始窗表打包 tar.gz，md5 9d8c1468d47c75df17600414fac28d2b）。"
    "**新脚本 `tools/_e45_verify_round46.py`（只读本地，POS 7/7 VERDICT: PASS）**：**前 45 轮只判过聚合层"
    "（CSI/BIAS）、机制探针层、S1 结构量层三条轴，从未打开 E45 的第四类产物 `runs/e45/evals/*.windows.json`"
    "（21 格 × 538 窗 × 4 阈值逐窗列联表）** ⇒ 本轮首次在**窗级**判读，**裁决零变化**，得两条新证据："
    "① **端到端复算通过**：由原始窗表按 `scripts/evaluate.py` 口径汇总，与已发布 `*.eval.json` 的 "
    "**336 次比较（21 格 × 4 阈值 × {CSI,BIAS,POD,FAR}）全部 max|diff| = 0.000e+00**（逐位相同）；"
    "**21 格 `window_keys` 完全一致且顺序一致** ⇒ **配对比较在窗级合法**（S3 的配对假设首次被独立验证）。"
    "② **决定性：给「臂 vs 对照」配零模型后，六臂增益与噪声不可区分**——用**对照臂 B0 的两个不同 seed 互比**"
    "作零模型（纯 seed 噪声）：三对控制的聚合 |ΔCSI| **噪声地板 = 0.038771**，且 B0_s0 vs B0_s1 的窗级配对中位差 "
    "**+0.0059～+0.0335（0/8 场次剔除后转负）**；而六臂窗级配对中位差仅 **A1 +0.000832、A2 −0.002185、"
    "A3a +0.003105、A3b +0.009133、A4 +0.012611、A4b +0.000333** ⇒ **六臂全部落在零模型噪声地板之内** ⇒ "
    "**六臂判负在窗级独立复现**，失效模式钉死为**「臂不稳、符号随种子翻转」**（A1 逐 seed +0.0660/−0.0500、"
    "A4b +0.0483/**−0.1153**、A4 −0.0130 反复变号）。"
    "**新登记一条口径纪律（反面教材）**：窗级配对 bootstrap 的 CI 在 **14/18** 格不跨零，**但复制单元是 seed（n=3）不是窗**"
    "——实测 **seed 层面增益离散度是窗级 CI 半宽的 9.1～30.8 倍** ⇒ **窗级 CI 的「显著」不构成效应显著的证据**，"
    "照搬会把已判负的臂误判为通过；**任何窗/帧/像素级重抽区间只描述该次采样的内部一致性**。"
    "另：**LOCO（留一风暴场次）**显示 A3a/A3b/A4/A4b 8/8 场次剔除后仍全正（增益非单场次驱动）、"
    "A1 7/8 与 A2 8/8 转负——**但零模型同样给出 0/8 不转负的形态 ⇒ LOCO 稳健性亦不能单独作机制证据**；"
    "且 `CP_01_09282353` 一场次贡献 **A3a 46.9%、A3b 41.2%、A4 67.3%** 的总增益（该场次 B0 CSI 仅 0.0296）"
    "⇒ 与第 40 轮「A4 增益 99.2% 来自 seed 0」**在另一条轴上同向**。"
    "**边界**：事后诊断、**不构成新判据** ⇒ **六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**、MASTER 不升版。"
    "\n"
)

# ----------------------------------------------------------------------- log block
LOG_BLOCK = """
### 2026-11-13 20:5x｜第 46 轮：状态复核（无远端改动，裁决不变）＋ 新证据＝窗级轴的首次判读

**远端**：提交器 **273 COMPLETED**（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、
只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；
日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；队列无 E45 作业；
`runs/` 下无 E46+ 新目录 ⇒ **无新活动**。**本轮未 sbatch、未改远端状态**（只取回原始窗表打包 tar.gz）。

**产物一致性**：远端 `runs/e45/*.json` 与本地 `tools/_evals/e45/*.json` md5 **13/13 全同**；
另取回 `runs/e45/evals/` 42 份原始表打包 `tools/_evals/e45/e45_raw_audit.tar.gz`
（md5 **9d8c1468d47c75df17600414fac28d2b**，与远端一致）。

**闸门**：`_check_tables` PASS（`introduced=[]`）、`_check_v245/246/247/248_row` 全 PASS、
`_e45_docs_numbers` PASS ⇒ **MASTER 仍 v2.48、裁决不变**。

**本轮新东西（前 45 轮只判聚合层／机制探针层／S1 结构量层三条轴）**——新脚本
`tools/_e45_verify_round46.py`（只读本地，**POS 7/7 VERDICT: PASS**）第一次打开 E45 的**第四类产物**
`runs/e45/evals/*.windows.json`（21 格 × 538 窗 × 4 阈值逐窗列联表）：

1. **端到端复算**：由原始窗表按 `scripts/evaluate.py` 口径汇总，与已发布 `*.eval.json` 比对，
   **336 次比较（21 格 × 4 阈值 × {CSI, BIAS, POD, FAR}）全部 max|diff| = 0.000e+00**（逐位相同）；
   **21 格 `window_keys` 完全一致且顺序一致** ⇒ **配对比较在窗级合法**（S3 的配对假设首次独立验证）。
2. **决定性：零模型对照 ⇒ 六臂增益与噪声不可区分**。用**对照臂 B0 的两个不同 seed 互比**（纯 seed 噪声）：
   三对控制的聚合 |ΔCSI| **噪声地板 = 0.038771**；B0_s0 vs B0_s1 的窗级配对中位差 **+0.0059～+0.0335**
   （0/8 场次剔除后转负），另两对 8/8 转负、幅度 −0.0433～−0.0022。六臂窗级配对中位差仅
   **A1 +0.000832、A2 −0.002185、A3a +0.003105、A3b +0.009133、A4 +0.012611、A4b +0.000333**
   ⇒ **全部落在噪声地板之内** ⇒ **六臂判负在窗级独立复现**，失效模式钉死为**「臂不稳、符号随种子翻转」**
   （A1 逐 seed +0.0660/−0.0500、A4b +0.0483/**−0.1153**、A4 −0.0130、A3a −0.0205）。
3. **新登记口径纪律（反面教材）**：窗级配对 bootstrap 的 CI 在 **14/18** 格不跨零，**但复制单元是 seed（n=3）
   不是窗**——**seed 层面增益离散度是窗级 CI 半宽的 9.1～30.8 倍** ⇒ **窗级 CI 的「显著」不构成效应显著的证据**，
   照搬会把已判负的臂误判为通过；**窗/帧/像素级重抽区间只描述该次采样的内部一致性**。
4. **LOCO（留一风暴场次，538 窗来自 8 个场次）**：**A3a/A3b/A4/A4b 8/8 场次剔除后中位数仍全正**
   （A3a 最低 +0.000477、A3b +0.008865、A4 +0.003778、A4b +0.010334）⇒ 增益非单场次驱动；
   **A1 7/8、A2 8/8 转负**。**但零模型同样给出 0/8 不转负的形态 ⇒ LOCO 稳健性不能单独作机制证据**。
5. **增益集中在单一场次**：`CP_01_09282353` 贡献 **A3a 46.9%、A3b 41.2%、A4 67.3%** 的总增益
   （该场次 B0 CSI 仅 0.0296、真值事件量 15 万，为八场次中第二弱）⇒ 与第 40 轮「A4 增益 99.2% 来自 seed 0」
   **在另一条轴上同向**。

**口径**：仍按中位数＋通过率报；配对与差值口径并列；窗级轴只作诊断。

**边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、复制单元固定为 seed）⇒ **裁决不变：六臂判负、
无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**（无数值与裁决变化）⇒ 新证据只进 `HANDOVER_E45.md` §4⑪、
上层 MEMORY.md 状态段与三份逐日日志。
"""


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def report(path: Path, status: str, detail: str = "") -> None:
    print(f"  {status:<10} {path}" + (f"  ({detail})" if detail else ""))


def insert_before_line(text: str, anchor_pred, block: str, path: Path) -> str:
    """Insert ``block`` immediately before the first line matching ``anchor_pred``."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if anchor_pred(line):
            return "".join(lines[:i]) + block + "".join(lines[i:])
    raise SystemExit(f"ANCHOR NOT FOUND in {path} -- refusing to skip silently")


def insert_after_line(text: str, anchor_pred, block: str, path: Path) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if anchor_pred(line):
            return "".join(lines[: i + 1]) + block + "".join(lines[i + 1 :])
    raise SystemExit(f"ANCHOR NOT FOUND in {path} -- refusing to skip silently")


def append_block(text: str, block: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    return text + block


def main() -> None:
    print("E45 round 46 write-back (idempotent, marker-matched)")
    results = []

    # ---- 1. HANDOVER: new section 4-11 ------------------------------------
    t = read(HANDOVER)
    if MARKER_HANDOVER in t:
        report(HANDOVER, "unchanged")
        results.append("handover unchanged")
    else:
        t2 = insert_before_line(
            t,
            lambda ln: ln.startswith("### 下面是结案前的执行记录"),
            HANDOVER_BLOCK.lstrip("\n") + "\n",
            HANDOVER,
        )
        write(HANDOVER, t2)
        report(HANDOVER, "written", "section 4-11")
        results.append("handover written")

    # ---- 2. upper MEMORY.md status section --------------------------------
    t = read(MEMORY)
    if MARKER_MEMORY in t:
        report(MEMORY, "unchanged")
        results.append("memory unchanged")
    else:
        t2 = insert_after_line(
            t,
            lambda ln: "第 45 次独立复核" in ln or "§4⑩" in ln,
            MEMORY_BULLET,
            MEMORY,
        )
        write(MEMORY, t2)
        report(MEMORY, "written", "status bullet")
        results.append("memory written")

    # ---- 3/4/5. the three daily logs --------------------------------------
    for path in (LOG_1111, LOG_1113, LOG_1113_PROJ):
        t = read(path)
        if MARKER_LOG in t:
            report(path, "unchanged")
            results.append(f"{path.name} unchanged")
            continue
        write(path, append_block(t, LOG_BLOCK))
        report(path, "written", "log block")
        results.append(f"{path.name} written")

    print("\n".join(f"  - {r}" for r in results))
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
