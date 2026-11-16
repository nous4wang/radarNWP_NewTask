"""Round 45 write-back: S1 structural-metrics axis (E45).

Idempotent, marker-matched write-back in the style of ``tools/_sync_v243.py``:

* every target is searched for an idempotency marker; if the marker is already
  present the target is reported ``unchanged`` and nothing is written;
* if an anchor cannot be found the script **raises** -- it never silently skips;
* no markdown table row is hand-spliced (there is no table in this round's
  write-back), so the column-count hazard does not apply.  Table rows, if any
  are ever added, must be built from a list of cells with the separator
  inserted by code.

Targets (E45 authority chain):
  1. R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md   (new section 4-10)
  2. R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md          (status section)
  3. R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md      (append)
  4. R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md      (append)
  5. R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-13.md (append)

Run:  python tools/_e45_writeback_round45.py
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Per-target idempotency markers.  A single shared marker is WRONG: the
#: marker must be a substring of the text that is actually inserted into that
#: target, otherwise the guard never fires for the other targets and a second
#: run appends them twice.  (Round 45 shipped exactly that bug; see
#: ``tools/_e45_repair_round45_dup.py``.)  Each marker below is a substring of
#: its own inserted block.
MARKER_HANDOVER = "第 45 轮新增证据"
MARKER_MEMORY = "第 45 次独立复核：裁决不变，新证据＝S1 结构量轴的**首次**判读"
MARKER_LOG = "第 45 轮：状态复核（无远端改动，裁决不变）＋ 新证据＝S1 结构量轴的首次判读"

UPPER = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJECT = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory")

HANDOVER = UPPER / "HANDOVER_E45.md"
MEMORY = UPPER / "MEMORY.md"
LOG_1111 = UPPER / "2026-11-11.md"
LOG_1113 = UPPER / "2026-11-13.md"
LOG_1113_PROJ = PROJECT / "2026-11-13.md"

# ---------------------------------------------------------------- handover 4-10
HANDOVER_BLOCK = """
**⑩（2026-11-13 19:3x 第 45 轮新增证据）S1 结构量轴的首次判读（前 44 轮只判了 CSI/BIAS 与机制探针两条轴）**——脚本 `tools/_e45_verify_round45.py`（只读本地，**VERDICT: PASS，POS 23/23**）：E45 的 stage-2 还产出过**第三份产物** `runs/e45/struct_metrics_all.json`（S1 结构探针：**13 个结构量 × 21 格**＝7 臂 × 3 seed，每格 538 窗 (window, lead) 帧取中位数），而**前 44 轮的判读从未碰过它**——全部只用了 `scores_summary*` / `s3_equalbias`（CSI/BIAS）与 `diag_probe_s*`（机制读数）两条轴。本轮第一次在它上面判读，**裁决零变化**（事后诊断，**不构成新判据**：S3 护栏固定 thr=2.0 mm），但拿到四条**独立于既有两条轴**的新证据：

1. **拓扑两臂的针尖被第二台独立仪器确认，并首次给出倍数**：S1 的 `tail_max_over_p99`（场最大÷场 p99）**真值 4.547525**；A3a 三 seed **46.0／51.2／56.72**（中位 51.2 ＝ 真值的 **11.3 倍**，逐 seed 10.1～12.5 倍）、A3b 三 seed **63.67／126.21／88.05**（中位 88.05 ＝ **19.4 倍**，逐 seed 14.0～**27.8 倍**）；**其余五臂全部 ≤ 真值的 1.06 倍**（B0 中位 3.831 ＝ 0.84 倍、A1 3.263、A2 3.910、A4 3.108、A4b 3.268）。逐帧场最大中位 A3a **75.158 mm**、A3b **166.837 mm**（A3b 逐 seed 最高 202.0 mm）＝ 真值 8.625 mm 的 **8.7 倍／19.3 倍**。⇒ **拓扑支结掉的结论在第二台仪器上独立复现**（机制探针的 `field_max_mm` 恒 402.43 mm 是**模型 clamp 上限**、本身不携带信息；S1 的场最大值是**逐帧实测**，不撞 clamp）。
2. **A4b 种子 2 的「量级塌缩」被第二台仪器独立确认**（v2.48 订正的「不是退化成常数」结论得到交叉验证）：该格 S1 场 `p99` = **0.3649**（真值 1.55 的 **0.24 倍**）、场最大 = **1.9655 mm**（真值 8.625 的 **0.23 倍**）、`dry_frac_lt_0p1` = **0.9605**（真值 0.7974）⇒ **整个场缩到真值的四分之一以下**，与机制探针的 `scale_mean` 0.0316、`scale_truth_corr` 0.2789 **同向**；**六臂 21 格里只有这一格是「场级塌缩」**（其余各格场级 p99/最大值均在真值的 0.7～2.1 倍内）⇒ 「A4b 种子 2 是量级塌缩＋与真值脱钩」**不是探针口径的假象**。
3. **新指纹：A4 的输出场几乎没有精确零像素**（21 格里唯一）：`probe_zero_frac`（**落在场最低值的像素比例**；真值 0.7974，因为真值的最小值就是 0）A4 三 seed **0.0012／0.0010／0.0010**，其余六臂全部 ≥ **0.44**（A4b 中位 0.5114、B0 0.7562、A3b 0.7109、A2 0.7352、A3a 0.7393、A1 0.7940）。而 A4 的**干区面积**（< 0.1 mm 比例）中位 0.8354 与对照 B0 的 0.7765 相若 ⇒ **A4 把干区表达成「微小正值」而不是「精确零」**。**候选解释（假设，未建立因果）**：A4 的输出是 `out = T·S`、`S = F_mm/(mean(F_mm)+1e-12)`（`models/scaled_structure/model.py:263`），干区若未被 `floor_at_zero` 压到精确零就会留下一层浅底；这与**待办②（`mean(out)=T` 破在绝对 `1e-12` 护栏）**、与 **A4b 种子 2 的整场塌缩**（F2）**同属一条「量通道 / 形状归一化」线索**，但**两者方向相反、尚未建立因果** ⇒ **修护栏后重测时必须同时判这三格，并按分支报**。
4. **新缺陷登记：`hf_share` 在 8/21 格是 NaN，成因是共享 helper 的非 nan 感知聚合**：`tools/_e41_struct_metrics.py::_median` 只过滤 `None`、**不过滤 NaN** ⇒ 任一 NaN 帧会污染整格中位数；而 `hf_share` 在**全干常数帧**上恰好是 0/0（z 标准化后功率为 0）⇒ NaN 格共 **8 个**（B0_s0、A1_s0、A1_s2、A2_s0、A2_s1、A2_s2、A4b_s0、A4b_s2），且**每一格都是干区主导的格**。⇒ **待办③（聚合改 `np.nanmedian`）的修复必须覆盖共享 helper，而不只是机制探针的 `maxratio`**——这是**同一个缺陷家族的第二个实例**，此前只登记了机制探针一处。

**口径与纪律**：① 本轮在 S1 轴上**两种口径并列报**（配对口径＝逐 seed 差值取中位数／差值口径＝中位数相减），六臂**符号全部一致**（A2 +0.079／+0.079、A3a +47.369／+47.369、A3b +83.884／+84.221 显著为正；A1 −0.277／−0.568、A4 −0.117／−0.723、A4b −0.139／−0.562 为负）⇒ **无种子对齐效应**（对照第 44 轮在 CSI 轴上发现的情形）；② 仍按「**中位数 + 通过率**」报，单点分数不作裁决；③ **S1 读数是「场级」证据，与 CSI/BIAS 的「窗级」证据互补而非替代**；④ **边界**：事后诊断，**不构成新判据**，**不改变任何裁决**（六臂仍全部判负、**无一臂进十 seed 算力**），**MASTER 不升版**（无数值与裁决变化）⇒ 新证据只进本节、上层 MEMORY.md 状态段与三份逐日日志。
"""

# ------------------------------------------------------------------- memory bullet
MEMORY_BULLET = (
    "- **（2026-11-13 19:3x 第 45 次独立复核：裁决不变，新证据＝S1 结构量轴的**首次**判读）** "
    "远端 E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、"
    "汇总 **291**、只读诊断 297/298）、`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；"
    "远端与本地 md5 **13/13 全同**；十套既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／"
    "`_e45_docs_numbers`／`_e45_verify_round27/40/43/44`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、"
    "**不升版**（无裁决与数值变化）；本轮**未下载、未 sbatch、未改远端状态**。"
    "**新脚本 `tools/_e45_verify_round45.py`（只读本地，POS 23/23 VERDICT: PASS）**：**前 44 轮的判读只用两条轴"
    "（CSI/BIAS ＋ 机制探针），从未碰过 E45 stage-2 产出的第三份产物 `struct_metrics_all.json`"
    "（S1 结构量 13 项 × 21 格，538 窗中位数）** ⇒ 本轮首次在该轴上判读，**裁决零变化**，但得四条新证据："
    "① **拓扑两臂被第二台独立仪器确认**：S1 `tail_max_over_p99` 真值 4.547525，A3a 三 seed 46.0/51.2/56.72"
    "（中位 11.3×真值）、A3b 63.67/126.21/88.05（中位 19.4×，最高 27.8×），**其余五臂全部 ≤1.06×**；"
    "逐帧场最大中位 A3a 75.158 mm／A3b 166.837 mm ＝ 真值 8.625 mm 的 8.7×／19.3× ⇒ **拓扑支结掉在第二台仪器上复现**"
    "（机制探针的 402.43 mm 是 clamp 上限、不携带信息；S1 场最大值是逐帧实测、不撞 clamp）。"
    "② **A4b 种子 2 的「量级塌缩」获独立确认**（v2.48 订正得到交叉验证）：该格 S1 场 p99 **0.3649**（真值 0.24×）、"
    "场最大 **1.9655 mm**（真值 0.23×）、`dry_frac` 0.9605（真值 0.7974）⇒ **整场塌到真值四分之一以下**，"
    "且 **21 格里仅此一格是场级塌缩** ⇒ 不是探针口径假象。③ **新指纹：A4 输出场几乎没有精确零像素**"
    "（`probe_zero_frac` 三 seed 0.0012/0.0010/0.0010，其余六臂全部 ≥0.44），而干区面积与对照相若 ⇒ "
    "**A4 把干区表达成微小正值而非精确零**；**候选解释（假设，未建立因果）**：与待办②（`mean(out)=T` 破在绝对 "
    "`1e-12` 护栏）、与 A4b 种子 2 的整场塌缩同属一条「量通道／形状归一化」线索，但方向相反 ⇒ "
    "**修护栏后重测时须同时判这三格并分分支报**。④ **新缺陷登记：`hf_share` 在 8/21 格为 NaN**，"
    "成因是共享 helper `tools/_e41_struct_metrics.py::_median` 只过滤 `None` 不过滤 NaN（`hf_share` 在全干常数帧上是 0/0）"
    "⇒ **待办③ 的修复必须覆盖共享 helper，不只是机制探针的 `maxratio`**（同一缺陷家族的第二个实例）。"
    "**口径**：S1 轴上**两种口径并列**（配对／差值），六臂**符号全部一致** ⇒ 无种子对齐效应；仍按中位数＋通过率报。"
    "**边界**：事后诊断、**不构成新判据** ⇒ **六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**。"
    "\n"
)

# ----------------------------------------------------------------------- log block
LOG_BLOCK = """
### 2026-11-13 19:3x｜第 45 轮：状态复核（无远端改动，裁决不变）＋ 新证据＝S1 结构量轴的首次判读

**远端**：提交器 **273 COMPLETED**（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 291、
只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐；`runs/e45/` **13 份** json 齐全；
日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。**本轮未下载、未 sbatch、未改远端状态。**

**产物一致性**：远端 `runs/e45/*.json` 与本地 `tools/_evals/e45/*.json` md5 **13/13 全同**。

**闸门**：`_check_tables` PASS（`introduced=[]`）、`_check_v245/246/247/248_row` 全 PASS、
`_e45_docs_numbers` PASS、`_e45_verify_round27/40/43/44` 全 PASS ⇒ **MASTER 仍 v2.48、裁决不变**。

**本轮新东西（前 44 轮只判 CSI/BIAS 与机制探针两条轴）**——新脚本 `tools/_e45_verify_round45.py`
（只读本地，**POS 23/23 VERDICT: PASS**）第一次判读 E45 stage-2 的**第三份产物**
`struct_metrics_all.json`（S1 结构量 13 项 × 21 格，538 窗中位数）：

1. **拓扑两臂被第二台独立仪器确认，并首次给出倍数**：S1 `tail_max_over_p99` 真值 4.547525，
   A3a 46.0/51.2/56.72（中位 **11.3×**）、A3b 63.67/126.21/88.05（中位 **19.4×**，最高 **27.8×**），
   **其余五臂全部 ≤1.06×**；逐帧场最大中位 A3a **75.158 mm**／A3b **166.837 mm**（真值 8.625 mm 的 8.7×／19.3×）
   ⇒ **拓扑支结掉在第二台仪器上独立复现**（机制探针的 402.43 mm 是 clamp 上限、不携带信息；S1 场最大值是逐帧实测）。
2. **A4b 种子 2 的「量级塌缩」获独立确认**（v2.48 订正交叉验证）：该格场 p99 **0.3649**（真值 0.24×）、
   场最大 **1.9655 mm**（真值 0.23×）、`dry_frac` 0.9605（真值 0.7974）⇒ 整场塌到真值四分之一以下，
   且 **21 格里仅此一格场级塌缩** ⇒ 不是探针口径假象。
3. **新指纹：A4 输出场几乎没有精确零像素**（`probe_zero_frac` 三 seed 0.0012/0.0010/0.0010，
   其余六臂全部 ≥0.44），而干区面积与对照相若 ⇒ A4 把干区表达成微小正值而非精确零；
   **候选解释（假设，未建立因果）**：与待办②（`1e-12` 护栏）、与 A4b 种子 2 的整场塌缩同属一条
   「量通道／形状归一化」线索但方向相反 ⇒ **修护栏后重测须同时判这三格、分分支报**。
4. **新缺陷登记**：`hf_share` 在 **8/21** 格为 NaN，成因是共享 helper
   `tools/_e41_struct_metrics.py::_median` 只过滤 `None` 不过滤 NaN（`hf_share` 在全干常数帧上是 0/0）
   ⇒ **待办③ 的修复必须覆盖共享 helper**，不只是机制探针的 `maxratio`（同一缺陷家族第二实例）。

**口径**：S1 轴上**配对口径与差值口径并列**（六臂符号全部一致 ⇒ 无种子对齐效应）；仍按中位数＋通过率报。

**边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm）⇒ **裁决不变：六臂判负、无一臂进十 seed、
不投十 seed 算力**；**MASTER 不升版**（无数值与裁决变化）⇒ 新证据只进 `HANDOVER_E45.md` §4⑩、
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
    print("E45 round 45 write-back (idempotent, marker-matched)")
    results = []

    # ---- 1. HANDOVER: new section 4-10 ------------------------------------
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
        report(HANDOVER, "written", "section 4-10")
        results.append("handover written")

    # ---- 2. upper MEMORY.md status section --------------------------------
    t = read(MEMORY)
    if MARKER_MEMORY in t:
        report(MEMORY, "unchanged")
        results.append("memory unchanged")
    else:
        t2 = insert_after_line(
            t,
            lambda ln: "§4⑨" in ln or "第 44 轮" in ln,
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
