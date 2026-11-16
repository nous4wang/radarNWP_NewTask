#!/usr/bin/env python3
"""Round-96 write-back: append the round-96 record to the four memory files.

Why this script exists
-----------------------
Round 96's verdict is IDENTICAL to what is already published (six arms
negative, no arm into the ten-seed run, no ten-seed compute) and it produced
NO new published number.  By the project's established rule -- "no new number
and no verdict change => do not touch docs/ and do not bump the MASTER
version" (rounds 87-91) -- ``docs/`` stays untouched at v2.54.  Only the
memory records need the round appended.

The four targets and their idempotency guards:

  1. R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md   (outer daily log)
  2. R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md (handover)
  3. R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-16.md
  4. R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md        (long-term)

Discipline carried over from round 92: **the guard must be a marker unique to
each file**.  A bare version string such as "v2.55" also occurs inside prose
elsewhere in the same files, which made an earlier guard mis-fire and silently
skip two files.  So each guard below is a file-specific marker, and every
anchor is verified to exist BEFORE anything is written.  A missing anchor is a
hard error, never a silent skip.

Two further correctness rules, both learned the hard way:

*   **line endings are preserved.**  All four files are CRLF.  The files are
    read with ``newline=""`` so the terminators survive as part of the text,
    and every inserted line carries the same terminator as its neighbours.
*   **the status bullet is replaced as a WHOLE line**, not as a prefix.
    Replacing only the prefix would leave the tail of the old bullet dangling
    after the new text.

Running this script twice must leave the files byte-identical to running it
once.
"""
from __future__ import annotations

import os
import sys

OUTER = r"R:/1radarNWPprediction"
PROJ = r"R:/1radarNWPprediction/radar-nwp-bench"

DAILY_OUTER = os.path.join(OUTER, ".workbuddy", "memory", "2026-11-11.md")
HANDOVER = os.path.join(OUTER, ".workbuddy", "memory", "HANDOVER_E45.md")
MEMORY_LT = os.path.join(OUTER, ".workbuddy", "memory", "MEMORY.md")
DAILY_PROJ = os.path.join(PROJ, ".workbuddy", "memory", "2026-11-16.md")

MARK = "E45-R96-BEGIN"
MARK_END = "E45-R96-END"

#: prefix of the long-term-memory status bullet that this round refreshes
STATUS_PREFIX = "- **E45 权威链全 `COMPLETED`**"
#: prefix of the per-round bullet that this round appends after
CONCL_PREFIX = "- **E45 第 95 轮（2026-11-16）**"

STATUS_NEW = (
    "- **E45 权威链全 `COMPLETED`**（273／274·282·290／275·283／291／297·298／314–317／343）；"
    "21 份评测齐、**14 份 run 级 json 远端↔本地 md5 14/14**；主跑评测树 42 份逐文件 md5 双向比对"
    "「缺失 0／多余 0／不一致 0」；S3 12/12 跨零护栏结构化重算一致，"
    "**两侧各 6 臂 × 20 个 bootstrap 种子 = 240/240 格全含零、合格种子数 0**；"
    "**第 96 轮（2026-11-16）新增三条互证审计面**：① 两个独立探针互相印证拓扑支结掉"
    "（`diag_probe` 的场最大值六格全 402.4288 mm ＝ 模型上限，`struct_metrics` 的**帧中位数**"
    "场最大 A3a 65.3–96.1／A3b 102.2–202.0 mm 在**每个种子**都高于所有非拓扑臂的 9.6 mm ⇒ "
    "针尖不是单帧孤例）；② E45 的 B0 记录与**别的实验**逐位相同"
    "（E43／E44 的 `B0_s0_ep00` 与 `B0_s1_ep00` 的 `csi_at_2.0`／`bias_at_2.0` 与 E45 **3/3 全等**）"
    "⇒ 交接文档里「B0 与 E43/E44 逐位相同」的登记**已实测确认**；③ 21 份主跑评测经**两条独立下载批次**"
    "取得（`raw_audit/` 与 `epoch_axis/`）**逐位相同 21/21**。**裁决不变＝六臂判负、无一臂进十 seed、"
    "不投十 seed 算力**；`docs/` 未改（无新数字、无裁决变化 ⇒ MASTER 仍 **v2.54**，不升版）；"
    "队列里 `qlfn2-f36/f37` 属同一 `wf` 账号下别的项目线、不干预；无 E46+ 正式训练目录。"
)

CONCL_NEW = (
    "- **E45 第 96 轮（2026-11-16）**：远端主链终态不变（273／275·283／274·282·290／291／297·298／"
    "314–317／343 全 `COMPLETED`）、`squeue -u wf` **空**、21 份 `.eval.json` ＋ 14 份 run 级 json 齐；"
    "**结案后（> 11-11 14:20:42）被修改的文件仅 1 份＝作业 343 自己的新产物 `s3_seedrobust.json`**；"
    "**14 份产物远端↔本地 md5 14/14 逐位全同**（本轮实测）；`sacct -S 2026-11-12` 主链 0 个作业 ⇒ "
    "**主链从未重启**；**无 E46+ 正式训练目录**；**S3 由 summary 291 当年自动跑完 ⇒ 本轮无需手动 sbatch**。"
    "**远端动作＝0**（只读查询；未 sbatch／未同步／未下载）。**裁决独立重导**"
    "（`tools/_e45_round96.py`，只读本地，**POS 22/22 VERDICT: PASS**）：头条表由 21 份主跑 "
    "`*.eval.json` 复算 ⇒ 七臂中位数与过地板通过率与已发布值 **max|dev| = 0.000e+00**"
    "（B0 0.121817／2/3；A3a 0.110015、A3b 0.116055 虽 3/3 过地板但**中位数均低于同批对照 B0** ⇒ 不晋级）；"
    "机制判据逐格重判与已发布一致（A1／A2 3/3、A3a／A3b 0/3、A4 3/3、A4b 2/3 过；"
    "六格 `scale_cv` 最小 1.2358 ＝ 底线 24.7 倍 ⇒ **全程未退化成常数**；`mean(out)=T` 仍 4/6 不达、"
    "**只记实现缺陷**）；S3 **数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，不读存储布尔）⇒ **12/12 跨零**、"
    "失效模式**分两支报**（`GAIN_WAS_AREA_OR_AMPLITUDE` 4 臂／`NO_GAIN_TO_BEGIN_WITH` 2 臂）；"
    "种子扫描 **两侧各 6 臂 × 20 个种子 = 240/240 格全含零**、**合格种子数 0**、注册种子逐位复现。"
    "**本轮三条新互证审计面**：① **两个独立探针互相印证拓扑支结掉**——`diag_probe` 的场最大值六格全 "
    "402.4288 mm（＝ `expm1(6)` 模型上限），而 `struct_metrics` 的**帧中位数**场最大 A3a 65.3–96.1 mm／"
    "A3b 102.2–202.0 mm 在**每个种子**都远高于所有非拓扑臂（≤9.6 mm）⇒ **针尖不是单帧孤例**；"
    "② **E45 的 B0 记录与别的实验逐位相同**——E43／E44 的 `B0_s0_ep00`、`B0_s1_ep00` 的 "
    "`csi_at_2.0`／`bias_at_2.0` 与 E45 **3/3 全等** ⇒ 交接文档的登记**已实测确认**；"
    "③ **两条独立下载批次的 21 份主跑评测逐位相同 21/21**（`raw_audit/` vs `epoch_axis/`）。"
    "**文档闸门** tables／v245／docs-numbers 全 PASS、`_check_v254_row` **30/30 PASS**；"
    "`_check_v248/251/252/253_row` 的失败项**均为 SNAPSHOT 类、按设计过期**（版本指针已前推到 v2.54、"
    "changelog 行后不再为空行；**CONTENT 类失败 = 0**）⇒ **未改 `docs/`**（无新数字、无裁决变化 ⇒ "
    "MASTER 仍 **v2.54**，不升版）。**下一步（不变）**：§4.5 三件实现缺陷修复（① 训练侧再接一次 "
    "`--model-kwargs {\"clamp_mm\":20.0}`；② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 "
    "`np.nanmedian` ＋ 报可用帧数）＋ **重开须重新提机制族**。"
)

BODY = """### 2026-11-16｜第 96 轮（E45 远端状态复核 ＋ **三条新互证审计面** ＋ **两处自身空断言修正**）｜远端动作＝0，裁决不变

- **远端终态（远端动作＝0）**：权威链 273／275·283（训练数组 7/7+7/7）／274·282·290（stage2）／291（summary）／297·298／314／315·316·317／343 全 `COMPLETED`；**`squeue -u wf` 空**（上一轮队列里那 8 个 `qlfn2-f38` 属同一 `wf` 账号下**别的项目线**，本轮已全部出队，仍不干预）；21 份 `.eval.json` ＋ **14 份** run 级 json 齐；**结案后（> 11-11 14:20:42）被修改的文件仅 1 份＝作业 343 自己的新产物 `s3_seedrobust.json`**（已发布产物零改动）；**14 份产物远端↔本地 md5 14/14 逐位全同**（本轮实测）；`sacct -S 2026-11-12` 主链 **0 个** e45-train／e45-stage2／e45-submit ⇒ **主链从未重启**；`runs/` 下**无 E46+ 正式训练目录**。**S3 由 summary 291 当年自动跑完 ⇒ 本轮无需手动 sbatch**（未 sbatch／未同步／未下载／未改远端状态）。
- **裁决独立重导**（`tools/_e45_round96.py`，只读本地，**POS 22/22 VERDICT: PASS**）：① 头条表由 **21 份主跑 `*.eval.json`** 的 `thresholds["2.0"].csi` 复算 ⇒ 七臂中位数与过地板通过率与已发布值 **max|dev| = 0.000e+00**（对照 B0 中位数 0.121817／**2/3** 过地板；A3a 0.110015、A3b 0.116055 虽各自 **3/3** 过地板，但**中位数均低于同批对照 B0** ⇒ 不晋级；超 Oracle 上界 0.1653 的仍只有 **A3b／A4** 各 1/3 seed ⇒ 过预报、须 S3 才判）；② 机制判据逐格重判与已发布**一致**（门指向 A1／A2 **3/3** 过、拓扑两臂 **0/3** 过、量通道 A4 **3/3**／A4b **2/3** 过；六格 `scale_cv` 最小 **1.2358** ＝ 底线 24.7 倍 ⇒ **全程没有任何一格真的退化成常数**；`mean(out)=T` 仍 **4/6 格不达** ⇒ **只记实现缺陷、不当机制结论**）；③ S3 **数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，**不读存储布尔**）⇒ **12/12 区间跨零**、失效模式**分两支报**（`GAIN_WAS_AREA_OR_AMPLITUDE` 4 臂／`NO_GAIN_TO_BEGIN_WITH` 2 臂）、六臂 `max_abs_bias_gap` 全 ≤4.1e-07；④ 种子稳健性扫读 `s3_seedrobust.json` ⇒ **两侧各 6 臂 × 20 个种子 = 240/240 格全含零**、按预注册晋级判据（点估 > 0 且区间不含零）**合格种子数 = 0**、注册种子**逐位复现（max|dev| = 0）**。
- **本轮三条新互证审计面**（前 95 轮都未测过）：① **两个独立探针互相印证「拓扑支结掉」**——`diag_probe` 的 `field_max_mm` 六格全 **402.4288 mm**（＝`expm1(6)` 的模型硬上限），而 `struct_metrics` 的**帧中位数**场最大（另一套实现 `tools/_e41_struct_metrics.py`）为 **A3a 65.3／75.2／96.1 mm、A3b 102.2／166.8／202.0 mm**，在**每一个种子**都远高于所有非拓扑臂（≤ **9.6** mm）⇒ **针尖不是单帧孤例，而是整条分布的上尾**；② **E45 的 B0 记录与别的实验逐位相同**——E43／E44 的 `B0_s0_ep00`、`B0_s1_ep00` 的 `csi_at_2.0`／`bias_at_2.0` 与 E45 **3/3 全等**（`0.12181703042952949`／`0.09169885007029324`）⇒ 交接文档里「B0 与 E43/E44 逐位相同」的登记**已实测确认**（不是拿 E45 自己的副本自证）；③ **两条独立下载批次的 21 份主跑评测逐位相同 21/21**（`raw_audit/` 与 `epoch_axis/` 两个目录各自独立取得）⇒ 已发布头条表所依赖的采集**可复现**。
- **文档闸门**：`_check_tables` **PASS**（MASTER／INDEX／E45 实验文件 mismatched=0）、`_check_v245_row` **PASS**、`_e45_docs_numbers` **PASS**、新闸门 `_check_v254_row` **POS 30/30 PASS**；`_check_v248/251/252/253_row` 的失败项**逐条核对后确认全部属 SNAPSHOT 类、按设计过期**（版本指针已由 v2.48 前推到 **v2.54**、v2.48 changelog 行后面不再是空行；**CONTENT 类失败 = 0**）⇒ **未改 `docs/`**（无新数字、无裁决变化 ⇒ **MASTER 仍 v2.54，不升版**）。
- **裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；**下一步（不变）**：§4.5 三件实现缺陷修复（① 训练侧再接一次 `--model-kwargs {"clamp_mm":20.0}`；② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ **重开须重新提机制族**。
- **本轮自身失误与修正（记录以防重犯）**：新写的审计脚本首跑报 **21/21 全绿，但其中两条断言是空的**——① 种子稳健性按 `equal_bias_sweep`／`equal_bias_reproduced` 取键，而文件里**等偏置半边实际存在 `sweep`／`reproduced`** ⇒ 拿到的是 **0 个**区间，于是「20/20 全含零」退化成了 **0 == 0 的空断言**；② B0 互证用 glob 扫全树，只匹配到 **E45 自己的 7 份副本**，等于拿自己证明自己。⇒ 两条都已改成**指名道姓**的检查：键名先与文件内容对照确认（并加「每半边各 20 格非空 ＋ 两半边各自与其已发布记录点估计相同」的归属断言），B0 互证改为**直接读 `e43_scores_summary.json`／`e44_scores_summary.json` 的 `B0_s0_ep00`／`B0_s1_ep00`**。⇒ **纪律：新增审计项必须先核对原始布局再写断言；「全绿」不等于「查到了东西」——空断言也是全绿，必须显式断言被查对象非空、且逐条核对断言对象与被查对象是不是同一个东西。**
- **产物**：`tools/_e45_round96.py`、`tools/_evals/e45/round96_status_audit.json`。
"""


def read(path: str) -> str:
    """Read preserving the original line terminators."""
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def eol_of(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def append_block(path: str, header: str) -> bool:
    """Append the marked block unless the marker is already present."""
    text = read(path)
    if MARK in text:
        print(f"  SKIP (marker present): {path}")
        return False
    if header and header not in text:
        raise SystemExit(f"ANCHOR MISSING in {path}: {header!r}")
    eol = eol_of(text)
    body = BODY.replace("\n", eol)
    block = f"{eol}{header}{eol}{eol}{MARK}{eol}{body}{MARK_END}{eol}"
    if not text.endswith(eol):
        text += eol
    write(path, text + block[len(eol):])
    print(f"  APPENDED: {path}")
    return True


def main() -> int:
    changed = 0

    print("outer daily log")
    changed += append_block(DAILY_OUTER, "## v2.54 第 95 轮")

    print("handover")
    changed += append_block(HANDOVER, "<!-- E45-R95-END -->")

    print("project daily log")
    changed += append_block(DAILY_PROJ, "<!-- E45-R95-END -->")

    print("long-term memory")
    text = read(MEMORY_LT)
    if MARK in text:
        print("  SKIP (marker present): " + MEMORY_LT)
    else:
        eol = eol_of(text)
        lines = text.split(eol)
        # --- refresh the status bullet as a WHOLE line
        hit = [i for i, l in enumerate(lines) if l.startswith(STATUS_PREFIX)]
        if len(hit) != 1:
            raise SystemExit(
                f"ANCHOR ERROR in {MEMORY_LT}: expected exactly 1 status bullet, found {len(hit)}"
            )
        lines[hit[0]] = STATUS_NEW
        # --- append the per-round bullet right after the round-95 bullet,
        #     together with this file's OWN marker so the guard can fire.
        #     (Round-96 lesson: a file whose edit path does not write the
        #     marker is a file whose idempotency guard never fires -- the
        #     second run re-inserted the bullet a second time.)
        hit2 = [i for i, l in enumerate(lines) if l.startswith(CONCL_PREFIX)]
        if len(hit2) != 1:
            raise SystemExit(
                f"ANCHOR ERROR in {MEMORY_LT}: expected exactly 1 round-95 bullet, found {len(hit2)}"
            )
        lines.insert(hit2[0] + 1, MARK)
        lines.insert(hit2[0] + 2, CONCL_NEW)
        lines.insert(hit2[0] + 3, MARK_END)
        write(MEMORY_LT, eol.join(lines))
        changed += 1
        print(f"  APPENDED: {MEMORY_LT}")

    print(f"\nfiles changed this run: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
