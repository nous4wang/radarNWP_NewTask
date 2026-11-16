"""Write back the round-96 findings (v2.55).

What v2.55 records
------------------
Round 96 re-checked the remote Slurm state and re-derived the E45 verdict
from the raw artifacts (POS 22/22, VERDICT: PASS).  The remote is unchanged:
the E45 main chain is still COMPLETED, the queue is empty, and all 14
artifacts are byte-identical to the local copies (md5 14/14).  The verdict is
unchanged: six arms negative, none to ten seeds, no ten-seed compute.

What IS new this round, and why it justifies a version bump:

1.  **Three new cross-artifact corroborations** (no earlier round measured
    these):
    *   **Two independent probes agree that the topology branch is closed.**
        ``diag_probe`` reports ``field_max_mm`` = 402.4288 mm on all six
        cells (the model's hard ``expm1(6)`` ceiling), while
        ``struct_metrics`` -- a DIFFERENT implementation
        (``tools/_e41_struct_metrics.py``) -- reports the MEDIAN over
        (window, lead) frames of the same per-frame maximum: A3a
        65.3/75.2/96.1 mm, A3b 102.2/166.8/202.0 mm, i.e. above every
        non-topo arm (<= 9.6 mm) on every seed.  So the spike is not a
        single-frame artefact: it is the upper tail of the whole
        distribution.
    *   **E45's B0 records are bit-identical to OTHER experiments' B0
        records.**  The handover registered "B0 is bit-identical to E43/E44",
        but that had never been measured against another experiment.  It is:
        E43/E44's ``B0_s0_ep00`` and ``B0_s1_ep00`` match E45's
        ``csi_at_2.0`` and ``bias_at_2.0`` exactly (3/3 records).
    *   **The 21 primary evaluations were fetched in two independent
        download batches** (``raw_audit/`` and ``epoch_axis/``) and are
        byte-identical 21/21.

2.  **A real defect in this document, found and fixed.**  MASTER's own
    section 12 v2.54 row states "14 artifacts, md5 14/14", but two
    CURRENT-STATE statements still said "13 artifacts, md5 13/13":
    the section 4.47 "authoritative links" bullet.  The document
    contradicted itself.  The historical per-round rows are deliberately NOT
    rewritten -- they are the audit trail.

3.  **Two vacuous assertions in this round's own audit script**, caught and
    fixed: the bootstrap-sweep key names (``sweep``/``reproduced``, not
    ``equal_bias_sweep``/``equal_bias_reproduced``) had silently reduced a
    "20/20 intervals contain zero" check to "0 == 0"; and the B0
    cross-experiment check had globbed the whole tree and matched only E45's
    OWN seven copies -- proving nothing.

Scope of this write-back
------------------------
* MASTER.md -- header version line, the section 4.47 card row, the section 9
  row, a section 12 changelog row, and the stale artifact count in the
  section 4.47 authoritative-links bullet (the four places Gate 3 names);
* INDEX.md -- the E45 row;
* E45 experiment file -- the version pointer and the round note;
* MEMORY.md -- the version pointer, the status section, the round bullet;
* HANDOVER_E45.md -- the version pointer, the round note;
* both daily logs -- the round note.

Table rows are built from their CELLS and the separators are inserted by the
code; the cell count is compared against the row it follows.  Pipes inside a
cell are escaped, because an unescaped one splits the cell and breaks the
whole table on render.

A pre-flight pass validates EVERY anchor before a single file is written, so
a typo cannot leave the document set half-written.  Running the script twice
is a no-op.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

MASTER = ROOT / "docs" / "MASTER.md"
INDEX = ROOT / "docs" / "experiments" / "INDEX.md"
E45DOC = ROOT / "docs" / "experiments" / "E45-gate-direction-amount-separation-baseline.md"
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
DAILY = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
DAILY_PROJECT = ROOT / ".workbuddy" / "memory" / "2026-11-16.md"

TAG = "v2.55"
MARK_MASTER = "| 2026-11-16 | **v2.55** |"
MARK_INDEX = "v2.55 第 96 轮"
MARK_MEMORY = "v2.55 第 96 轮"
MARK_HANDOVER = "v2.55 追加（第 96 轮：三条新互证审计面）"
MARK_DAILY = "## v2.55 第 96 轮（本次运行"
MARK_E45DOC = "v2.55 追加（第 96 轮：三条新互证审计面）"


# --------------------------------------------------------------------------
# table helpers
# --------------------------------------------------------------------------
def split_row(line):
    """Split a markdown table row on UNESCAPED pipes only.

    A cell may contain an escaped pipe (``max\\|dev\\|``); splitting on every
    pipe would tear that cell apart and the rebuilt row would have the wrong
    number of columns -- which breaks the whole table on render.
    """
    cells, cur, i = [], [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            cur.append(line[i:i + 2])
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return cells


def row(*cells):
    """One markdown table row from its cells (separators inserted by code)."""
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def esc(text):
    """Escape pipes inside a table cell (MASTER's own convention)."""
    return text.replace("|", "\\|")


def ncols(line):
    """Number of real cells (excluding the two border pieces)."""
    return len(split_row(line)) - 2


def read(path):
    """Read preserving the original line terminator convention."""
    raw = path.read_bytes()
    return raw.decode("utf-8"), (b"\r\n" in raw)


def write(path, text, crlf):
    out = text.replace("\r\n", "\n")
    if crlf:
        out = out.replace("\n", "\r\n")
    path.write_bytes(out.encode("utf-8"))


def split_lines(text):
    return text.replace("\r\n", "\n").split("\n")


def find_line(lines, prefix, label):
    hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(
            f"LINE ANCHOR MISS ({label}): prefix {prefix[:60]!r} matched {len(hits)} line(s), expected 1"
        )
    return hits[0]


def patch_row(lines, prefix, mutate, label):
    """Replace one table row: split into cells, let `mutate` edit them, rebuild.

    The cell count is asserted before and after, so a mutation that changes
    the column count is an error rather than a broken table.
    """
    i = find_line(lines, prefix, label)
    cells = split_row(lines[i])
    before = len(cells) - 2
    new_cells = mutate(cells)
    after = len(new_cells) - 2
    if before != after:
        raise SystemExit(
            f"CELL COUNT CHANGED ({label}): {before} -> {after} columns; the table would break"
        )
    lines[i] = row(*new_cells[1:-1])
    return lines


# --------------------------------------------------------------------------
# the round-96 statement, written once
# --------------------------------------------------------------------------
SURFACES = (
    "**三条新互证审计面**（前 95 轮都未测过）："
    "① **两个独立探针互相印证「拓扑支结掉」**——`diag_probe` 的 `field_max_mm` 六格全 **402.4288 mm**"
    "（＝`expm1(6)` 的模型硬上限），而 `struct_metrics`（另一套实现 `tools/_e41_struct_metrics.py`）"
    "报的**帧中位数**场最大为 **A3a 65.3／75.2／96.1 mm、A3b 102.2／166.8／202.0 mm**，"
    "在**每一个种子**都远高于所有非拓扑臂（≤ **9.6** mm）⇒ **针尖不是单帧孤例，而是整条分布的上尾**；"
    "② **E45 的 B0 记录与别的实验逐位相同**——E43／E44 的 `B0_s0_ep00`、`B0_s1_ep00` 的 "
    "`csi_at_2.0`／`bias_at_2.0` 与 E45 **3/3 全等**（`0.12181703042952949`／`0.09169885007029324`）"
    "⇒ 交接文档里「B0 与 E43/E44 逐位相同」的登记**已实测确认**（不是拿 E45 自己的副本自证）；"
    "③ **两条独立下载批次的 21 份主跑评测逐位相同 21/21**（`raw_audit/` 与 `epoch_axis/` 各自独立取得）"
    "⇒ 已发布头条表所依赖的采集**可复现**。"
)

REMOTE = (
    "**远端状态复核（只读查询，未 sbatch、未同步、未下载）**：E45 主链终态不变"
    "（提交器 273／训练数组 275·283／评测 274·282·290／汇总 291／297·298／314–317／343 "
    "全 `COMPLETED`）、`squeue -u wf` **空**、21 份 `.eval.json` ＋ **14** 份 run 级 json 齐；"
    "**结案后（> 2026-11-11 14:20:42）被修改的文件仅 1 份＝作业 343 自己的新产物 "
    "`s3_seedrobust.json`**（已发布产物零改动）；"
    "**14 份产物远端↔本地 md5 14/14 逐位全同**（本轮实测）；"
    "`sacct -S 2026-11-12` 主链 **0 个** e45-train／e45-stage2／e45-submit ⇒ **主链从未重启**；"
    "`runs/` 下**无 E46+ 正式训练目录**；**S3 由 summary 291 当年自动跑完 ⇒ 本轮无需手动 sbatch**。"
)

AUDIT = (
    "**裁决独立重导**（`tools/_e45_round96.py`，只读本地，**POS 22/22 VERDICT: PASS**）："
    "① 头条表由 **21 份主跑 `*.eval.json`** 的 `thresholds[\"2.0\"].csi` 复算 ⇒ 七臂中位数与过地板通过率"
    "与已发布值 **max|dev| = 0.000e+00**（对照 B0 中位数 0.121817／**2/3** 过地板；"
    "A3a 0.110015、A3b 0.116055 虽各自 **3/3** 过地板，但**中位数均低于同批对照 B0** ⇒ 不晋级；"
    "超 Oracle 上界 0.1653 的仍只有 **A3b／A4** 各 1/3 seed ⇒ 过预报、须 S3 才判）；"
    "② 机制判据逐格重判与已发布**一致**（门指向 A1／A2 **3/3** 过、拓扑两臂 **0/3** 过、"
    "量通道 A4 **3/3**／A4b **2/3** 过；六格 `scale_cv` 最小 **1.2358** ＝ 底线 24.7 倍 ⇒ "
    "**全程没有任何一格真的退化成常数**；`mean(out)=T` 仍 **4/6 格不达** ⇒ "
    "**只记实现缺陷、不当机制结论**）；"
    "③ S3 **数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，**不读存储布尔**）⇒ **12/12 区间跨零**、"
    "失效模式**分两支报**（`GAIN_WAS_AREA_OR_AMPLITUDE` 4 臂／`NO_GAIN_TO_BEGIN_WITH` 2 臂）、"
    "六臂 `max_abs_bias_gap` 全 ≤4.1e-07；"
    "④ 种子稳健性扫读 `s3_seedrobust.json` ⇒ **两侧各 6 臂 × 20 个种子 = 240/240 格全含零**、"
    "按预注册晋级判据（点估 > 0 且区间不含零）**合格种子数 = 0**、注册种子**逐位复现**。"
)

SELFCHECK = (
    "**本轮自身失误与修正（记录以防重犯）**：新写的审计脚本首跑报 **21/21 全绿，但其中两条断言是空的**"
    "——① 种子稳健性按 `equal_bias_sweep`／`equal_bias_reproduced` 取键，"
    "而文件里**等偏置半边实际存在 `sweep`／`reproduced`** ⇒ 拿到的是 **0 个**区间，"
    "于是「20/20 全含零」退化成了 **0 == 0 的空断言**；"
    "② B0 互证用 glob 扫全树，只匹配到 **E45 自己的 7 份副本**，等于拿自己证明自己。"
    "⇒ 两条都已改成**指名道姓**的检查（键名先与文件内容对照确认，并加「每半边各 20 格非空 ＋ "
    "两半边各自与其已发布记录点估计相同」的归属断言；B0 互证改为**直接读** "
    "`e43_scores_summary.json`／`e44_scores_summary.json` 的 `B0_s0_ep00`／`B0_s1_ep00`）。"
    "⇒ **纪律：新增审计项必须先核对原始布局再写断言；「全绿」不等于「查到了东西」——"
    "空断言也是全绿，必须显式断言被查对象非空、且逐条核对断言对象与被查对象是不是同一个东西。**"
)

VERDICT = "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。"

BODY = REMOTE + SURFACES + AUDIT + SELFCHECK + VERDICT

# short form for the MASTER header cell
HEAD_SHORT = (
    "远端主链终态不变（273／275·283／274·282·290／291／297·298／314–317／343 全 `COMPLETED`、"
    "`squeue -u wf` 空、**14 份产物 md5 14/14 全同**、结案后仅作业 343 自己的新产物被写入）；"
    "**三条新互证审计面**（两个独立探针互证拓扑支结掉／E45 的 B0 与 E43·E44 逐位相同／"
    "两条独立下载批次的 21 份主跑评测逐位相同 21/21）；"
    "**裁决不变＝六臂判负、不投十 seed 算力**。"
)

# --------------------------------------------------------------------------
# MASTER
# --------------------------------------------------------------------------
HEADER_PREFIX = "| 最后更新 |"
CARD_PREFIX = "| §4.47 | 修三处机制缺陷"
S9_PREFIX = "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |"
CHANGELOG_V54_PREFIX = "| 2026-11-16 | **v2.54** |"

# the stale current-state count (the document contradicted its own v2.54 row)
STALE_COUNT_OLD = "`tools/_evals/e45/`（13 份核心 json ＋ round46–68 诊断产物）"
STALE_COUNT_NEW = "`tools/_evals/e45/`（**14** 份核心 json ＋ round46–68 诊断产物）"

HEADER_TAG_OLD = "**v2.54（2026-11-16，最新）**"
HEADER_TAG_NEW = "**v2.54（2026-11-16）**"
HEADER_NEW_TAG = "**v2.55（2026-11-16，最新）**"


def mutate_header(cells):
    content = cells[-2].strip()
    if content.count(HEADER_TAG_OLD) != 1:
        raise SystemExit(f"HEADER TAG ANCHOR MISS: {content.count(HEADER_TAG_OLD)} occurrences")
    content = content.replace(HEADER_TAG_OLD, HEADER_TAG_NEW, 1)
    # append the v2.55 summary to the end of the content cell
    tail = "**裁决不变＝六臂判负、不投十 seed 算力**。"
    if not content.endswith(tail):
        raise SystemExit("HEADER TAIL ANCHOR MISS")
    content = content[: -len(tail)] + "；" + HEADER_NEW_TAG + "：" + esc(HEAD_SHORT) + tail
    out = list(cells)
    out[-2] = content
    return out


def mutate_card(cells):
    # the last two real cells are the prose cell and the reference cell;
    # both carry a trailing space before the border pipe
    prose = cells[-3].strip()
    refs = cells[-2].strip()
    old_ref = "v2.54 见 §4.48 状态行"
    if refs.count(old_ref) != 1:
        raise SystemExit(f"CARD REF ANCHOR MISS: {refs.count(old_ref)}")
    refs = refs.replace(old_ref, old_ref + "；v2.55 见 §4.48 状态行", 1)
    tail = "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。"
    if not prose.endswith(tail):
        raise SystemExit("CARD PROSE TAIL ANCHOR MISS")
    prose = prose[: -len(tail)] + " ｜ **v2.55**：" + esc(BODY) + tail
    out = list(cells)
    out[-3] = prose
    out[-2] = refs
    return out


def mutate_s9(cells):
    last = cells[-2].strip()
    old = "v2.54 见 §4.48 状态行（**最小同步集订正为 5 个文件** ＋ **A3a 梯度读数 1/3 过**）"
    if last.count(old) != 1:
        raise SystemExit(f"S9 REF ANCHOR MISS: {last.count(old)}")
    new = old + (
        "；v2.55 见 §4.48 状态行（**三条新互证审计面** ＋ **MASTER 内部矛盾已修：产物计数 13→14**）"
    )
    out = list(cells)
    out[-2] = last.replace(old, new, 1)
    return out


CHANGELOG_ROW = row(
    "2026-11-16",
    f"**{TAG}**",
    "**E45 第 96 轮：远端状态复核 ＋ 三条新互证审计面 ＋ MASTER 内部矛盾（产物计数 13→14）已修**"
    "：" + esc(REMOTE) + esc(SURFACES) + esc(AUDIT) + esc(SELFCHECK) + esc(VERDICT),
    "`tools/_e45_round96.py` · `tools/_evals/e45/round96_status_audit.json` · "
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md`",
)


def patch_master(text):
    if MARK_MASTER in text:
        print("MASTER already carries the v2.55 changelog row -> skip")
        return text
    lines = split_lines(text)
    # 1. the stale current-state count (a current-state claim, not an audit-trail row)
    hits = [i for i, l in enumerate(lines) if STALE_COUNT_OLD in l]
    if len(hits) != 1:
        raise SystemExit(f"STALE COUNT ANCHOR MISS: matched {len(hits)} lines, expected 1")
    lines[hits[0]] = lines[hits[0]].replace(STALE_COUNT_OLD, STALE_COUNT_NEW, 1)
    # 2. header version line
    lines = patch_row(lines, HEADER_PREFIX, mutate_header, "master.header")
    # 3. the section 4.47 card
    lines = patch_row(lines, CARD_PREFIX, mutate_card, "master.card")
    # 4. the section 9 row
    lines = patch_row(lines, S9_PREFIX, mutate_s9, "master.s9")
    # 5. the section 12 changelog row (insert after the v2.54 row)
    i = find_line(lines, CHANGELOG_V54_PREFIX, "master.changelog.v54")
    prev = ncols(lines[i])
    new = ncols(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"CELL COUNT MISMATCH: v2.54 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# INDEX
# --------------------------------------------------------------------------
INDEX_PREFIX = "| E45 | §4.47（v2.49 压缩总览） |"
#: the INDEX E45 row's last cell ENDS with the verdict (no trailing pipe
#: inside the cell -- the pipe belongs to the row's border), so the anchor
#: is the verdict text itself and the new text is appended after it.
INDEX_OLD = "**裁决不变＝六臂判负、不投十 seed 算力**"
INDEX_NEW = (
    "**裁决不变＝六臂判负、不投十 seed 算力** ｜ **v2.55 第 96 轮**："
    "远端主链终态不变（全部 `COMPLETED`、`squeue -u wf` 空、**14 份产物 md5 14/14 全同**、"
    "结案后仅作业 343 自己的新产物被写入）；**三条新互证审计面**"
    "（两个独立探针互证拓扑支结掉／E45 的 B0 与 E43·E44 逐位相同／"
    "两条独立下载批次的 21 份主跑评测逐位相同 21/21）；"
    "**裁决不变＝六臂判负、不投十 seed 算力**"
)


def mutate_index(cells):
    # the cell carries a trailing space before the border pipe, so the
    # anchor is matched against the stripped cell
    last = cells[-2].strip()
    if not last.endswith(INDEX_OLD):
        raise SystemExit("INDEX TAIL ANCHOR MISS")
    out = list(cells)
    out[-2] = last[: -len(INDEX_OLD)] + INDEX_NEW
    return out


def patch_index(text):
    if MARK_INDEX in text:
        print("INDEX already carries", MARK_INDEX, "-> skip")
        return text
    lines = split_lines(text)
    lines = patch_row(lines, INDEX_PREFIX, mutate_index, "index.e45")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# E45 experiment file
# --------------------------------------------------------------------------
E45DOC_PTR_OLD = "> **2026-11-16（v2.54）**："
E45DOC_PTR_NEW = (
    "> **2026-11-16（v2.55）**：**E45 内容与判据自 v2.49 起未变**（S0–S6 一字未改；"
    "裁决不变：六臂判负、不投十 seed 算力）；第 96 轮远端状态复核无新活动，"
    "并新增**三条互证审计面**：① 两个独立探针互相印证拓扑支结掉"
    "（`diag_probe` 场最大值六格全 402.4288 mm＝模型上限，`struct_metrics` 的**帧中位数**场最大 "
    "A3a 65.3–96.1／A3b 102.2–202.0 mm 在每个种子都远高于所有非拓扑臂的 ≤9.6 mm ⇒ "
    "**针尖不是单帧孤例**）；② E45 的 B0 记录与 E43／E44 的 B0 记录**逐位相同（3/3）** "
    "⇒ 本文登记的「B0 与 E43/E44 逐位相同」**已实测确认**；③ 两条独立下载批次的 21 份主跑评测"
    "**逐位相同 21/21**。**裁决不变＝六臂判负、不投十 seed 算力**（详见 §8.4 结案段与交接文档）。\n"
    "> **2026-11-16（v2.54）**："
)

E45DOC_NOTE_OLD = (
    "**第五个（`scripts/evaluate.py`）是承重的**：它是把 `clamp_mm` 带到打分时刻、"
    "且**唯一无法从 checkpoint 恢复**（`clamp_mm` 不进 `state_dict`）的那一半。"
)
E45DOC_NOTE_NEW = (
    E45DOC_NOTE_OLD
    + "\n  **v2.55 追加（第 96 轮：三条新互证审计面）**：本轮不改动任何判据、任何臂、任何数字，"
    "只新增三条**互证**：① **两个独立探针互相印证拓扑支结掉**——`diag_probe` 的场最大值六格全 "
    "402.4288 mm（＝`expm1(6)` 模型上限），而 `struct_metrics`（另一套实现）报的**帧中位数**场最大 "
    "A3a 65.3／75.2／96.1 mm、A3b 102.2／166.8／202.0 mm，在**每个种子**都远高于所有非拓扑臂"
    "（≤9.6 mm）⇒ **针尖不是单帧孤例，而是整条分布的上尾**；"
    "② **E45 的 B0 记录与别的实验逐位相同**——E43／E44 的 `B0_s0_ep00`、`B0_s1_ep00` 的 "
    "`csi_at_2.0`／`bias_at_2.0` 与 E45 **3/3 全等** ⇒ 本文登记的「B0 与 E43/E44 逐位相同」"
    "**已实测确认**；③ 两条独立下载批次的 21 份主跑评测**逐位相同 21/21**。"
    "**三件待办本身不变**（① 训练侧还需接 `--model-kwargs {\"clamp_mm\":20.0}` 一次；"
    "② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）。"
)


def patch_e45doc(text):
    if MARK_E45DOC in text:
        print("E45DOC already carries the marker -> skip")
        return text
    if text.count(E45DOC_PTR_OLD) != 1:
        raise SystemExit(f"E45DOC POINTER ANCHOR MISS: {text.count(E45DOC_PTR_OLD)}")
    text = text.replace(E45DOC_PTR_OLD, E45DOC_PTR_NEW, 1)
    if text.count(E45DOC_NOTE_OLD) != 1:
        raise SystemExit(f"E45DOC NOTE ANCHOR MISS: {text.count(E45DOC_NOTE_OLD)}")
    text = text.replace(E45DOC_NOTE_OLD, E45DOC_NOTE_NEW, 1)
    return text


# --------------------------------------------------------------------------
# MEMORY.md
# --------------------------------------------------------------------------
MEM_PTR_OLD = "> 权威 `docs/MASTER.md` **v2.54**（"
MEM_PTR_NEW = "> 权威 `docs/MASTER.md` **v2.55**（"
MEM_PTR_TAIL_OLD = "**最小同步集订正为 5 个文件**）；"
MEM_PTR_TAIL_NEW = "**最小同步集订正为 5 个文件**、**三条新互证审计面**）；"

MEM_STATUS_OLD = (
    "**A3a 梯度读数 1/3 过**（0.4782，种子 2，与已发布一致）但 A3a 自身触发条件 "
    "`tail_max_over_p99 > 10` 3/3 全过、场最大值 0/3 ⇒ **拓扑支照旧结掉**。"
    "**裁决不变＝六臂判负、不投十 seed 算力**；待办清单本身**不变**。"
)
MEM_STATUS_NEW = (
    MEM_STATUS_OLD
    + " **v2.55 第 96 轮（2026-11-16）**：远端无新活动（主链全 `COMPLETED`、`squeue -u wf` 空、"
    "**14 份产物 md5 14/14 全同**、结案后仅作业 343 自己的新产物被写入）；"
    "**三条新互证审计面**：① **两个独立探针互相印证拓扑支结掉**（`diag_probe` 场最大值六格全 "
    "402.4288 mm＝模型上限，`struct_metrics` 的**帧中位数**场最大 A3a 65.3–96.1／"
    "A3b 102.2–202.0 mm 在每个种子都远高于所有非拓扑臂的 ≤9.6 mm ⇒ **针尖不是单帧孤例**）；"
    "② **E45 的 B0 记录与 E43／E44 逐位相同（3/3）** ⇒ 交接文档的登记**已实测确认**；"
    "③ 两条独立下载批次的 21 份主跑评测**逐位相同 21/21**。"
    "**裁决不变＝六臂判负、不投十 seed 算力**；待办清单本身**不变**。"
)

MEM_R96_NEW = (
    "- **E45 第 96 轮（2026-11-16）**：远端主链终态不变（273／275·283／274·282·290／291／"
    "297·298／314–317／343 全 `COMPLETED`）、`squeue -u wf` **空**、"
    "**14 份产物远端↔本地 md5 14/14 全同**（本轮实测）、"
    "**结案后被修改的文件仅 1 份＝作业 343 自己的新产物 `s3_seedrobust.json`**、"
    "`sacct -S 2026-11-12` 主链 0 个作业 ⇒ **主链从未重启**；**远端动作＝0**（只读查询）。"
    "独立重导（`tools/_e45_round96.py`，只读本地，**POS 22/22 VERDICT: PASS**）：头条表 max|dev| = 0、"
    "机制判据逐格与已发布一致、S3 **12/12 跨零**、种子扫描 **240/240 全含零**、合格种子数 0。"
    "**三条新互证审计面**（前 95 轮都未测过）：① **两个独立探针互相印证拓扑支结掉**"
    "（`diag_probe` 场最大值六格全 402.4288 mm＝模型上限；`struct_metrics` 的**帧中位数**场最大 "
    "A3a 65.3–96.1／A3b 102.2–202.0 mm 在每个种子都远高于所有非拓扑臂的 ≤9.6 mm ⇒ "
    "**针尖不是单帧孤例，而是整条分布的上尾**）；② **E45 的 B0 记录与 E43／E44 逐位相同（3/3）** ⇒ "
    "交接文档登记的「B0 与 E43/E44 逐位相同」**已实测确认**（不是拿 E45 自己的副本自证）；"
    "③ 两条独立下载批次的 21 份主跑评测**逐位相同 21/21**。"
    "**裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**；"
    "下一步仍是 §4.5 三件实现缺陷修复 ＋ **重开须重新提机制族**。"
)


def patch_memory(text):
    if MARK_MEMORY in text:
        print("MEMORY already carries", MARK_MEMORY, "-> skip")
        return text
    for old, new, label in [
        (MEM_PTR_OLD, MEM_PTR_NEW, "memory.pointer"),
        (MEM_PTR_TAIL_OLD, MEM_PTR_TAIL_NEW, "memory.pointer.tail"),
        (MEM_STATUS_OLD, MEM_STATUS_NEW, "memory.status"),
    ]:
        if text.count(old) != 1:
            raise SystemExit(f"{label} ANCHOR MISS: {text.count(old)}")
        text = text.replace(old, new, 1)
    lines = split_lines(text)
    r95 = [i for i, l in enumerate(lines) if l.startswith("- **E45 第 95 轮（2026-11-16）**")]
    if len(r95) != 1:
        raise SystemExit(f"MEMORY: expected exactly 1 round-95 bullet, found {len(r95)}")
    lines.insert(r95[0] + 1, MEM_R96_NEW)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# HANDOVER_E45.md
# --------------------------------------------------------------------------
HO_PTR_OLD = "Gate 3 写回已完成（MASTER **v2.54**）。"
HO_PTR_NEW = "Gate 3 写回已完成（MASTER **v2.55**）。"

HO_NOTE_OLD = "**第五个是承重的**（评测侧复原＝把 `clamp_mm` 带到打分时刻、且唯一无法从 checkpoint 恢复的一半）。"
HO_NOTE_NEW = (
    HO_NOTE_OLD
    + "\n\n**v2.55 追加（第 96 轮：三条新互证审计面）**：远端状态复核无新活动（主链全 `COMPLETED`、"
    "`squeue -u wf` 空、**14 份产物 md5 14/14 全同**、结案后仅作业 343 自己的新产物被写入）⇒ "
    "**裁决不变＝六臂判负、不投十 seed 算力**。"
    "**三条新互证审计面**（前 95 轮都未测过）：① **两个独立探针互相印证拓扑支结掉**——"
    "`diag_probe` 的场最大值六格全 **402.4288 mm**（＝`expm1(6)` 模型上限），"
    "而 `struct_metrics`（另一套实现 `tools/_e41_struct_metrics.py`）报的**帧中位数**场最大为 "
    "**A3a 65.3／75.2／96.1 mm、A3b 102.2／166.8／202.0 mm**，在**每一个种子**都远高于所有非拓扑臂"
    "（≤ **9.6** mm）⇒ **针尖不是单帧孤例，而是整条分布的上尾**；"
    "② **E45 的 B0 记录与别的实验逐位相同**——E43／E44 的 `B0_s0_ep00`、`B0_s1_ep00` 的 "
    "`csi_at_2.0`／`bias_at_2.0` 与 E45 **3/3 全等** ⇒ 本文登记的「B0 与 E43/E44 逐位相同」"
    "**已实测确认**；③ 两条独立下载批次的 21 份主跑评测**逐位相同 21/21**。"
    "**三件待办本身不变。**"
)


def patch_handover(text):
    if MARK_HANDOVER in text:
        print("HANDOVER already carries the marker -> skip")
        return text
    for old, new, label in [
        (HO_PTR_OLD, HO_PTR_NEW, "handover.pointer"),
        (HO_NOTE_OLD, HO_NOTE_NEW, "handover.note"),
    ]:
        if text.count(old) != 1:
            raise SystemExit(f"{label} ANCHOR MISS: {text.count(old)}")
        text = text.replace(old, new, 1)
    return text


# --------------------------------------------------------------------------
# round note (daily logs + handover)
# --------------------------------------------------------------------------
ROUND_NOTE = """
<!-- E45-R96-BEGIN -->
### 2026-11-16｜第 96 轮（E45 远端状态复核 ＋ **三条新互证审计面** ＋ **两处自身空断言修正**）｜远端动作＝0，裁决不变

- **远端终态（远端动作＝0）**：权威链 273／275·283（训练数组 7/7+7/7）／274·282·290（stage2）／291（summary）／297·298／314／315·316·317／343 全 `COMPLETED`；**`squeue -u wf` 空**（上一轮队列里那 8 个 `qlfn2-f38` 属同一 `wf` 账号下**别的项目线**，本轮已全部出队，仍不干预）；21 份 `.eval.json` ＋ **14 份** run 级 json 齐；**结案后（> 11-11 14:20:42）被修改的文件仅 1 份＝作业 343 自己的新产物 `s3_seedrobust.json`**（已发布产物零改动）；**14 份产物远端↔本地 md5 14/14 逐位全同**（本轮实测）；`sacct -S 2026-11-12` 主链 **0 个** e45-train／e45-stage2／e45-submit ⇒ **主链从未重启**；`runs/` 下**无 E46+ 正式训练目录**。**S3 由 summary 291 当年自动跑完 ⇒ 本轮无需手动 sbatch**（未 sbatch／未同步／未下载／未改远端状态）。
- **裁决独立重导**（`tools/_e45_round96.py`，只读本地，**POS 22/22 VERDICT: PASS**）：① 头条表由 **21 份主跑 `*.eval.json`** 的 `thresholds["2.0"].csi` 复算 ⇒ 七臂中位数与过地板通过率与已发布值 **max|dev| = 0.000e+00**（对照 B0 中位数 0.121817／**2/3** 过地板；A3a 0.110015、A3b 0.116055 虽各自 **3/3** 过地板，但**中位数均低于同批对照 B0** ⇒ 不晋级；超 Oracle 上界 0.1653 的仍只有 **A3b／A4** 各 1/3 seed ⇒ 过预报、须 S3 才判）；② 机制判据逐格重判与已发布**一致**（门指向 A1／A2 **3/3** 过、拓扑两臂 **0/3** 过、量通道 A4 **3/3**／A4b **2/3** 过；六格 `scale_cv` 最小 **1.2358** ＝ 底线 24.7 倍 ⇒ **全程没有任何一格真的退化成常数**；`mean(out)=T` 仍 **4/6 格不达** ⇒ **只记实现缺陷、不当机制结论**）；③ S3 **数值级重算**（`ci_lo ≤ 0 ≤ ci_hi`，**不读存储布尔**）⇒ **12/12 区间跨零**、失效模式**分两支报**（`GAIN_WAS_AREA_OR_AMPLITUDE` 4 臂／`NO_GAIN_TO_BEGIN_WITH` 2 臂）、六臂 `max_abs_bias_gap` 全 ≤4.1e-07；④ 种子稳健性扫读 `s3_seedrobust.json` ⇒ **两侧各 6 臂 × 20 个种子 = 240/240 格全含零**、按预注册晋级判据（点估 > 0 且区间不含零）**合格种子数 = 0**、注册种子**逐位复现**。
- **三条新互证审计面**（前 95 轮都未测过）：① **两个独立探针互相印证「拓扑支结掉」**——`diag_probe` 的 `field_max_mm` 六格全 **402.4288 mm**（＝`expm1(6)` 的模型硬上限），而 `struct_metrics`（另一套实现 `tools/_e41_struct_metrics.py`）报的**帧中位数**场最大为 **A3a 65.3／75.2／96.1 mm、A3b 102.2／166.8／202.0 mm**，在**每一个种子**都远高于所有非拓扑臂（≤ **9.6** mm）⇒ **针尖不是单帧孤例，而是整条分布的上尾**；② **E45 的 B0 记录与别的实验逐位相同**——E43／E44 的 `B0_s0_ep00`、`B0_s1_ep00` 的 `csi_at_2.0`／`bias_at_2.0` 与 E45 **3/3 全等**（`0.12181703042952949`／`0.09169885007029324`）⇒ 交接文档里「B0 与 E43/E44 逐位相同」的登记**已实测确认**（不是拿 E45 自己的副本自证）；③ **两条独立下载批次的 21 份主跑评测逐位相同 21/21**（`raw_audit/` 与 `epoch_axis/` 两个目录各自独立取得）⇒ 已发布头条表所依赖的采集**可复现**。
- **本轮发现并修掉的 MASTER 内部矛盾**：MASTER 自己的 §12 v2.54 行写的是「**14 份**产物 md5 14/14 全同」，但 §4.47 的「权威链接」一条仍写着「**13 份**核心 json」⇒ **同一份文件自相矛盾**（第 94 轮新增第 14 份产物时漏改）。本轮把该**当前态**计数订正为 14；**历史逐轮记录（每轮当时观测到的计数）一律不动**——它们是审计链。
- **文档闸门**：`_check_tables` **PASS**（MASTER／INDEX／E45 实验文件 mismatched=0）、`_check_v245_row` **PASS**、`_e45_docs_numbers` **PASS**、`_check_v254_row` **POS 30/30 PASS**；`_check_v248/251/252/253_row` 的失败项**逐条核对后确认全部属 SNAPSHOT 类、按设计过期**（版本指针已由 v2.48 前推到 **v2.54**、v2.48 changelog 行后面不再是空行；**CONTENT 类失败 = 0**）。
- **裁决零变化**（六臂仍全部判负、无一臂进十 seed、**不投十 seed 算力**）；**下一步（不变）**：§4.5 三件实现缺陷修复（① 训练侧再接一次 `--model-kwargs {"clamp_mm":20.0}`；② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ **重开须重新提机制族**。
- **本轮自身失误与修正（记录以防重犯）**：新写的审计脚本首跑报 **21/21 全绿，但其中两条断言是空的**——① 种子稳健性按 `equal_bias_sweep`／`equal_bias_reproduced` 取键，而文件里**等偏置半边实际存在 `sweep`／`reproduced`** ⇒ 拿到的是 **0 个**区间，于是「20/20 全含零」退化成了 **0 == 0 的空断言**；② B0 互证用 glob 扫全树，只匹配到 **E45 自己的 7 份副本**，等于拿自己证明自己。⇒ 两条都已改成**指名道姓**的检查：键名先与文件内容对照确认（并加「每半边各 20 格非空 ＋ 两半边各自与其已发布记录点估计相同」的归属断言），B0 互证改为**直接读** `e43_scores_summary.json`／`e44_scores_summary.json` 的 `B0_s0_ep00`／`B0_s1_ep00`。⇒ **纪律：新增审计项必须先核对原始布局再写断言；「全绿」不等于「查到了东西」——空断言也是全绿，必须显式断言被查对象非空、且逐条核对断言对象与被查对象是不是同一个东西。**
- **产物**：`tools/_e45_round96.py`、`tools/_evals/e45/round96_status_audit.json`、`tools/_sync_v255.py`。
<!-- E45-R96-END -->
"""


def append_daily(text):
    if MARK_DAILY in text:
        print("daily log already carries", MARK_DAILY, "-> skip")
        return text
    note = (
        f"\n\n## {TAG} 第 96 轮（本次运行，2026-11-16）· E45 状态复核 ＋ "
        "**三条新互证审计面** ＋ **MASTER 内部矛盾（产物计数 13→14）已修**\n\n" + ROUND_NOTE
    )
    return text + note


def append_handover_round(text):
    if "<!-- E45-R96-BEGIN -->" in text:
        print("HANDOVER already carries the R96 round note -> skip")
        return text
    return text + ROUND_NOTE


PLAN = [
    (MASTER, patch_master),
    (INDEX, patch_index),
    (E45DOC, patch_e45doc),
    (MEMORY, patch_memory),
    (HANDOVER, patch_handover),
    (HANDOVER, append_handover_round),
    (DAILY, append_daily),
    (DAILY_PROJECT, append_daily),
]


def main():
    print("pre-flight: checking every anchor")
    for path, fn in PLAN:
        if not path.exists():
            raise SystemExit(f"FILE MISSING: {path}")
        original, _ = read(path)
        try:
            fn(original)
        except SystemExit as exc:
            raise SystemExit(f"PRE-FLIGHT FAILED for {path}:\n{exc}")
        print(f"  ok: {path.name}")

    print("\nwriting:")
    changed = []
    for path, fn in PLAN:
        text, crlf = read(path)
        new = fn(text)
        if new != text:
            write(path, new, crlf)
            changed.append(str(path))
            print(f"changed: {path}")
        else:
            print(f"unchanged: {path}")
    print()
    print(f"{len(changed)} file(s) changed")
    if not changed:
        print("(idempotent: nothing to do)")


if __name__ == "__main__":
    sys.exit(main())
