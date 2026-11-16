"""Write back the E45 document-number audit and the correction it found (v2.47).

What v2.47 adds
---------------
Rounds 9-34 re-verified E45's ARTEFACTS and the WORDING of its documents.
None of them ever checked the NUMBERS that those documents print.  This
round did, and it closed that gap -- and the gap was not empty.

The audit (two new read-only tools)
------------------------------------
1. ``tools/_e45_recompute_raw.py`` runs on the remote and rebuilds the
   three-seed table directly from the 21 evaluation reports, with its own
   parsing, then compares it with the stored ``scores_summary.json``.
   Result: every arm's median CSI@2.0, median BIAS@2.0, per-seed values,
   pass rate and over-Oracle count agree -- VERDICT: PASS.

2. ``tools/_e45_docs_numbers.py`` parses the three tables of the E45
   experiment file (the three-seed score table, the seed-0 score table and
   the S3 equal-bias table) and compares every cell with the recorded
   data; it also scans MASTER.md, INDEX.md and HANDOVER_E45.md for a
   number that is quoted as a median next to an E45 arm and contradicts
   the recorded table.

The one defect found
--------------------
The seed-0 score table (E45 experiment file, section 8.2) prints A4's
``CSI@2.0`` as **0.175501**.  The authoritative value -- read both from
the remote evaluation report ``runs/e45/evals/A4_s0_ep00.eval.json`` and
from the stored ``scores_summary_s0.json`` -- is

    0.1755001059215604

which rounds to **0.175500** at six decimals.  So the table carries a
one-digit error in its last place.  The same digit had been copied into
the handover's seed-0 table, and is therefore corrected there too.

The error changes NOTHING, and that is a claim, not an assertion:

* section 8.2 says in its own heading that a single seed does not
  constitute a verdict ("单 seed，不构成裁决");
* the three-seed table -- which IS the basis of the verdict -- was
  recomputed from the raw reports this round and every cell matches;
* the pass rates and the S3 equal-bias verdict are unchanged;
* the mechanism read-outs (gate correlation, gradient share, field
  maximum, scale channel) are unchanged.

Why this is written back at all
-------------------------------
Because the whole point of E45's "finding four" is that a number in this
project must be traceable to an artefact.  A digit that does not match
its artefact is precisely the failure the project exists to prevent, and
it propagates: the same wrong digit had reached the handover, which is
the file every new session reads first.

Scope of this write-back
-------------------------
* MASTER.md -- header version line, section 4.47 card, section 9 row,
  section 12 changelog row (four places, Gate 3);
* INDEX.md -- the E45 row;
* the E45 experiment file -- the corrected digit plus a note at the table;
* MEMORY.md -- the version pointer and a new status entry;
* HANDOVER_E45.md -- the version pointer and the corrected digit;
* both daily logs -- the round note.

The 2026-11-11 daily log is deliberately NOT edited: daily logs are
append-only, and the correction is registered in this round's log, in
the MASTER changelog and in the handover instead.

Every replacement is anchored on a literal that must occur exactly the
expected number of times; a missing anchor is an error, never a silent
skip.  The new section 12 row is built from its CELLS; the separators
are inserted by the code, and the pipe count is compared against the row
it follows so the table cannot silently break.  Running the script twice
is a no-op.
"""
from pathlib import Path
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
DAILY = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md")
DAILY_PROJECT = Path(".workbuddy/memory/2026-11-13.md")

TAG = "v2.47"
OLD_TAG = "v2.46"


def row(*cells):
    """One markdown table row from its cells (separators inserted by code)."""
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def unescaped_pipes(line):
    return len(re.findall(r"(?<!\\)\|", line))


def apply(text, old, new, count=1, label=""):
    """Replace `old` with `new`, exactly `count` times (idempotent)."""
    n_old = text.count(old)
    n_new = text.count(new)
    if n_old == count:
        if n_new:
            raise SystemExit(f"BOTH ANCHORS PRESENT ({label}): old={n_old} new={n_new}")
        return text.replace(old, new)
    if n_new >= 1 and n_old == 0:
        return text  # already applied by an earlier run
    raise SystemExit(
        f"ANCHOR MISS ({label}): expected {count} occurrence(s), found {n_old}"
        f" (new text present {n_new}x)\n  {old[:90]!r}"
    )


# --------------------------------------------------------------------------
# the corrections, per file
# --------------------------------------------------------------------------

#: the one wrong digit, and the value that replaces it
BAD = "0.175501"
GOOD = "0.175500"

HEADER_OLD = (
    "2026-11-13（v2.46，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
    "没有一条进得了十 seed 正式跑**；v2.46 把三件待办的根因定位到源码层并订正其中两件的"
    "**错误因果表述**，裁决不变）"
)
HEADER_NEW = (
    "2026-11-13（v2.47，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
    "没有一条进得了十 seed 正式跑**；v2.47 首次把**文档里印的数字**与原始产物逐格对账，"
    f"查出并订正 §8.2 seed-0 表 A4 `CSI@2.0` 的一个末位笔误（{BAD} → {GOOD}），"
    "**裁决不变**）"
)

CARD_OLD = "**但 A1／A2 早已被 S3 等偏置对照判负，故此读数不改变任何裁决**。 |"
CARD_NEW = (
    "**但 A1／A2 早已被 S3 等偏置对照判负，故此读数不改变任何裁决**。"
    "**v2.47 数字对账**：本轮首次把**文档里印的数字**与原始产物逐格对账"
    "（远端从 21 份评测报告独立重算三 seed 表，再逐格核对 E45 文件的三张表）"
    f"⇒ 查出并订正 §8.2 seed-0 表 A4 `CSI@2.0` 的一个**末位笔误**（{BAD} → {GOOD}，"
    "真值 0.1755001059215604），**不构成裁决变化**"
    "（三 seed 中位数／过地板通过率／S3 六臂判负一个数都没动）。 |"
)

S9_OLD = (
    "**A4b 的量通道在第三个种子退化**（相关性 0.2789 < 0.3）"
    "——这条只对两个种子成立，不当机制结论。"
)
S9_NEW = (
    S9_OLD
    + "｜ **v2.47**：文档里的数字已与原始产物逐格对账，订正 §8.2 seed-0 表 A4 "
    f"`CSI@2.0` 的末位笔误（{BAD} → {GOOD}），**裁决不变**。"
)

INDEX_OLD = "⇒ **修好后须重读预注册的 `maxratio_as_trained > 0.9` 判据** |"
INDEX_NEW = (
    "⇒ **修好后须重读预注册的 `maxratio_as_trained > 0.9` 判据** ｜ "
    "**v2.47 数字对账**：文档里的数字已与原始产物逐格对账（远端从 21 份评测报告独立重算，"
    f"再逐格核对三张表）⇒ 查出并订正 §8.2 seed-0 表 A4 `CSI@2.0` 的末位笔误"
    f"（{BAD} → {GOOD}），**裁决不变** |"
)

E45_ROW_OLD = "| A4 | 0.175501 | 1.6131 | 0.10724 |"
E45_ROW_NEW = "| A4 | 0.175500 | 1.6131 | 0.10724 |"

E45_NOTE_OLD = (
    "地板 0.0924｜Oracle 上界 0.1653。B0 的 `CSI@2.0` 与 E43／E44 的 `B0_s0` "
    "**逐位相同**（可复现）。"
)
E45_NOTE_NEW = (
    E45_NOTE_OLD
    + f" **v2.47 订正**：上表 A4 的 `CSI@2.0` 原印 {BAD}，而原始评测报告 "
    "`runs/e45/evals/A4_s0_ep00.eval.json` 的真值是 0.1755001059215604"
    f"（四舍五入到六位为 **{GOOD}**）⇒ 已订正；**只影响这一个末位数字，不构成任何裁决**"
    "（本节标题已写明「单 seed，不构成裁决」，三 seed 分表与 S3 裁决均未变）。"
)

HANDOVER_ROW_OLD = (
    "| **A4** | **0.175501** | 1.6131 | 0.10724 | 超 Oracle 上界 0.1653；BIAS 爆表 |"
)
HANDOVER_ROW_NEW = (
    "| **A4** | **0.175500** | 1.6131 | 0.10724 | 超 Oracle 上界 0.1653；BIAS 爆表 |"
)

HANDOVER_FLOOR_OLD = "地板 **0.0924**（持久性）｜Oracle **0.1653**。"
HANDOVER_FLOOR_NEW = (
    HANDOVER_FLOOR_OLD
    + f" **v2.47 订正**：A4 的 `CSI@2.0` 原印 {BAD}，原始评测报告的真值是 "
    f"0.1755001059215604（六位为 {GOOD}）⇒ 已订正；**单 seed，不作裁决**。"
)

MEMORY_TAIL_OLD = "只新增一个只读复核脚本。）**"
MEMORY_TAIL_NEW = (
    "只新增一个只读复核脚本。）**\n\n"
    "**（2026-11-13 07:37 第 35 次独立复核：**远端 273/291 全 `COMPLETED`、队列空、"
    "21 份评测齐、提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；"
    "**远端状态与前三十四轮完全一致** ⇒ **六臂判负、不投十 seed 算力**不变。"
    "**本轮首次核对「文档里印的数字」本身**（前三十四轮只核产物与文档的措辞与结构）："
    "① 远端新脚本 `tools/_e45_recompute_raw.py` 从 21 份评测报告**独立重算**三 seed 表"
    "（七臂的中位数／三 seed 范围／逐 seed 值／过地板通过率／超 Oracle 次数）"
    "⇒ 与 `scores_summary.json` 逐项相同（VERDICT: PASS）；"
    "② 新脚本 `tools/_e45_docs_numbers.py` 把 E45 实验文件的**三张表逐格**与记录数据比对"
    "⇒ **查出唯一一处缺陷**：§8.2 seed-0 分表里 A4 的 `CSI@2.0` 印作 **0.175501**，"
    "而原始报告真值 0.1755001059215604（六位应为 **0.175500**）⇒ 已订正"
    "（同一笔误在 HANDOVER 的 seed-0 表也一并订正；2026-11-11 逐日日志按只增不改保留原样）。"
    "**该笔误不改变任何裁决**：三 seed 中位数、过地板通过率、S3 六臂判负、"
    "门指向／拓扑支／量通道三条机制线的读数**全部未变**。"
    "**新增可复用规则**：文档审计要**分两层**——先验措辞／结构／版本指针，再验印出来的数字；"
    "**前者过了不等于后者过**，而且后者的失真会以末位笔误的形式逐级传染到交接文档与逐日日志 ⇒ "
    "数字对账必须**从原始产物重新推导**，不能拿文档互相对照。）**"
)

MASTER_FIXES = [
    (HEADER_OLD, HEADER_NEW, 1, "master.header.version"),
    (CARD_OLD, CARD_NEW, 1, "master.card.e45"),
    (S9_OLD, S9_NEW, 1, "master.s9.row"),
]

INDEX_FIXES = [
    (INDEX_OLD, INDEX_NEW, 1, "index.e45.row"),
]

E45DOC_FIXES = [
    (E45_ROW_OLD, E45_ROW_NEW, 1, "e45doc.seed0.row"),
    (E45_NOTE_OLD, E45_NOTE_NEW, 1, "e45doc.seed0.note"),
]

MEMORY_FIXES = [
    ("> 权威 `docs/MASTER.md` v2.46；", "> 权威 `docs/MASTER.md` v2.47；", 1,
     "memory.header.version"),
    (MEMORY_TAIL_OLD, MEMORY_TAIL_NEW, 1, "memory.status.r35"),
]

HANDOVER_FIXES = [
    ("Gate 3 写回已完成（MASTER **v2.46**）。", "Gate 3 写回已完成（MASTER **v2.47**）。",
     1, "handover.pointer"),
    (HANDOVER_ROW_OLD, HANDOVER_ROW_NEW, 1, "handover.seed0.row"),
    (HANDOVER_FLOOR_OLD, HANDOVER_FLOOR_NEW, 1, "handover.seed0.note"),
]

# --------------------------------------------------------------------------
# new §12 changelog row (MASTER)
# --------------------------------------------------------------------------

CHANGELOG_ROW = row(
    "2026-11-13",
    f"**{TAG}**",
    "**E45 文档数字首次与原始产物逐格对账，查出并订正一处末位笔误（裁决不变）**："
    "远端状态与前三十四轮完全一致（提交器 273 与汇总作业 291 全 `COMPLETED`、"
    "21 份评测齐、队列空、13 份产物与本地 md5 全同）⇒ **六臂判负、不投十 seed 算力**不变。"
    "**本轮新增的检查**：前三十四轮只核产物与文档的**措辞与结构**，从未核过"
    "**文档里印出来的数字本身**；本轮补上这一环 —— ① 在远端用 "
    "`tools/_e45_recompute_raw.py` 从 21 份评测报告**独立重算**三 seed 表"
    "（七臂的中位数／三 seed 范围／逐 seed 值／过地板通过率／超 Oracle 次数），"
    "与 `scores_summary.json` **逐项相同**；② 用 `tools/_e45_docs_numbers.py` 把 "
    "E45 实验文件的**三张表（三 seed 分表、seed-0 分表、S3 等偏置表）逐格**与记录数据比对，"
    "并扫描 MASTER／INDEX／HANDOVER 的散文里与中位数相邻的数字。"
    f"**结果：唯一的缺陷是 §8.2 seed-0 分表里 A4 的 `CSI@2.0` 印作 {BAD}，"
    "而原始报告 `runs/e45/evals/A4_s0_ep00.eval.json` 的真值是 0.1755001059215604"
    f"（四舍五入到六位应为 {GOOD}）⇒ 一个末位笔误**，已订正"
    "（同一笔误已传到 HANDOVER 的 seed-0 表，一并订正；2026-11-11 的逐日日志按「只增不改」"
    "保留原样，订正登记在本行与本轮日志里）。"
    "**该笔误不构成任何裁决变化**：§8.2 标题已写明「单 seed，不构成裁决」，"
    "而三 seed 中位数、过地板通过率与 S3 六臂判负的裁决**一个数都没动**"
    "（本轮已从原始产物逐格重算确认）；门指向／拓扑支／量通道三条机制线的读数亦未变。"
    "**新增可复用规则**：**文档审计必须分两层** —— 先验「措辞／结构／版本指针」，"
    "再验「印出来的数字」；**前者过了不等于后者过**，且后者的失真会以末位笔误的形式"
    "**逐级传染**到交接文档与逐日日志 ⇒ 数字对账要**从原始产物重新推导**，"
    "而不是拿文档互相对照。",
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` · "
    "`tools/_e45_recompute_raw.py` · `tools/_e45_docs_numbers.py` · "
    "`tools/_check_v247_row.py`",
)


def patch_master(text):
    if TAG in text:
        print("MASTER already carries", TAG, "-> skip")
        return text
    for old, new, cnt, label in MASTER_FIXES:
        text = apply(text, old, new, cnt, label)
    # insert the v2.47 changelog row right after the v2.46 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-13 | **v2.46** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.46 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.46 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


def patch_index(text):
    if TAG in text:
        print("INDEX already carries", TAG, "-> skip")
        return text
    for old, new, cnt, label in INDEX_FIXES:
        text = apply(text, old, new, cnt, label)
    return text


def patch_plain(text, fixes, name, tag):
    if tag in text:
        print(name, "already carries", tag, "-> skip")
        return text
    for old, new, cnt, label in fixes:
        text = apply(text, old, new, cnt, label)
    return text


def append_daily(text):
    if TAG in text:
        print("daily log already carries", TAG, "-> skip")
        return text
    note = (
        f"\n\n## {TAG}（本次运行，2026-11-13）· E45 状态复核 + 文档数字首次逐格对账\n\n"
        "**远端状态无变化**：提交器 **273** `COMPLETED`（墙钟 05:04:04）、汇总作业 **291** "
        "`COMPLETED`（00:14:23）、三 seed 训练数组 275／283 与评测 274／282／290 全 "
        "`COMPLETED`、21 份评测齐、队列空；远端 `runs/e45/` 下 13 个 json 与本地 "
        "`tools/_evals/e45/` md5 **13/13 全同**；**未改任何远端状态、未 sbatch 训练／评测、"
        "未下载**。\n\n"
        "**本轮的新东西：第一次核对「文档里印出来的数字」**。前三十四轮核的是产物、"
        "文档措辞、结构与版本指针，**从来没有核过数字本身**——而 E45 的「finding four」"
        "说的恰恰是「一个数字必须能追到产物」。本轮补上这一环，用两个新只读脚本：\n\n"
        "1. `tools/_e45_recompute_raw.py`（远端跑）：**从 21 份评测报告独立重算**三 seed 表"
        "（自己解析 tag、自己算中位数与通过率、自己数超 Oracle 次数），"
        "再与 `scores_summary.json` 比对 ⇒ 七臂的中位数、三 seed 范围、逐 seed 值、"
        "过地板通过率、超 Oracle 次数**逐项相同** ⇒ `VERDICT: PASS`。\n"
        "2. `tools/_e45_docs_numbers.py`（本地跑）：**解析 E45 实验文件的真实 markdown 表格**"
        "（三 seed 分表、seed-0 分表、S3 等偏置表），逐格与记录数据比对；"
        "并扫描 MASTER／INDEX／HANDOVER 的散文里「紧跟在『中位』之后的数字」是否与记录矛盾。\n\n"
        f"**查出唯一一处缺陷**：§8.2 seed-0 分表里 A4 的 `CSI@2.0` 印作 **{BAD}**，"
        "而原始报告 `runs/e45/evals/A4_s0_ep00.eval.json` 的真值是 "
        f"0.1755001059215604（六位应为 **{GOOD}**）⇒ **末位笔误**，已订正；"
        "同一笔误已传到 HANDOVER 的 seed-0 表，一并订正。\n\n"
        "**不改变任何裁决**：§8.2 标题自己写着「单 seed，不构成裁决」；三 seed 分表本轮"
        "从原始产物重算后**逐格全对**；过地板通过率、S3 六臂判负、"
        "门指向／拓扑支／量通道三条机制线的读数**一个数都没动**。\n\n"
        "**新增可复用规则**：**文档审计必须分两层** —— 先验「措辞／结构／版本指针」，"
        "再验「印出来的数字」；**前者过了不等于后者过**。而且后者的失真会以**末位笔误**的"
        "形式逐级传染（本次就从实验文件传到了交接文档）⇒ **数字对账必须从原始产物重新推导**，"
        "不能拿文档互相对照。\n\n"
        "**结论与待办（不变）**：六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；"
        "拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；"
        "重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／"
        "`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与"
        "「地板口径不稳」。\n"
    )
    return text + note


def main():
    plan = [
        (MASTER, patch_master),
        (INDEX, patch_index),
        (E45DOC, lambda t: patch_plain(t, E45DOC_FIXES, "E45DOC", TAG)),
        (MEMORY, lambda t: patch_plain(t, MEMORY_FIXES, "MEMORY", TAG)),
        (HANDOVER, lambda t: patch_plain(t, HANDOVER_FIXES, "HANDOVER", TAG)),
        (DAILY, append_daily),
        (DAILY_PROJECT, append_daily),
    ]
    changed = []
    for path, fn in plan:
        if not path.exists():
            raise SystemExit(f"FILE MISSING: {path}")
        text = path.read_text(encoding="utf-8")
        new = fn(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
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
