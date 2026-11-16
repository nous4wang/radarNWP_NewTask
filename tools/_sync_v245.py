"""Correct the E45 amount-channel claim to be per-arm (v2.45).

What went wrong in v2.44
-----------------------
The write-back said "the amount channel never degenerated into a constant
across the three seeds".  The probe's raw JSON says otherwise for one arm:
A4b in seed 2 has ``scale_truth_corr = 0.2789 < 0.3`` and therefore
``scale_is_constant = True``.  The non-degeneracy claim holds for A4 (all
three seeds) and for A4b only in 2 of 3 seeds.

The verdicts are untouched: S3 already refuted all six arms, the topological
branch stays closed, the gate stays "structurally fixed but not the
bottleneck", and no arm enters the >=10 seed line.  Only the wording is
wrong, so only the wording is corrected.

Discipline
----------
* Every replacement is anchored on a short literal that must occur exactly
  the expected number of times -- a missing anchor is an error, never a
  silent skip.
* The new §12 changelog row is built from its CELLS; the separators are
  inserted by the code, and the pipe count is compared against the row it
  follows so the table cannot silently break.
* Running the script twice is a no-op (the v2.45 anchor is already present).
"""
from pathlib import Path
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
DAILY = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")

TAG = "v2.45"


def row(*cells):
    """One markdown table row from its cells (separators inserted by code)."""
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def unescaped_pipes(line):
    return len(re.findall(r"(?<!\\)\|", line))


def apply(text, old, new, count=1, label=""):
    """Replace `old` with `new`, exactly `count` times.

    Idempotency: if the new text is already in place, the fix counts as
    applied and nothing is changed.  A missing old anchor with the new text
    absent is a hard error -- never a silent skip.
    """
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
# corrections, per file
# --------------------------------------------------------------------------

MASTER_FIXES = [
    # header version line: v2.44 -> v2.45, with what changed
    (
        "2026-11-11（v2.44，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
        "没有一条进得了十 seed 正式跑**）",
        "2026-11-11（v2.45，**E45 三个种子全部跑完并结案＝负面：六条臂全部判负，"
        "没有一条进得了十 seed 正式跑**；v2.45 只订正量通道的分臂表述，裁决不变）",
        1,
        "master.header.version",
    ),
    # header: amount channel claim
    (
        "**量通道确实在做量**（三个种子都没退化成常数）",
        "**量通道确实在做量**（A4 三个种子都没退化；A4b 第三个种子退化 ⇒ 只在 2/3 seed 成立）",
        1,
        "master.header.amount",
    ),
    # §4.47 card
    (
        "（场最大值仍 402.43 mm），量通道三 seed 都没退化成常数；",
        "（场最大值仍 402.43 mm），量通道 A4 三 seed 都没退化、A4b 只在 2/3 seed 成立；",
        1,
        "master.card.amount",
    ),
    # §9 row and §12 v2.44 row (same wording, both must be corrected)
    (
        "**量通道三 seed 都没退化成常数**（变异系数 1.24～3.02、与真值相关 0.55～0.74）",
        "**量通道 A4 三 seed 都没退化**（`scale_cv` 1.24～2.00、与真值相关 0.55～0.74）、"
        "**A4b 只在 2/3 seed 成立**（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
        "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）",
        2,
        "master.status+changelog.amount",
    ),
]

INDEX_FIXES = [
    (
        "**量通道三 seed 都没退化成常数，但也不涨分 ⇒ 量通道不是瓶颈**",
        "**量通道 A4 三 seed 都没退化、A4b 只在 2/3 seed 成立（第三个种子 "
        "`scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数），但也不涨分 ⇒ 量通道不是瓶颈**",
        1,
        "index.amount",
    ),
]

E45DOC_FIXES = [
    (
        "量通道三个种子都没退化成常数\n   （`scale_cv` 1.24～3.02、与真值相关 0.55～0.74）；",
        "量通道 A4 三个种子都没退化（`scale_cv` 1.24～2.00、与真值相关 0.55～0.74）、"
        "A4b 只在 2/3 seed 成立（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
        "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ `scale_is_constant` 为真 ⇒ 退化成常数）；",
        1,
        "e45doc.conclusion3",
    ),
]

MEMORY_FIXES = [
    (
        "**量通道三 seed 未退化成常数但不涨分 ⇒ 量通道不是瓶颈**。",
        "**量通道 A4 三 seed 未退化、A4b 只在 2/3 seed 成立（第三个种子退化）"
        "但不涨分 ⇒ 量通道不是瓶颈**。",
        1,
        "memory.status",
    ),
    (
        "③**量通道三 seed 都没退化成常数**（`scale_cv` 1.24～3.02、`scale_truth_corr` 0.55～0.74）"
        "**但不涨分 ⇒ 量通道不是瓶颈**",
        "③**量通道 A4 三 seed 都没退化**（`scale_cv` 1.24～2.00、`scale_truth_corr` 0.55～0.74）、"
        "**A4b 只在 2/3 seed 成立**（`scale_cv` 1.89～3.02、与真值相关 0.66～0.71，"
        "第三个种子 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）**但不涨分 ⇒ 量通道不是瓶颈**",
        1,
        "memory.mechanism",
    ),
    (
        "**量通道不是瓶颈**（确实在做量但不涨分）。",
        "**量通道不是瓶颈**（A4 上确实在做量但不涨分；A4b 另有一个种子量通道退化）。",
        1,
        "memory.locations",
    ),
]

# --------------------------------------------------------------------------
# new §12 changelog row (MASTER)
# --------------------------------------------------------------------------

CHANGELOG_ROW = row(
    "2026-11-11",
    f"**{TAG}**",
    "**E45 结案结论的量通道读数订正（裁决不变）**：远端复核无异议 —— 提交器 273 与汇总作业 291 "
    "全部 `COMPLETED`，21 份评测齐，S3 等偏置对照六臂原始增益区间**全部跨零**、配平后**全部归零或仍跨零** "
    "⇒ **六臂判负、不投十 seed 算力**这一裁决不变。但逐读原始探针 JSON 发现 v2.44 有一处**过度断言**："
    "「量通道三 seed 都没退化成常数」**只对 A4 成立**（A4 的 `scale_cv` 1.24～2.00、"
    "`scale_truth_corr` 0.55～0.74、`scale_is_constant` 三 seed 均 False），"
    "**A4b 只在 2/3 seed 成立**（第三个种子 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant` = True "
    "⇒ 量通道退化）。⇒ 已按分臂表述订正 MASTER 头部版本行／§4.47 卡片／§9 行、§12 v2.44 行、"
    "INDEX 的 E45 行、E45 实验文件 §8.4 结案第 3 条与 `MEMORY.md`；"
    "**门指向（结构上已解决、不是瓶颈）、拓扑支（按预注册结掉）、S3 六臂判负、"
    "不投十 seed 算力、须重新提机制族**五项裁决均未改动（A4 的量通道机制主张成立且不涨分 ⇒ "
    "量通道不是瓶颈；A4b 因第三个种子退化与偏置爆表仍舍弃）。"
    "**本轮新增可复用规则**：写回里的机制读数必须**逐臂对读原始探针 JSON**，"
    "不得把单臂（或两臂中的多数）读数概括成全臂结论；这类过度断言在表格里会被 ✘ 标出、"
    "却在正文里被写成全称命题。",
    "`docs/experiments/E45-gate-direction-amount-separation-baseline.md` · "
    "`tools/_evals/e45/{diag_probe_s0,diag_probe_s1,diag_probe_s2,scores_summary,s3_equalbias}.json`",
)


def patch_master(text):
    if TAG in text:
        print("MASTER already carries", TAG, "-> skip")
        return text
    for old, new, cnt, label in MASTER_FIXES:
        text = apply(text, old, new, cnt, label)
    # insert the v2.45 changelog row right after the v2.44 row
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-11 | **v2.44** |")]
    if len(idx) != 1:
        raise SystemExit(f"CHANGELOG ANCHOR MISS: found {len(idx)} v2.44 changelog rows")
    i = idx[0]
    prev = unescaped_pipes(lines[i])
    new = unescaped_pipes(CHANGELOG_ROW)
    if prev != new:
        raise SystemExit(f"PIPE COUNT MISMATCH: v2.44 row {prev} cells vs new row {new} cells")
    lines.insert(i + 1, CHANGELOG_ROW)
    return "\n".join(lines)


def patch_index(text):
    if TAG in text:
        print("INDEX already carries", TAG, "-> skip")
        return text
    for old, new, cnt, label in INDEX_FIXES:
        text = apply(text, old, new, cnt, label)
    # append the v2.45 note to the last cell of the E45 row
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith("| E45 | §4.47 |")]
    if len(hits) != 1:
        raise SystemExit(f"INDEX E45 ROW MISS: found {len(hits)}")
    i = hits[0]
    body = lines[i].rstrip()
    if not body.endswith("|"):
        raise SystemExit("INDEX E45 ROW does not end with a pipe")
    lines[i] = body[:-1] + (
        " **v2.45 量通道读数订正**：v2.44 的「量通道三 seed 都没退化成常数」**只对 A4 成立**；"
        "A4b 第三个种子 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant` = True ⇒ 量通道退化。"
        "**裁决不变**（S3 六臂全跨零 ⇒ 六臂判负、不投十 seed 算力） |"
    )
    return "\n".join(lines)


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
        f"\n\n## {TAG}（本次运行，16:09）· E45 状态复核 + 量通道读数订正\n\n"
        "**远端状态无变化**：提交器 **273** `COMPLETED`（墙钟 05:04:04）、汇总作业 **291** "
        "`COMPLETED`（00:14:23）、三 seed 训练数组 275／283 与评测 274／282／290 全 `COMPLETED`、"
        "21 份评测齐；远端 `runs/e45/` 下 12 个 json（按 seed 分文件 9 份 + 合并 4 份 − 无）全部已取回本地 "
        "`tools/_evals/e45/`（13 份，含 seed-2 的 `scores_summary_s2.json`）。**未做任何远端修改、未手动 sbatch**。\n\n"
        "**复核结论**：v2.44 的裁决与全部数字**与原始产物逐项一致**（S3 六臂原始增益区间全跨零、"
        "配平后全归零或仍跨零；3-seed 中位数/过地板通过率；门指向 +0.39～+0.69；场最大值三 seed 全 402.43 mm；"
        "`scale_cv` 1.24～3.02）⇒ **六臂判负、不投十 seed 算力**不变。\n\n"
        "**但发现一处过度断言并已订正**：v2.44 写的「量通道三 seed 都没退化成常数」**只对 A4 成立** —— "
        "探针原始数据里 **A4b 的 seed 2 `scale_truth_corr` = 0.2789 < 0.3 ⇒ `scale_is_constant` = True**"
        "（E45 文件 §8.4 的表格其实已标 ✘，但 MASTER/INDEX/MEMORY 的正文写成了全称命题）。"
        "已用 `tools/_sync_v245.py` 按**分臂表述**订正 MASTER 头部版本行／§4.47 卡片／§9 行／"
        "§12 v2.44 行／新增 §12 v2.45 行、INDEX 的 E45 行、E45 文件 §8.4 结案第 3 条、`MEMORY.md`。\n\n"
        "**新增可复用规则**：写回里的机制读数必须**逐臂对读原始探针 JSON**，"
        "不得把单臂读数概括成全臂结论。\n"
    )
    return text + note


def main():
    plan = [
        (MASTER, patch_master),
        (INDEX, patch_index),
        (E45DOC, lambda t: patch_plain(t, E45DOC_FIXES, "E45DOC", TAG)),
        (MEMORY, lambda t: patch_plain(t, MEMORY_FIXES, "MEMORY", TAG)),
        (DAILY, append_daily),
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
            print("patched:", path)
        else:
            print("unchanged:", path)
    print("\nchanged files:", len(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
