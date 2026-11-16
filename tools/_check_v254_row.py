"""Gate check for the v2.54 write-back (read-only).

Verifies:
  1. every table row the write-back touched still has the same cell
     count as its neighbours (an unescaped pipe would silently break
     the whole table on render);
  2. the version pointers in every document that should carry v2.54;
  3. no stale "4 files" claim is left where it is now wrong;
  4. the round-95 markers are present in the right files;
  5. the E45 verdict line is intact and unchanged in meaning.

VERDICT: PASS / FAIL
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

PIPE = re.compile(r"(?<!\\)\|")
CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + label + ((" :: " + detail) if detail else ""))
    return bool(ok)


def cells(line):
    return len(PIPE.findall(line))


def row_block(path, prefix):
    """The table header rows above `prefix` and the rows around it."""
    lines = path.read_text(encoding="utf-8").split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    return lines, hits


def neighbour_cell_count(path, prefix, span=3):
    """The modal cell count of the rows surrounding the matching row."""
    lines, hits = row_block(path, prefix)
    if not hits:
        return None, 0
    i = hits[0]
    counts = []
    for j in range(max(0, i - span), min(len(lines), i + span + 1)):
        l = lines[j]
        if l.startswith("|") and not set(l) <= set("|- :"):
            counts.append(cells(l))
    if not counts:
        return None, 0
    from collections import Counter
    modal, n = Counter(counts).most_common(1)[0]
    return modal, n


def main():
    print("=" * 72)
    print("v2.54 write-back gate check")
    print("=" * 72)

    print("\n=== 1. table row integrity (cell counts) ===")
    targets = [
        (MASTER, "| 2026-11-16 | **v2.54** |", "| 2026-11-16 | **v2.53** |"),
        (MASTER, "| 最后更新 |", None),
        (MASTER, "| §4.47 | 修三处机制缺陷", None),
        (MASTER, "| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |", None),
        (INDEX, "| E45 | §4.47（v2.49 压缩总览） |", None),
    ]
    for path, prefix, ref in targets:
        modal, n = neighbour_cell_count(path, prefix)
        lines, hits = row_block(path, prefix)
        if not hits:
            check("%s :: %s exists" % (path.name, prefix[:30]), False)
            continue
        c = cells(lines[hits[0]])
        check("%s :: %s :: cell count %d == modal %d"
              % (path.name, prefix[:30], c, modal if modal else -1),
              modal is not None and c == modal)
        if ref:
            rhits = [l for l in lines if l.startswith(ref)]
            check("%s :: the row it follows (%s) has the same cell count"
                  % (path.name, ref[:26]),
                  len(rhits) == 1 and cells(rhits[0]) == c)

    print("\n=== 2. version pointers ===")
    for path, needle, cnt in (
        (MASTER, "**v2.54（2026-11-16，最新）**", 1),
        (MASTER, "| 2026-11-16 | **v2.54** |", 1),
        (MASTER, "v2.54 见 §4.48 状态行", 2),
        (INDEX, "v2.54 第 95 轮", 1),
        (E45DOC, "> **2026-11-16（v2.54）**：", 1),
        (MEMORY, "docs/MASTER.md` **v2.54**", 1),
        (HANDOVER, "MASTER **v2.54**", 1),
    ):
        n = path.read_text(encoding="utf-8").count(needle)
        check("%s :: %r x%d" % (path.name, needle[:40], cnt), n == cnt, "found %d" % n)

    print("\n=== 3. the stale claim is gone where it is now wrong ===")
    m = MASTER.read_text(encoding="utf-8")
    # The phrase must survive ONLY inside this round's own changelog row,
    # where it is the description of the correction ("由 4 个订正为 5 个").
    # Everywhere else it was an assertion about the current state.
    stale_lines = [i for i, l in enumerate(m.split("\n"), 1)
                   if "最小同步集仍是这 4 个文件" in l
                   and not l.startswith("| 2026-11-16 | **v2.54** |")]
    check("MASTER asserts a 4-file minimal sync set nowhere outside this "
          "round's changelog row", not stale_lines, "lines = %s" % stale_lines)
    check("MASTER states the corrected 5-file set",
          "最小同步集**已订正为 5 个文件**" in m)
    check("MASTER's open-question item now says 5 files",
          "- ① **5 个**修复文件是否按最小集同步远端" in m)
    stale_files = [i for i, l in enumerate(m.split("\n"), 1)
                   if "4 个修复文件" in l
                   and not l.startswith("| 2026-11-16 | **v2.54** |")]
    check("MASTER keeps no '4 个修复文件' assertion outside the changelog row",
          not stale_files, "lines = %s" % stale_files)
    mem = MEMORY.read_text(encoding="utf-8")
    # the git-status line legitimately mentions "4 个修复文件" (the count of
    # uncommitted fix files as of that round); the to-do item must not.
    check("MEMORY's to-do item says 5 files",
          "5. 待用户裁决：① **5 个**修复文件按最小集同步远端" in mem)
    check("MEMORY's git-status line is left as history (still mentions 4)",
          "MASTER/INDEX/E45 实验文件 + 4 个修复文件都在工作区未提交" in mem)

    print("\n=== 4. round-95 markers ===")
    ho = HANDOVER.read_text(encoding="utf-8")
    for path, needle in (
        (HANDOVER, "<!-- E45-R95-BEGIN -->"),
        (HANDOVER, "<!-- E45-R95-END -->"),
        (HANDOVER, "v2.54 追加（第 95 轮：最小同步集订正为 5 个文件）"),
        (MEMORY, "- **E45 第 95 轮（2026-11-16）**"),
        (E45DOC, "v2.54 追加（第 95 轮：最小同步集订正为 5 个文件）"),
    ):
        check("%s :: %r" % (path.name, needle[:46]),
              needle in path.read_text(encoding="utf-8"))

    print("\n=== 5. the verdict is unchanged ===")
    for path in (MASTER, INDEX, E45DOC, MEMORY, HANDOVER):
        t = path.read_text(encoding="utf-8")
        check("%s :: still states the negative verdict" % path.name,
              "六臂判负" in t and "不投十 seed 算力" in t)
    check("MASTER's header still says the latest is v2.54",
          "**v2.54（2026-11-16，最新）**" in m)

    n = len(CHECKS)
    npass = sum(1 for _, ok, _ in CHECKS if ok)
    verdict = "PASS" if npass == n else "FAIL"
    print("\n" + "=" * 72)
    print("POS %d/%d   VERDICT: %s" % (npass, n, verdict))
    print("=" * 72)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
