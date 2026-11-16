"""Gate check for the v2.55 write-back (read-only).

Verifies:
  1. every table row the write-back touched still has the same CELL count as
     its neighbours -- counted on UNESCAPED pipes only, because a cell may
     legitimately contain an escaped pipe (``max\\|dev\\|``) and splitting on
     every pipe would report a wrong column count;
  2. the version pointers in every document that should carry v2.55;
  3. the stale artifact count (13) is gone from MASTER's current-state
     statements and the corrected count (14) is in place;
  4. the round-96 markers are present in the right files;
  5. the E45 verdict line is intact and unchanged in meaning in every
     document.

The SNAPSHOT assertions of the older gates (``_check_v248/251/252/253_row``)
are expected to fail by design once the version pointer moves on: they pin
the pointer to the version they were written for.  This gate pins the
CURRENT version instead.

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
DAILY = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
DAILY_PROJECT = ROOT / ".workbuddy" / "memory" / "2026-11-16.md"

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + label + ((" :: " + detail) if detail else ""))
    return bool(ok)


def split_row(line):
    """Split a markdown table row on UNESCAPED pipes only."""
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


def ncols(line):
    return len(split_row(line)) - 2


def read(path):
    return path.read_text(encoding="utf-8")


def main():
    # ---------------- 1. table integrity ----------------
    for rel, path in [
        ("MASTER", MASTER),
        ("INDEX", INDEX),
        ("E45DOC", E45DOC),
    ]:
        lines = read(path).split("\n")
        bad = []
        header = None
        for n, l in enumerate(lines, 1):
            s = l.strip()
            if s.startswith("|") and s.endswith("|") and not set(s) <= set("|- :"):
                c = ncols(l)
                if header is None:
                    header = c
                elif c != header:
                    bad.append((n, c, header))
            else:
                header = None
        check(
            f"{rel}: every table row has its table's cell count",
            not bad,
            f"mismatched rows={len(bad)}" + (f" {bad[:4]}" if bad else ""),
        )

    master = read(MASTER)
    index = read(INDEX)
    e45doc = read(E45DOC)
    memory = read(MEMORY)
    handover = read(HANDOVER)
    daily = read(DAILY)
    daily_proj = read(DAILY_PROJECT)

    # ---------------- 2. version pointers ----------------
    check("MASTER header points at v2.55", "**v2.55（2026-11-16，最新）**" in master,
          f"count={master.count('**v2.55（2026-11-16，最新）**')}")
    check("MASTER carries a v2.55 changelog row", master.count("| 2026-11-16 | **v2.55** |") == 1,
          f"count={master.count('| 2026-11-16 | **v2.55** |')}")
    check("INDEX E45 row carries v2.55", "v2.55 第 96 轮" in index,
          f"count={index.count('v2.55 第 96 轮')}")
    check("E45DOC carries the v2.55 pointer", "> **2026-11-16（v2.55）**：" in e45doc,
          f"count={e45doc.count('> **2026-11-16（v2.55）**：')}")
    check("MEMORY points at v2.55", "docs/MASTER.md` **v2.55**" in memory,
          f"count={memory.count('docs/MASTER.md` **v2.55**')}")
    check("HANDOVER points at v2.55", "Gate 3 写回已完成（MASTER **v2.55**）。" in handover,
          f"count={handover.count('Gate 3 写回已完成（MASTER **v2.55**）。')}")

    # ---------------- 3. the stale artifact count ----------------
    check("MASTER no longer claims 13 artifacts in its current-state links",
          "13 份核心 json" not in master,
          f"occurrences={master.count('13 份核心 json')}")
    check("MASTER states 14 artifacts in its current-state links",
          "**14** 份核心 json" in master,
          f"occurrences={master.count('**14** 份核心 json')}")
    check("MASTER's v2.55 changelog row records the 13->14 correction",
          "产物计数 13→14" in master,
          f"occurrences={master.count('产物计数 13→14')}")
    check("the historical per-round rows are untouched (13-份 audit trail intact)",
          master.count("13 份 json") >= 1,
          f"historical occurrences={master.count('13 份 json')}")

    # ---------------- 4. round-96 markers ----------------
    for name, text, marker in [
        ("MASTER", master, "| 2026-11-16 | **v2.55** |"),
        ("INDEX", index, "v2.55 第 96 轮"),
        ("E45DOC", e45doc, "v2.55 追加（第 96 轮：三条新互证审计面）"),
        ("MEMORY", memory, "v2.55 第 96 轮"),
        ("HANDOVER", handover, "v2.55 追加（第 96 轮：三条新互证审计面）"),
        ("HANDOVER (round note)", handover, "<!-- E45-R96-BEGIN -->"),
        ("daily (outer)", daily, "## v2.55 第 96 轮（本次运行"),
        ("daily (project)", daily_proj, "## v2.55 第 96 轮（本次运行"),
    ]:
        check(f"{name} carries the round-96 marker", text.count(marker) == 1,
              f"count={text.count(marker)}")

    # ---------------- 5. the verdict is intact ----------------
    for name, text in [("MASTER", master), ("INDEX", index), ("E45DOC", e45doc),
                       ("MEMORY", memory), ("HANDOVER", handover)]:
        check(f"{name} still states the negative verdict",
              "六臂判负" in text and "不投十 seed 算力" in text,
              f"六臂判负={text.count('六臂判负')} 不投十 seed 算力={text.count('不投十 seed 算力')}")
    check("MASTER still states no arm entered the ten-seed run",
          "无一臂进十 seed" in master,
          f"count={master.count('无一臂进十 seed')}")

    # ---------------- 6. the three new surfaces are recorded ----------------
    for name, text in [("MASTER", master), ("E45DOC", e45doc), ("HANDOVER", handover)]:
        check(f"{name} records the cross-probe corroboration",
              "两个独立探针互相印证" in text,
              f"count={text.count('两个独立探针互相印证')}")
        check(f"{name} records the cross-experiment B0 identity",
              "B0 记录与别的实验逐位相同" in text or "B0 记录与 E43" in text,
              f"count={text.count('B0 记录与别的实验逐位相同') + text.count('B0 记录与 E43')}")
        check(f"{name} records the two-collection-path identity",
              "逐位相同 21/21" in text,
              f"count={text.count('逐位相同 21/21')}")

    # ---------------- 7. the two vacuous-assertion lessons are recorded ----------------
    for name, text in [("MASTER", master), ("HANDOVER", handover)]:
        check(f"{name} records the vacuous-assertion lesson",
              "空断言" in text,
              f"count={text.count('空断言')}")

    npass = sum(1 for _, ok, _ in CHECKS if ok)
    ntotal = len(CHECKS)
    print()
    print(f"POS {npass}/{ntotal}")
    verdict = "PASS" if npass == ntotal else "FAIL"
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
