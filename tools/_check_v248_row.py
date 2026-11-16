"""Standalone gate for the v2.48 write-back.

Checks, all read-only:

1.  every file that must carry the v2.48 marker carries it, and the
    version pointers agree (MASTER header, INDEX E45 row, E45 experiment
    file, MEMORY header, HANDOVER pointer);
2.  the four Gate-3 places in MASTER are all present (header version
    line, the section 4.47 card, the section 9 row, a section 12
    changelog row) and the changelog row has the same cell count as the
    v2.47 row it follows;
3.  the WRONG claim is gone from every place that stated it -- the
    conclusion prose must no longer say A4b's seed-2 cell "退化成常数"
    (degenerated into a constant), because the CV branch never fired;
4.  the RIGHT claim is present: the correction states the correlation
    branch fired, that A4b seed 2 has the highest scale_cv of the six
    cells, and that the amount collapsed in magnitude;
5.  the pre-registered discard of A4b and the negative verdict are still
    stated (the correction must not silently drop them);
6.  the historical section 12 rows (v2.44-v2.47) are all still present,
    still have one consistent cell count, and no longer assert the wrong
    branch -- this project rewrites a changelog row IN PLACE when it
    carries a wrong conclusion (the v2.45 write-back did exactly that
    to the v2.44 row), and it must keep doing so;
7.  no forbidden claim is asserted anywhere in the five documents.

Exit code 0 iff every assertion holds.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

MASTER = ROOT / "docs" / "MASTER.md"
INDEX = ROOT / "docs" / "experiments" / "INDEX.md"
E45DOC = ROOT / "docs" / "experiments" / "E45-gate-direction-amount-separation-baseline.md"
MEMORY = Path(r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md")
HANDOVER = Path(r"R:\1radarNWPprediction\.workbuddy\memory\HANDOVER_E45.md")

TAG = "v2.48"

pos, neg = [], []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        pos.append(label)
        print(f"  PASS  {label}")
    else:
        neg.append(label)
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))


def cells(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line)) - 1


def vtuple(text: str):
    """The HIGHEST version token in `text`.

    The first match is NOT the pointer: the INDEX E45 row and the E45
    experiment file carry the whole version history (v2.41 ... v2.48),
    so the pointer is the newest token, not the oldest.
    """
    toks = [(int(a), int(b)) for a, b in re.findall(r"v(\d+)\.(\d+)", text or "")]
    return max(toks) if toks else None


texts = {}
for name, path in (("MASTER", MASTER), ("INDEX", INDEX), ("E45DOC", E45DOC),
                   ("MEMORY", MEMORY), ("HANDOVER", HANDOVER)):
    if not path.exists():
        print(f"  FAIL  {name} missing: {path}")
        neg.append(f"{name} missing")
        continue
    texts[name] = path.read_text(encoding="utf-8")

print("=== 1. version pointers ===")
vers = {}
if "MASTER" in texts:
    head = "\n".join(texts["MASTER"].splitlines()[:20])
    vers["MASTER"] = vtuple(head)
if "INDEX" in texts:
    rows = [l for l in texts["INDEX"].splitlines()
            if l.strip().startswith("|") and "E45" in l]
    vers["INDEX"] = vtuple("\n".join(rows))
if "E45DOC" in texts:
    vers["E45DOC"] = vtuple(texts["E45DOC"])
if "MEMORY" in texts:
    vers["MEMORY"] = vtuple("\n".join(texts["MEMORY"].splitlines()[:20]))
if "HANDOVER" in texts:
    vers["HANDOVER"] = vtuple("\n".join(texts["HANDOVER"].splitlines()[:20]))
for k, v in vers.items():
    print(f"        {k:9s} = v{v[0]}.{v[1]}" if v else f"        {k:9s} = None")
known = [v for v in vers.values() if v]
check(len(known) == 5 and len(set(known)) == 1,
      "all five version pointers agree", str(vers))
check(known and known[0] == (2, 48),
      "the version pointer is v2.48", str(known))
for name in texts:
    check(TAG in texts[name], f"{name} carries the {TAG} marker")

print("\n=== 2. Gate-3 places in MASTER ===")
m = texts.get("MASTER", "")
m_lines = m.split("\n")
check(any(l.startswith("| 最后更新 |") and TAG in l for l in m_lines[:20]),
      "MASTER header version line carries v2.48")
card = [l for l in m_lines if l.startswith("| **E45 / §4.47")]
check(len(card) == 1 and TAG in card[0], "MASTER section 4.47 card carries v2.48")
s9 = [l for l in m_lines if l.startswith("| §4.47 |")]
check(len(s9) == 1 and TAG in s9[0], "MASTER section 9 row carries v2.48")
rows = [(i, l) for i, l in enumerate(m_lines)
        if l.startswith("| 2026-11-13 | **v2.4") and "**v2.48**" in l]
check(len(rows) == 1, "MASTER section 12 has exactly one v2.48 changelog row")
if rows:
    i, l = rows[0]
    prev = [x for x in m_lines[:i] if x.startswith("| 2026-11-13 | **v2.4")]
    check(bool(prev), "the v2.48 changelog row follows another v2.4x row")
    if prev:
        check(cells(prev[-1]) == cells(l),
              f"the v2.48 row has the same cell count as the row it follows "
              f"({cells(l)} vs {cells(prev[-1])})")
    check(m_lines[i + 1].strip() == "",
          "the v2.48 changelog row is followed by a blank line")

print("\n=== 3. the wrong claim is gone from the conclusion prose ===")
WRONG = "0.2789 < 0.3 ⇒ 退化成常数"
WRONG2 = "0.2789 < 0.3 ⇒ `scale_is_constant` 为真 ⇒ 退化成常数"
WRONG3 = "0.2789 < 0.3 ⇒ `scale_is_constant` = True ⇒ 量通道退化"
WRONG4 = "0.2789 < 0.3 ⇒ 量通道退化"
for name in ("MASTER", "INDEX", "E45DOC", "MEMORY", "HANDOVER"):
    if name not in texts:
        continue
    t = texts[name]
    hits = t.count(WRONG) + t.count(WRONG2) + t.count(WRONG3) + t.count(WRONG4)
    check(hits == 0,
          f"{name} no longer asserts '退化成常数' for the 0.2789 cell",
          f"{hits} occurrence(s) left")

print("\n=== 4. the right claim is present ===")
for name in ("MASTER", "INDEX", "E45DOC", "MEMORY", "HANDOVER"):
    if name not in texts:
        continue
    t = texts[name]
    check("相关性分支" in t or "触发的是相关性分支" in t,
          f"{name} names the correlation branch")
    check("3.0188" in t, f"{name} cites A4b seed 2's scale_cv 3.0188")
    check("24.7" in t, f"{name} cites the 24.7x margin on the CV floor")
    check("量级塌缩" in t, f"{name} states the magnitude collapse")

print("\n=== 5. verdict still stated ===")
for name in ("MASTER", "INDEX", "E45DOC", "MEMORY", "HANDOVER"):
    if name not in texts:
        continue
    t = texts[name]
    check("六臂" in t and ("判负" in t or "负面" in t),
          f"{name} still states the negative verdict")
    check("不投十 seed" in t or "无一臂" in t or "十 seed" in t,
          f"{name} still states that no arm enters the 10-seed line")

print("\n=== 6. the historical changelog rows survive and are consistent ===")
if "MASTER" in texts:
    hist = []
    for tag in ("**v2.44**", "**v2.45**", "**v2.46**", "**v2.47**"):
        r = [l for l in m_lines if l.startswith("| 2026-11-1") and tag in l]
        check(len(r) == 1, f"the {tag} changelog row is still present")
        if r:
            hist.append((tag, r[0]))
    counts = {tag: cells(l) for tag, l in hist}
    check(len(set(counts.values())) == 1,
          f"the historical changelog rows all have the same cell count "
          f"({counts})")
    # they must not carry the wrong branch any more
    bad_rows = [tag for tag, l in hist
                if any(w in l for w in (WRONG, WRONG2, WRONG3, WRONG4))]
    check(not bad_rows,
          "no historical changelog row asserts the wrong branch any more",
          str(bad_rows))
    # and the new row must be present and consistent with them
    new = [l for l in m_lines if l.startswith("| 2026-11-1") and "**v2.48**" in l]
    if new:
        check(cells(new[0]) == counts.get("**v2.47**", -1),
              "the v2.48 row has the same cell count as the historical rows")

print("\n=== 7. forbidden claims absent as assertions ===")
PROHIBITION = ("禁写", "不得写", "不能写", "不可写", "不可再写", "不得再写",
               "不得宣称", "不可宣称", "不得用", "不可用")
FORBIDDEN = ["已实现 SOTA", "已解决 novelty", "C0 被推翻",
             "LABEL 有孤立针状尖峰"]
for name, t in texts.items():
    lines = t.splitlines()
    violations = []
    for i, line in enumerate(lines):
        for phrase in FORBIDDEN:
            pos_ = line.find(phrase)
            if pos_ < 0:
                continue
            prefix = line[:pos_]
            prev = "\n".join(lines[max(0, i - 2):i])
            if any(m in prefix for m in PROHIBITION):
                continue
            if any(m in prev for m in PROHIBITION):
                continue
            violations.append((i + 1, phrase))
    check(not violations, f"{name} asserts no forbidden claim",
          str(violations))

print("\n" + "=" * 78)
print(f"POS {len(pos)}/{len(pos) + len(neg)}   NEG {len(neg)}/{len(pos) + len(neg)}")
if neg:
    print("VERDICT: FAIL")
    sys.exit(1)
print("VERDICT: PASS")
