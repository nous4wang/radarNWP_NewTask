"""Standalone check for the v2.47 write-back.

What is checked
---------------
1. STRUCTURE of the new §12 changelog row: it exists exactly once, it has
   the same cell count as the v2.46 row it follows (a hand-written row with
   the wrong number of cells breaks the whole §12 table), and it sits
   directly after it.
2. The version pointers agree: MASTER's header, MEMORY.md's header and
   HANDOVER_E45.md's pointer all name the SAME version, and that version
   is the one the newest changelog row carries.
3. The write-back is present where it must be: INDEX's E45 row, the E45
   experiment file and MEMORY.md's status section all carry v2.47.
4. THE ACTUAL CORRECTION: the seed-0 score table of the E45 experiment
   file and the seed-0 table of HANDOVER_E45.md now print A4's
   ``CSI@2.0`` with the digits the recorded artefact implies, and the old
   wrong digit is gone from every authoritative file.

The numbers are checked by ``tools/_e45_docs_numbers.py`` and by the
remote ``tools/_e45_recompute_raw.py``; this script checks that the
write-back itself is well-formed and that the corrected digit is really
in place.

Usage:  python tools/_check_v247_row.py
Exit code 0 iff every assertion holds.
"""
from pathlib import Path
import json
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
S0 = Path("tools/_evals/e45/scores_summary_s0.json")

TAG = "v2.47"
PREV = "v2.46"

BAD = "0.175501"

pos: list = []
neg: list = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        pos.append(label)
        print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))
    else:
        neg.append(label)
        print(f"  FAIL  {label}" + (f"   [{detail}]" if detail else ""))


def cells(line: str) -> int:
    """Cell count, ignoring escaped pipes."""
    s = re.sub(r"\\\|", "ESC", line)
    parts = s.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return len(parts)


def vtuple(text: str):
    m = re.search(r"v(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


m_text = MASTER.read_text(encoding="utf-8")
m_lines = m_text.split("\n")
mem_text = MEMORY.read_text(encoding="utf-8")
ho_text = HANDOVER.read_text(encoding="utf-8")
e45_text = E45DOC.read_text(encoding="utf-8")

print("--- 1. the changelog row's structure ---------------------------------")
rows = [
    (i, l) for i, l in enumerate(m_lines)
    if l.startswith("| 2026-11-") and re.search(r"\*\*v2\.4[5-7]\*\*", l)
]
for i, l in rows:
    print(f"L{i+1} cells={cells(l)}  {l[:46]}")

v246 = [(i, l) for i, l in rows if f"**{PREV}**" in l]
v247 = [(i, l) for i, l in rows if f"**{TAG}**" in l]
check(len(v246) == 1, f"the {PREV} changelog row is unique", f"found {len(v246)}")
check(len(v247) == 1, f"the {TAG} changelog row is unique", f"found {len(v247)}")

if v246 and v247:
    c46 = cells(v246[0][1])
    c47 = cells(v247[0][1])
    check(c46 == c47,
          f"the {TAG} row has the same cell count as the {PREV} row",
          f"{PREV}={c46} cells, {TAG}={c47} cells")
    check(v247[0][0] == v246[0][0] + 1,
          f"the {TAG} row sits directly after the {PREV} row",
          f"L{v247[0][0]+1} vs L{v246[0][0]+1}")
    check("2026-11-13" in v247[0][1][:24],
          f"the {TAG} row is dated 2026-11-13", v247[0][1][:20])

print()
print("--- 2. the version pointers agree -------------------------------------")
m_head = re.search(r"最后更新 \| 2026-\d\d-\d\d（(v2\.\d+)", m_text)
mem_head = re.search(r"`docs/MASTER\.md` (v2\.\d+)", mem_text)
ho_head = re.search(r"MASTER \*\*(v2\.\d+)\*\*", ho_text)

check(m_head is not None, "MASTER's header version is readable",
      m_head.group(1) if m_head else "not found")
check(mem_head is not None, "MEMORY.md's version pointer is readable",
      mem_head.group(1) if mem_head else "not found")
check(ho_head is not None, "HANDOVER's version pointer is readable",
      ho_head.group(1) if ho_head else "not found")

if m_head and mem_head and ho_head:
    vals = {m_head.group(1), mem_head.group(1), ho_head.group(1)}
    check(len(vals) == 1,
          "MASTER / MEMORY.md / HANDOVER name the SAME version",
          f"MASTER={m_head.group(1)} MEMORY={mem_head.group(1)} "
          f"HANDOVER={ho_head.group(1)}")
    # A LEGAL later bump must not turn this gate into a false failure --
    # same relaxation the v2.46 gate received when v2.47 landed (see the
    # round-35 note: "版本指针断言用 >= 而不是 ==").
    check(vtuple(m_head.group(1)) >= vtuple(TAG),
          f"the three pointers name {TAG} or later", f"got {m_head.group(1)}")
    check(bool(v247), f"MASTER's changelog carries a {TAG} row")

print()
print("--- 3. the write-back is present where it must be ----------------------")
check(TAG in INDEX.read_text(encoding="utf-8"), f"INDEX's E45 row carries {TAG}")
check(TAG in e45_text, f"the E45 experiment file carries {TAG}")
check(TAG in mem_text, f"MEMORY.md's status section carries {TAG}")
check(TAG in ho_text, f"HANDOVER carries {TAG}")

print()
print("--- 4. the correction itself ------------------------------------------")
# the digit the artefact implies
truth = None
if S0.exists():
    s0 = json.loads(S0.read_text(encoding="utf-8"))
    truth = float(s0["A4_s0_ep00"]["csi_at_2.0"])
    want = f"{truth:.6f}"
    check(want == "0.175500", "the artefact's A4 seed-0 CSI@2.0 rounds to 0.175500",
          f"raw={truth!r} -> {want}")
    # the E45 experiment file's seed-0 table
    m = re.search(r"^\| A4 \| (\d+\.\d+) \|", e45_text, re.M)
    check(m is not None and m.group(1) == want,
          "the E45 experiment file's seed-0 table prints the corrected digit",
          m.group(1) if m else "row not found")
    # the handover's seed-0 table
    m2 = re.search(r"^\| \*\*A4\*\* \| \*\*(\d+\.\d+)\*\* \|", ho_text, re.M)
    check(m2 is not None and m2.group(1) == want,
          "HANDOVER's seed-0 table prints the corrected digit",
          m2.group(1) if m2 else "row not found")
else:
    check(False, "the recorded seed-0 summary is present", str(S0))

#: a table row may name the old digit only when it is clearly talking about
#: the CORRECTION (the notes have to name it: "0.175501 -> 0.175500")
CORRECTION_MARKERS = ("v2.47", "订正", "末位笔误", "对账", "→ 0.175500")

for name, text in (("MASTER.md", m_text), ("INDEX.md", INDEX.read_text(encoding="utf-8")),
                   ("the E45 experiment file", e45_text), ("MEMORY.md", mem_text),
                   ("HANDOVER_E45.md", ho_text)):
    # The old digit may legitimately appear inside the CORRECTION NOTES
    # (they have to name it: "0.175501 -> 0.175500").  What must not
    # survive is the old digit as a REPORTED SCORE, i.e. inside a table
    # row that does not talk about the correction itself.
    bad_rows = [l for l in text.split("\n")
                if l.strip().startswith("|") and BAD in l
                and not any(marker in l for marker in CORRECTION_MARKERS)]
    check(not bad_rows, f"{name} has no table row that still prints {BAD} as a score",
          f"{len(bad_rows)} offending row(s)")
    if bad_rows:
        for l in bad_rows[:3]:
            print(f"        offending: {l[:120]}")

print()
print(f"POS {len(pos)}/{len(pos) + len(neg)}   NEG {len(neg)}/{len(neg)}")
if neg:
    print("VERDICT: FAIL")
    sys.exit(1)
print("VERDICT: PASS")
sys.exit(0)
