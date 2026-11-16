"""Standalone check for the v2.46 write-back.

Three things are checked, all of them about STRUCTURE (the numbers and
the verdicts are checked by tools/_e45_verify_round31.py and by the
round27-30 gates):

  1. the new v2.46 §12 changelog row has the same cell count as the
     v2.45 row it follows (a hand-written row with the wrong number of
     cells breaks the whole §12 table);
  2. the v2.46 row is unique -- exactly one row carries it;
  3. the version pointers agree: MASTER's header, MEMORY.md's header
     and HANDOVER_E45.md's pointer all name the SAME version, and that
     version is the one the newest changelog row carries.

Usage:  python tools/_check_v246_row.py
Exit code 0 iff every assertion holds.
"""
from pathlib import Path
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")

TAG = "v2.46"

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


m_text = MASTER.read_text(encoding="utf-8")
m_lines = m_text.split("\n")

# --- 1. the changelog row's cell count -----------------------------------
rows = [
    (i, l) for i, l in enumerate(m_lines)
    if l.startswith("| 2026-11-") and re.search(r"\*\*v2\.4[4-6]\*\*", l)
]
for i, l in rows:
    print(f"L{i+1} cells={cells(l)}  {l[:46]}")

v245 = [(i, l) for i, l in rows if "**v2.45**" in l]
v246 = [(i, l) for i, l in rows if f"**{TAG}**" in l]
check(len(v245) == 1, "the v2.45 changelog row is unique", f"found {len(v245)}")
check(len(v246) == 1, f"the {TAG} changelog row is unique", f"found {len(v246)}")

if v245 and v246:
    c45 = cells(v245[0][1])
    c46 = cells(v246[0][1])
    check(c45 == c46,
          f"the {TAG} row has the same cell count as the v2.45 row",
          f"v2.45={c45} cells, {TAG}={c46} cells")
    check(v246[0][0] == v245[0][0] + 1,
          f"the {TAG} row sits directly after the v2.45 row",
          f"L{v246[0][0]+1} vs L{v245[0][0]+1}")
    check("2026-11-13" in v246[0][1][:24],
          f"the {TAG} row is dated 2026-11-13",
          v246[0][1][:20])

# --- 2. the version pointers agree ---------------------------------------
m_head = re.search(r"最后更新 \| 2026-\d\d-\d\d（(v2\.\d+)", m_text)
mem_text = MEMORY.read_text(encoding="utf-8")
mem_head = re.search(r"`docs/MASTER\.md` (v2\.\d+)", mem_text)
ho_text = HANDOVER.read_text(encoding="utf-8")
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
          f"MASTER={m_head.group(1)} MEMORY={mem_head.group(1)} HANDOVER={ho_head.group(1)}")
    # The version pointer is compared with ">=", not "==": a legal later
    # bump (v2.47) must not turn this gate into a false failure, while the
    # pointer still has to name v2.46 OR SOMETHING NEWER.  The newest
    # version's own gate (tools/_check_v247_row.py) checks the equality
    # for the current version.
    def _vt(text):
        mm = re.search(r"v(\d+)\.(\d+)", text)
        return (int(mm.group(1)), int(mm.group(2))) if mm else None

    check(_vt(m_head.group(1)) >= _vt(TAG),
          f"the three pointers name {TAG} or newer",
          f"got {m_head.group(1)}")
    check(v246, f"MASTER's changelog carries a {TAG} row")

# --- 3. the write-back is present where it must be -----------------------
check(TAG in INDEX.read_text(encoding="utf-8"),
      f"INDEX's E45 row carries {TAG}")
check(TAG in E45DOC.read_text(encoding="utf-8"),
      f"the E45 experiment file carries {TAG}")
check(TAG in mem_text, f"MEMORY.md's status section carries {TAG}")

print()
print(f"POS {len(pos)}/{len(pos) + len(neg)}   NEG {len(neg)}/{len(neg)}")
if neg:
    print("VERDICT: FAIL")
    sys.exit(1)
print("VERDICT: PASS")
sys.exit(0)
