"""Verify the v2.53 write-back rendered correctly.

Deliberately avoids shell heredocs: the point is to check byte-level
pipe escaping, and shell quoting is exactly what makes that ambiguous.
"""
import pathlib
import re
import sys

sys.path.insert(0, "tools")
from _check_v253_row import cells  # the project's own cell counter

ROOT = pathlib.Path(".").resolve()
MASTER = ROOT / "docs" / "MASTER.md"
INDEX = ROOT / "docs" / "experiments" / "INDEX.md"

m = MASTER.read_text(encoding="utf-8")
rows = m.split("\n")

v53 = [l for l in rows if l.startswith("| 2026-11-16 | **v2.53** |")]
print(f"v2.53 changelog rows: {len(v53)}")
row = v53[0]
print(f"  cells (project counter): {cells(row)}")
print(f"  unescaped pipes        : {len(re.findall(r'(?<!\\\\)\\|', row))}")
print(f"  escaped pipes          : {row.count(chr(92) + '|')}")

# the deviation token must be ESCAPED on disk, never raw
print()
print("  raw 'max|dev|' present  :", "max|dev|" in row)
print("  escaped 'max\\\\|dev\\\\|'  :", ("max" + chr(92) + "|dev" + chr(92) + "|") in row)
i = row.find("dev")
print("  byte context            :", repr(row[max(0, i - 30):i + 30]))

# the same for the INDEX E45 row and the header row
print()
for label, path, prefix in (
    ("INDEX E45 row", INDEX, "| E45 | §4.47（v2.49 压缩总览） |"),
    ("MASTER header", MASTER, "| 最后更新 |"),
):
    t = path.read_text(encoding="utf-8")
    hits = [l for l in t.split("\n") if l.startswith(prefix)]
    r = hits[0]
    print(f"{label}: cells={cells(r)} raw_max|dev|={'max|dev|' in r} "
          f"escaped={('max' + chr(92) + '|dev' + chr(92) + '|') in r}")

# every table in MASTER / INDEX / E45DOC must have a uniform cell count
print()
for p in ("docs/MASTER.md",
          "docs/experiments/INDEX.md",
          "docs/experiments/E45-gate-direction-amount-separation-baseline.md"):
    t = (ROOT / p).read_text(encoding="utf-8")
    header = None
    bad = []
    for n, line in enumerate(t.split("\n"), 1):
        s = line.strip()
        if not s.startswith("|"):
            header = None
            continue
        c = cells(line)
        if header is None:
            header = c
        elif c != header:
            bad.append((n, c, header))
    print(f"{p}: mismatched={len(bad)} {bad[:3]}")
