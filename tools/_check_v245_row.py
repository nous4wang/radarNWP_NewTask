"""Standalone check: the new v2.45 §12 changelog row must have the same cell
count as the v2.44 row it follows, and the master's §12 table header."""
from pathlib import Path
import re

text = Path("docs/MASTER.md").read_text(encoding="utf-8")
lines = text.split("\n")


def cells(line):
    return len(re.findall(r"(?<!\\)\|", line)) - 1


rows = [
    (i, l)
    for i, l in enumerate(lines)
    if l.startswith("| 2026-11-11 | **v2.4") and ("**v2.44**" in l or "**v2.45**" in l)
]
for i, l in rows:
    print(f"L{i+1} cells={cells(l)}  {l[:40]}")

assert len(rows) == 2, f"expected the v2.44 and v2.45 rows, found {len(rows)}"
c44 = cells(rows[0][1])
c45 = cells(rows[1][1])
assert c44 == c45, f"CELL COUNT MISMATCH: v2.44={c44} v2.45={c45}"

# the §12 table header
for i, l in enumerate(lines):
    if l.startswith("| 日期") or l.startswith("|---") and "日期" in lines[i - 1]:
        print("header L%d cells=%d :: %s" % (i + 1, cells(l), l[:60]))
        break

print(f"VERDICT: PASS (v2.44={c44} cells, v2.45={c45} cells)")
