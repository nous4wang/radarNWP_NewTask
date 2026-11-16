"""One-off: reword the v2.51 changelog clause and refresh the MASTER row.

The document gate ``tools/_e45_docs_numbers.py`` scopes its prose scan by
clause (a line is split at Chinese punctuation) and treats a number as a
median claim when the SAME clause names an E45 arm and carries the median
keyword.  The original wording put A3a/A3b's medians in the clause that
named B0, so the gate read them as corrupted B0 medians.

The fix is purely about WHERE each number sits: each median now lives in
the clause that names its own arm.  The claim is unchanged.
"""
import importlib.util
import pathlib
import re

spec = importlib.util.spec_from_file_location("s", "tools/_sync_v251.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

p = pathlib.Path("tools/_sync_v251.py")
t = p.read_text(encoding="utf-8")

old = ('"② **3/3 过地板的只有 A3a／A3b，但两臂中位数均低于同批对照 B0**（0.110015／0.116055 < 0.121817）"')
new = ('"② **3/3 过地板的只有 A3a／A3b**：A3a 中位数 0.110015、A3b 中位数 0.116055，"')
assert old in t, "script anchor not found"
t = t.replace(old, new)
old2 = ('    "且都低于 Oracle 上界 0.1653 ⇒ **不晋级**；')
new2 = ('    "**两者均低于同批对照 B0 的 0.121817**、且都低于 Oracle 上界 0.1653 ⇒ **不晋级**；')
assert old2 in t, "script anchor 2 not found"
t = t.replace(old2, new2, 1)
p.write_text(t, encoding="utf-8")
print("script: changelog clause reworded")

# reload so CHANGELOG_ROW reflects the new wording
spec = importlib.util.spec_from_file_location("s2", "tools/_sync_v251.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)

M = pathlib.Path("docs/MASTER.md")
lines = M.read_text(encoding="utf-8").split("\n")
hits = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-16 | **v2.51** |")]
assert len(hits) == 1, f"expected exactly one v2.51 changelog row, found {len(hits)}"
i = hits[0]
old_row = lines[i]
new_row = m2.CHANGELOG_ROW


def pipes(s: str) -> int:
    return len(re.findall(r"(?<!\\)\|", s))


assert pipes(old_row) == pipes(new_row), (pipes(old_row), pipes(new_row))
lines[i] = new_row
M.write_text("\n".join(lines), encoding="utf-8")
print(f"MASTER: v2.51 changelog row refreshed, pipe count {pipes(new_row)}")
