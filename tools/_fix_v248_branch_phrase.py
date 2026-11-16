"""In-place factual correction of the "退化成常数" wording (v2.48, part 2).

Why this script exists
-----------------------
The v2.48 write-back (tools/_sync_v248.py) corrected the places where the
CONCLUSION is stated -- the MASTER header, the section 4.47 card, the
section 9 row, the INDEX row, the E45 experiment file's section 8.4,
the MEMORY status bullet and the HANDOVER seed-2 warning.

It deliberately left the historical section 12 changelog rows and the
dated per-round notes inside MEMORY's status section alone, on the
reading that a changelog is an audit trail.  But the project's own
precedent cuts the other way: v2.45 rewrote the section 12 v2.44 row
IN PLACE when that row carried a wrong conclusion ("量通道三 seed 都没
退化成常数" -> per-arm wording), and MEMORY's status bullets are living
text, not a dated log.

And the phrase is not merely outdated -- it is FACTUALLY WRONG.  The
amount channel in A4b seed 2 is the most variable of the six cells
(scale_cv 3.0188, the maximum); what happened is a collapse in magnitude
plus a loss of correlation with the truth.  A reader who finds
"⇒ 退化成常数" in the changelog is being told something the artefacts
contradict, which is exactly the failure mode this project exists to
prevent.

So: rewrite the phrase wherever it still asserts the wrong branch, in
MASTER.md and MEMORY.md only.

The DAILY LOGS (2026-*.md) are NOT touched -- they are append-only by
project rule (the 2026-11-11 log keeps its original text even after
v2.47; the correction is registered in the current round's log, in the
changelog and in the handover instead).

Every replacement is idempotent: after it runs, the old phrase is gone
and a second run changes nothing.
"""
from pathlib import Path
import re
import sys

MASTER = Path("docs/MASTER.md")
INDEX = Path("docs/experiments/INDEX.md")
E45DOC = Path("docs/experiments/E45-gate-direction-amount-separation-baseline.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")

#: the corrected tail, used for every variant
FIXED = (
    "0.2789 < 0.3 ⇒ 触发相关性分支"
    "（该格 `scale_cv` 3.0188 是六格最高 ⇒ 不是「退化成常数」，"
    "实为量级塌缩＋与真值脱钩）"
)

#: every way the wrong branch was asserted, longest first so that the
#: longer forms are rewritten before their prefixes
VARIANTS = [
    "0.2789 < 0.3 ⇒ `scale_is_constant` 为真 ⇒ 退化成常数",
    "0.2789 < 0.3 ⇒ `scale_is_constant` = True ⇒ 量通道退化",
    "0.2789 < 0.3 ⇒ 量通道退化",
    "0.2789 < 0.3 ⇒ 退化成常数",
]


def patch(path):
    text = path.read_text(encoding="utf-8")
    original = text
    total = 0
    for v in VARIANTS:
        n = text.count(v)
        if n:
            text = text.replace(v, FIXED)
            total += n
            print(f"  {path.name}: replaced {n}x  :: {v[:60]}")
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"changed: {path}")
    else:
        print(f"unchanged: {path}")
    return total


def main():
    print("=== in-place correction of the wrong-branch wording ===")
    total = 0
    for p in (MASTER, INDEX, E45DOC, MEMORY):
        if not p.exists():
            raise SystemExit(f"FILE MISSING: {p}")
        total += patch(p)
    print()
    print(f"{total} occurrence(s) rewritten")

    # nothing may still assert the wrong branch
    for p in (MASTER, INDEX, E45DOC, MEMORY):
        t = p.read_text(encoding="utf-8")
        left = t.count("0.2789 < 0.3 ⇒ 退化成常数") \
            + t.count("0.2789 < 0.3 ⇒ `scale_is_constant` 为真 ⇒ 退化成常数") \
            + t.count("0.2789 < 0.3 ⇒ `scale_is_constant` = True ⇒ 量通道退化") \
            + t.count("0.2789 < 0.3 ⇒ 量通道退化")
        if left:
            raise SystemExit(f"STILL ASSERTED in {p}: {left} occurrence(s)")
    print("verified: no document still asserts the wrong branch")


if __name__ == "__main__":
    sys.exit(main())
