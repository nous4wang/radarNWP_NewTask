"""Round 45 repair: remove the duplicated write-back blocks and fix the markers.

What went wrong
---------------
``tools/_e45_writeback_round45.py`` used ONE idempotency marker
("第 45 轮新增证据"), but that string only occurs in the HANDOVER block.  The
MEMORY bullet and the three daily-log blocks did not contain it, so the second
run of the write-back script appended them a second time:

    MEMORY.md            bullet   x2
    2026-11-11.md        logblock x2
    2026-11-13.md        logblock x2
    <project>/2026-11-13.md  logblock x2
    HANDOVER_E45.md      section  x1  (correct)

This script removes the surplus copies, keeping the FIRST occurrence of each
block.  Deduplication compares the exact block text (rstrip'ed), NOT the tail of
the file -- the round-43 lesson: comparing tails misfires because the first
copy's tail contains the second copy.

Then the marker in the write-back script is fixed to be per-target, so the
idempotency guard actually fires for every target.

Run:  python tools/_e45_repair_round45_dup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _e45_writeback_round45 import (  # noqa: E402
    HANDOVER,
    HANDOVER_BLOCK,
    LOG_1111,
    LOG_1113,
    LOG_1113_PROJ,
    LOG_BLOCK,
    MEMORY,
    MEMORY_BULLET,
)


def dedupe(path: Path, block: str) -> tuple[str, int]:
    """Keep the first occurrence of the exact ``block``; drop the rest."""
    text = path.read_text(encoding="utf-8")
    needle = block.strip("\n")
    first = text.find(needle)
    if first < 0:
        return text, 0
    removed = 0
    while True:
        second = text.find(needle, first + len(needle))
        if second < 0:
            break
        # swallow the whole block plus the blank line that follows it
        end = second + len(needle)
        while end < len(text) and text[end] == "\n":
            end += 1
        text = text[:second] + text[end:]
        removed += 1
    path.write_text(text, encoding="utf-8")
    return text, removed


def main() -> None:
    print("E45 round 45 duplicate repair")

    # ---- MEMORY.md: the bullet is one line ---------------------------------
    p = MEMORY
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    keep, removed = [], 0
    seen = False
    for ln in lines:
        if ln.lstrip().startswith("- **（2026-11-13 19:3x 第 45 次独立复核"):
            if seen:
                removed += 1
                continue
            seen = True
        keep.append(ln)
    p.write_text("".join(keep), encoding="utf-8")
    print(f"  {p}: removed {removed} duplicate bullet(s)")

    # ---- the three daily logs: multi-line blocks ----------------------------
    for p in (LOG_1111, LOG_1113, LOG_1113_PROJ):
        _, removed = dedupe(p, LOG_BLOCK)
        print(f"  {p}: removed {removed} duplicate log block(s)")

    # ---- HANDOVER: must stay exactly one ------------------------------------
    htext = HANDOVER.read_text(encoding="utf-8")
    hneedle = HANDOVER_BLOCK.strip("\n")
    hcount = htext.count(hneedle)
    if hcount > 1:
        first = htext.find(hneedle)
        head = htext[: first + len(hneedle)]
        tail = htext[first + len(hneedle) :].replace(hneedle, "")
        htext = head + tail
        HANDOVER.write_text(htext, encoding="utf-8")
        print(f"  {HANDOVER}: removed {hcount - 1} duplicate section(s)")
    else:
        print(f"  {HANDOVER}: section count {hcount} (ok)")

    print("done")


if __name__ == "__main__":
    sys.exit(main())
