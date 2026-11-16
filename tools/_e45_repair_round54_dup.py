"""Repair: de-duplicate the round-53 write-back block that got appended 3x.

Cause: the round-54 gate sweep executed `tools/_e45_verify_round53.py`, and
that script is itself a write-back script which is NOT idempotent for the log
files (it appends unconditionally).  It therefore appended its round-53 block
a second and third time to the three log targets, landing AFTER the round-54
block.

Repair rule (from round 45's lesson): keep the FIRST copy byte-for-byte and
drop the later copies; never slice a file from "my marker" to "the next
heading" in a way that would remove other rounds' write-back blocks.

This script is a no-op when there is exactly one occurrence.
"""

import re
import sys
from pathlib import Path

R53 = "第 53 轮（机制剂量"

TARGETS = [
    ("UPPER_LOG_1114",
     Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md")),
    ("PROJ_LOG_1114",
     Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md")),
    ("AUTOMATION_MEM",
     Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
          "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")),
]

# a "block" starts at a level-3 heading or at a bolded 补记 lead-in
BLOCK_START = re.compile(r"^(### |\*\*补记，)")


def split_blocks(lines):
    starts = [i for i, ln in enumerate(lines) if BLOCK_START.match(ln)]
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(lines)
        out.append((s, e))
    return out


def main():
    total_removed = 0
    for name, path in TARGETS:
        if not path.exists():
            print("MISSING  %s (%s)" % (name, path))
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        blocks = split_blocks(lines)
        hits = [(s, e) for (s, e) in blocks
                if any(R53 in ln for ln in lines[s:e])]
        if len(hits) <= 1:
            print("ok        %-16s copies=%d (nothing to do)" % (name, len(hits)))
            continue
        keep = hits[0]
        drop = hits[1:]
        remove = set()
        for (s, e) in drop:
            for i in range(s, e):
                remove.add(i)
        new_lines = [ln for i, ln in enumerate(lines) if i not in remove]
        new_text = "\n".join(new_lines)
        # the dropped blocks were appended with a blank separator line before
        # them; make sure we did not leave a run of >2 blank lines
        new_text = re.sub(r"\n{4,}", "\n\n\n", new_text)
        path.write_text(new_text, encoding="utf-8")
        chk = path.read_text(encoding="utf-8")
        n = chk.count(R53)
        if n != 1:
            raise SystemExit("REPAIR FAILED in %s: %d copies remain (expected 1)"
                             % (name, n))
        removed = len(remove)
        total_removed += removed
        print("repaired  %-16s removed %d lines, copies 3 -> 1" % (name, removed))
    print("total lines removed: %d" % total_removed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
