"""Repair: de-duplicate the round-43 log block and fix its idempotency marker.

What went wrong
---------------
tools/_log_round43.py appended its note using the marker "第 43 轮补记",
but the text it actually inserts reads "补记，第 43 轮" -- the marker was
NOT a substring of the inserted text.  The round-41 lesson says exactly
this: the marker must be a literal substring of the inserted text, or the
guard is blind.  So the second run appended the block again.

This script:
1.  removes the duplicate copy from both 2026-11-13.md files, keeping
    exactly one and keeping the FIRST occurrence (byte-identical
    verification before deleting anything);
2.  corrects the marker in tools/_log_round43.py so a third run is a
    no-op.

Deduplication is done on the full inserted block, not on a marker
string, and the two copies are compared byte-for-byte before one is
removed -- a blind delete of "the second one" would be wrong if they
ever differed.

Read-only with respect to the project's documents apart from the
de-duplication itself.
"""
from pathlib import Path
import sys

DAILY_UP = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md")
DAILY_PJ = Path(".workbuddy/memory/2026-11-13.md")
LOGGER = Path("tools/_log_round43.py")

#: the first line of the appended block -- the block starts with this and
#: runs to the end of the file (it is appended last)
BLOCK_HEAD = "- **（2026-11-13 17:0x 补记，第 43 轮）**"


def dedupe(path):
    text = path.read_text(encoding="utf-8")
    idx = []
    start = 0
    while True:
        j = text.find(BLOCK_HEAD, start)
        if j < 0:
            break
        idx.append(j)
        start = j + 1
    print(f"  {path}: {len(idx)} copy/copies of the round-43 block")
    if len(idx) <= 1:
        print("    nothing to do")
        return False
    # Extract each BLOCK (from its head to the next head, or to EOF) and
    # compare the BLOCKS -- comparing tails would be wrong, because the
    # first copy's tail contains the second copy.
    blocks = []
    for k, j in enumerate(idx):
        end = idx[k + 1] if k + 1 < len(idx) else len(text)
        blocks.append(text[j:end])
    first = blocks[0]
    for k, b in enumerate(blocks[1:], start=1):
        # compare the CONTENT: the copies can differ in trailing blank
        # lines only, because each append added its own newlines
        if b.rstrip() != first.rstrip():
            raise SystemExit(
                f"COPIES DIFFER in {path} (copy #{k}) -- refusing to delete "
                f"anything automatically")
    # keep the first block, drop the rest; the text before the first head
    # is kept verbatim
    head = text[:idx[0]]
    new = head + first
    while "\n\n\n\n" in new:
        new = new.replace("\n\n\n\n", "\n\n")
    path.write_text(new, encoding="utf-8")
    print(f"    removed {len(idx) - 1} duplicate copy/copies (byte-identical "
          f"blocks verified)")
    return True


def fix_marker():
    text = LOGGER.read_text(encoding="utf-8")
    old = 'append_once(DAILY_UP, "第 43 轮补记", APPEND_2026_11_13)'
    new = 'append_once(DAILY_UP, "补记，第 43 轮", APPEND_2026_11_13)'
    if old in text:
        text = text.replace(old, new)
        LOGGER.write_text(text, encoding="utf-8")
        print("  fixed the idempotency marker in tools/_log_round43.py")
    elif new in text:
        print("  the idempotency marker is already correct")
    else:
        raise SystemExit("ANCHOR MISS: the marker call site was not found")


def main():
    print("=== de-duplicate the round-43 log block ===")
    for p in (DAILY_UP, DAILY_PJ):
        if not p.exists():
            raise SystemExit(f"FILE MISSING: {p}")
        dedupe(p)

    print("=== fix the idempotency marker ===")
    fix_marker()

    print("=== verify ===")
    for p in (DAILY_UP, DAILY_PJ):
        t = p.read_text(encoding="utf-8")
        n = t.count(BLOCK_HEAD)
        print(f"  {p.name}: block copies = {n}, size = {len(t)} chars, "
              f"lines = {len(t.splitlines())}")
        if n != 1:
            raise SystemExit(f"STILL DUPLICATED in {p}: {n}")
    print("VERDICT: PASS")


if __name__ == "__main__":
    sys.exit(main())
