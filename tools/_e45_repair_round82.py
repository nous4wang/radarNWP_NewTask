#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair the side effects of the stray round-16 / round-53 write-back runs.

Context (round 82): a blanket loop over the document gates also executed
historical write-back scripts that are NOT read-only. Two of them appended a
copy of an already-present block into the daily logs:

  * tools/_e45_log_round16.py    -> appended a fresh "第 16 次独立复核" block
    (timestamped with the CURRENT run time) to 2026-11-12.md (upper + project)
    and to the automation memory file.
  * tools/_e45_verify_round53.py -> appended a second "第 53 轮（机制剂量…）"
    block to 2026-11-14.md (upper + project) and to the automation memory file.

Which copy is the stray one is decided by POSITION, not by text equality
(the two copies differ in wording, so a byte-equality dedup finds nothing):

  * 2026-11-12.md : the stray heading is the one carrying TODAY's timestamp
    ("2026-11-15 15:36"); the authentic block is "2026-11-12 07:07".
  * 2026-11-14.md : the stray block is the one sitting immediately BEFORE the
    "<!-- E45-R62-BEGIN -->" marker; the authentic block is the earlier one
    (it also carries the later rounds' 补记 54/55 sections, which proves it is
    the original).
  * automation memory : the stray region is everything between
    "<!-- E45-R81-END -->" and "<!-- E45-R62-BEGIN -->".

Contract:
  * every removal is anchored and asserted (expected counts, expected
    neighbours) - the script refuses to write if any assertion fails;
  * a .bak copy of every touched file is written before the first write;
  * the four authoritative files (HANDOVER_E45.md / MEMORY.md / 2026-11-11.md /
    project 2026-11-15.md) are NOT touched - they already match the round-81
    md5 baseline byte-for-byte.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import sys

UPPER = r"R:/1radarNWPprediction/.workbuddy/memory"
PROJ = r"R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory"
AUTOMEM = os.path.join(PROJ, "automations",
                       "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")
BACKUP = os.path.join(PROJ, "_repair_r82_backup")

R16_LOGS = [os.path.join(UPPER, "2026-11-12.md"),
            os.path.join(PROJ, "2026-11-12.md")]
R53_LOGS = [os.path.join(UPPER, "2026-11-14.md"),
            os.path.join(PROJ, "2026-11-14.md")]

R16_HEAD = re.compile(r"^#{2,4} .*第 16 次独立复核")
R53_HEAD = re.compile(r"^#{2,4} .*第 53 轮（机制剂量")
R62_MARK = "<!-- E45-R62-BEGIN -->"
R81_MARK = "<!-- E45-R81-END -->"

AUTHENTIC_R16_STAMP = "2026-11-12 07:07"


def read(p: str) -> str:
    return io.open(p, encoding="utf-8", newline="").read()


def write(p: str, s: str) -> None:
    with io.open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(s)


def backup(p: str) -> str:
    if not os.path.isdir(BACKUP):
        os.makedirs(BACKUP)
    tag = "upper_" if p.startswith(UPPER) else "proj_"
    dst = os.path.join(BACKUP, tag + os.path.basename(p) + ".r82bak")
    shutil.copy2(p, dst)
    return dst


def section_bounds(lines, start):
    """End (exclusive) of the section starting at `start` = next heading."""
    for j in range(start + 1, len(lines)):
        if re.match(r"^#{2,4} ", lines[j]):
            return j
    return len(lines)


def remove_r16(path: str) -> bool:
    lines = read(path).split("\n")
    hits = [i for i, l in enumerate(lines) if R16_HEAD.match(l)]
    if len(hits) == 1:
        print("  OK   %s : round-16 heading already unique (nothing to do)"
              % os.path.basename(path))
        return False
    assert len(hits) == 2, "%s: expected 2 round-16 headings, found %d" % (
        path, len(hits))
    stray = [i for i in hits if AUTHENTIC_R16_STAMP not in lines[i]]
    assert len(stray) == 1, "%s: expected exactly 1 stray round-16 heading" % path
    i = stray[0]
    end = section_bounds(lines, i)
    # swallow the blank line that separates this block from the previous one
    start = i
    if start > 0 and lines[start - 1].strip() == "":
        start -= 1
    bak = backup(path)
    write(path, "\n".join(lines[:start] + lines[end:]))
    print("  FIXED %s : removed stray round-16 block at line %d "
          "(kept authentic '%s')  backup=%s"
          % (os.path.basename(path), i + 1, AUTHENTIC_R16_STAMP, bak))
    return True


def remove_r53(path: str) -> bool:
    lines = read(path).split("\n")
    hits = [i for i, l in enumerate(lines) if R53_HEAD.match(l)]
    if len(hits) == 1:
        print("  OK   %s : round-53 heading already unique (nothing to do)"
              % os.path.basename(path))
        return False
    assert len(hits) == 2, "%s: expected 2 round-53 headings, found %d" % (
        path, len(hits))
    r62 = [i for i, l in enumerate(lines) if R62_MARK in l]
    assert len(r62) == 1, "%s: expected exactly 1 %s marker" % (path, R62_MARK)
    # The stray block is the one sitting IMMEDIATELY BEFORE the R62 marker:
    # it must be the last round-53 heading before the marker, and no other
    # heading may separate it from the marker.
    before = [i for i in hits if i < r62[0]]
    assert len(before) >= 1, "%s: no round-53 heading before the R62 marker" % path
    i = before[-1]
    others = [j for j in range(i + 1, r62[0]) if re.match(r"^#{2,4} ", lines[j])]
    assert not others, "%s: round-53 heading at line %d is not adjacent to the " \
        "R62 marker (separated by %r) - refusing" % (path, i + 1, others)
    end = section_bounds(lines, i)
    start = i
    if start > 0 and lines[start - 1].strip() == "":
        start -= 1
    bak = backup(path)
    write(path, "\n".join(lines[:start] + lines[end:]))
    print("  FIXED %s : removed stray round-53 block at line %d  backup=%s"
          % (os.path.basename(path), i + 1, bak))
    return True


def remove_automem() -> bool:
    lines = read(AUTOMEM).split("\n")
    r81 = [i for i, l in enumerate(lines) if R81_MARK in l]
    r62 = [i for i, l in enumerate(lines) if R62_MARK in l]
    assert len(r81) == 1 and len(r62) == 1, "automation memory: marker counts wrong"
    region = lines[r81[0] + 1:r62[0]]
    txt = "\n".join(region)
    has_r16 = "第 16 次独立复核" in txt
    has_r53 = "第 53 轮（机制剂量" in txt
    if not has_r16 and not has_r53:
        print("  OK   automation memory : stray region already removed "
              "(nothing to do)")
        return False
    assert has_r16, "automation memory: stray R16 block not found"
    assert has_r53, "automation memory: stray R53 block not found"
    assert not re.search(r"E45-R(7[0-9]|8[0-9])-BEGIN", txt), \
        "automation memory: region contains a recent marker - refusing"
    bak = backup(AUTOMEM)
    write(AUTOMEM, "\n".join(lines[:r81[0] + 1] + lines[r62[0]:]))
    print("  FIXED automation memory : removed stray region (R16+R53 blocks, "
          "%d lines) between %s and %s  backup=%s"
          % (len(region), R81_MARK, R62_MARK, bak))
    return True


def verify() -> None:
    print("\n=== verification ===")
    # Count HEADINGS only: the round-82 write-back prose legitimately mentions
    # these round names inside a bullet, so a raw substring count would report
    # a false duplicate.
    def head_count(path: str, pattern) -> int:
        t = read(path)
        return sum(1 for l in t.splitlines() if pattern.match(l))

    for p in R16_LOGS:
        assert head_count(p, R16_HEAD) == 1, "%s: round-16 heading not unique" % p
        assert AUTHENTIC_R16_STAMP in read(p), \
            "%s: authentic round-16 block lost" % p
        print("  OK  %s : round-16 heading count = 1 (authentic kept)" % p)
    for p in R53_LOGS:
        assert head_count(p, R53_HEAD) == 1, "%s: round-53 heading not unique" % p
        assert "补记，第 54 轮" in read(p), \
            "%s: authentic round-53 block lost" % p
        print("  OK  %s : round-53 heading count = 1 (authentic kept)" % p)
    t = read(AUTOMEM)
    assert head_count(AUTOMEM, R16_HEAD) == 1, "automem: round-16 heading not unique"
    assert head_count(AUTOMEM, R53_HEAD) == 1, "automem: round-53 heading not unique"
    assert t.count(R62_MARK) == 1 and t.count(R81_MARK) == 1
    print("  OK  automation memory : R16 and R53 headings each unique, "
          "markers intact")


def main() -> int:
    print("=== backup dir: %s ===" % BACKUP)
    for p in R16_LOGS:
        remove_r16(p)
    for p in R53_LOGS:
        remove_r53(p)
    remove_automem()
    verify()
    print("\nREPAIR: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
