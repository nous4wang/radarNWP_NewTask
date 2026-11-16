"""Fix the round-52 write-back cosmetics (idempotent).

Two defects in the r52 block as first written:
  1. the heading rendered the round number twice -- "第 52 轮 第 52 轮（...））" --
     because the template prefix "第 52 轮 " was concatenated with MARK, which
     already contains "第 52 轮（...）" (the r51 block has the same flaw; it is
     left untouched here to avoid churning another round's text);
  2. the lead-in said "四条前五十一轮都看不见的结构性事实" while five facts are
     actually listed.

Both are pure text fixes inside the r52 block; the guard marker is untouched,
so the block's identity and its idempotency guarantee are unaffected.
"""

from pathlib import Path

BAD_HEAD = "**⑯（2026-11-14 第 52 轮 第 52 轮（臂间合并检验轴的首次判读））**"
GOOD_HEAD = "**⑯（2026-11-14 第 52 轮（臂间合并检验轴的首次判读））**"
BAD_COUNT = "但拿到四条前五十一轮都看不见的结构性事实"
GOOD_COUNT = "但拿到五条前五十一轮都看不见的结构性事实"

TARGETS = [
    Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md"),
    Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md"),
    Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-14.md"),
    Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md"),
    Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-14.md"),
    Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
         "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md"),
]

MARK = "第 52 轮（臂间合并检验轴的首次判读）"

for path in TARGETS:
    text = path.read_text(encoding="utf-8")
    n_mark = text.count(MARK)
    if n_mark != 1:
        raise SystemExit("MARKER COUNT %d in %s (expected 1)" % (n_mark, path))
    orig = text
    text = text.replace(BAD_HEAD, GOOD_HEAD)
    text = text.replace(BAD_COUNT, GOOD_COUNT)
    if text == orig:
        print("%-22s unchanged (no cosmetic defect)" % path.name)
        continue
    path.write_text(text, encoding="utf-8")
    check = path.read_text(encoding="utf-8")
    if check.count(MARK) != 1:
        raise SystemExit("MARKER COUNT broke in %s" % path)
    if BAD_HEAD in check:
        raise SystemExit("BAD HEAD still present in %s" % path)
    print("%-22s fixed (marker still x1)" % path.name)
