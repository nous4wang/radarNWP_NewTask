"""Decide where a hole in the history window actually is, and what the
interpolation should write into it.

The question the code has to answer: the history window is 11 *positions* in
the radar array. If the radar series skips a 6-minute step, the positions are
still consecutive but the *times* are not. So which slot holds the hole?
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data.index import minutes_between, scan_case

DATA = Path("/home/wf/data/radarNWP/BJ")

case_name = "CP_01_06230359_00001"
index = scan_case(DATA / case_name)
times = list(index.radar_times)

print(f"case {case_name}: {len(times)} radar frames")
print("first 20 stamps:", times[:20])
print()

# Where are the 12-minute steps?
for i, (a, b) in enumerate(zip(times, times[1:])):
    gap = minutes_between(b, a)
    if gap != 6:
        print(f"  position {i} -> {i+1}: {a} -> {b}  gap = {gap} minutes "
              f"({'HOLE between these two positions' if gap == 12 else 'other'})")

print()
print("=" * 78)
print("The window that contains the hole")
print("=" * 78)
print("Take the issue time at position r. The history is positions r-10..r.")
print("If the hole is between positions j and j+1, then:")
print("  - the SLOT j+1 in the window holds the frame at time t[j+1]")
print("  - but time t[j+1] is 12 minutes after t[j]")
print("  - the physically missing frame is the one at t[j] + 6 minutes")
print()

# Build the actual history for a window whose middle is the hole and show
# what the loader would read.
for r in range(len(times)):
    hist_indices = list(range(r - 10, r + 1))
    if hist_indices[0] < 0:
        continue
    window_times = [times[i] for i in hist_indices]
    gaps = [minutes_between(b, a) for a, b in zip(window_times, window_times[1:])]
    if 12 in gaps:
        step_i = gaps.index(12)
        print(f"issue position {r} (time {times[r]}):")
        print(f"  window slots 0..10 -> times {window_times}")
        print(f"  12-minute gap between slot {step_i} (t={window_times[step_i]}) "
              f"and slot {step_i+1} (t={window_times[step_i+1]})")
        print(f"  the missing frame is at t = {window_times[step_i]} + 6 min")
        print(f"  slot {step_i+1} physically holds the frame at "
              f"{window_times[step_i+1]} (12 min later)")
        print()
        print("  Interpretation A (the current code):")
        print(f"    hist[{step_i+1}] := 0.5*(hist[{step_i}] + hist[{step_i+2}])")
        print(f"    -> the slot for the missing t+6 frame gets the average of the")
        print(f"       frames at t[{step_i}] and t[{step_i+2}]")
        print()
        print("  Interpretation B (the reviewer's claim):")
        print(f"    the good observation at t={window_times[step_i+1]} is destroyed")
        print(f"    and the missing t+6 frame is never reconstructed")
        break

print()
print("=" * 78)
print("How many windows have a hole, and where")
print("=" * 78)
from collections import Counter

slot_hist = Counter()
edge_holes = 0
total = 0
for case_dir in sorted(DATA.glob("CP_*")):
    if not case_dir.is_dir():
        continue
    idx = scan_case(case_dir)
    ts = list(idx.radar_times)
    for r in range(10, len(ts)):
        total += 1
        hist_indices = list(range(r - 10, r + 1))
        wt = [ts[i] for i in hist_indices]
        for step_i, (a, b) in enumerate(zip(wt, wt[1:])):
            if minutes_between(b, a) == 12:
                slot_hist[step_i + 1] += 1
                if step_i + 1 == 10:
                    edge_holes += 1
print(f"windows examined: {total}")
print("hole slot -> count (slot 1..10):")
for s in range(1, 11):
    n = slot_hist.get(s, 0)
    flag = "  <- at the very edge of the window (not repaired by the guard)" if s in (1, 10) else ""
    print(f"  slot {s:2d}: {n:5d}{flag}")
print(f"total holes: {sum(slot_hist.values())}")
