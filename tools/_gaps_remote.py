"""Find the cases whose *history* windows contain an internal hole.

The criterion that matters for the robustness experiment is not "the label
count differs from the radar count" -- that is common and mostly harmless,
because the extra label frames sit outside the history window. What matters
is a hole *inside* the 11-frame history, which is what the interpolation
has to repair.

Two ways to find it:
  1. the radar timestamps themselves skip a 6-minute step
  2. the radar frame count is short, so the window spans a longer stretch
     of time than 11 consecutive frames would
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from radar_nwp_bench.data.index import minutes_between, scan_case

DATA = Path("/home/wf/data/radarNWP/BJ")

print(f"{'case':28s} {'radar':>6s} {'label':>6s} {'nwp':>4s} "
      f"{'radar gaps':>11s} {'lbl gaps':>9s} {'label>radar':>12s}")
print("-" * 84)

cases = []
for case_dir in sorted(DATA.glob("CP_*")):
    if not case_dir.is_dir():
        continue
    index = scan_case(case_dir)
    radar_times = list(index.radar_times)
    label_times = list(index.label_times)

    radar_gaps = [
        (a, b) for a, b in zip(radar_times, radar_times[1:])
        if minutes_between(b, a) != 6
    ]
    label_gaps = [
        (a, b) for a, b in zip(label_times, label_times[1:])
        if minutes_between(b, a) != 6
    ]
    n_rad = len(radar_times)
    n_lbl = len(label_times)
    print(f"{case_dir.name:28s} {n_rad:6d} {n_lbl:6d} {len(index.nwp_times):4d} "
          f"{len(radar_gaps):11d} {len(label_gaps):9d} {n_lbl - n_rad:12d}")
    cases.append((case_dir.name, n_rad, n_lbl, len(radar_gaps), len(label_gaps)))

print()
print("=" * 84)
print("cases whose radar series itself has a skipped 6-minute step")
print("=" * 84)
skipped = [c for c in cases if c[3] > 0]
print(f"count: {len(skipped)}")
for name, n_rad, n_lbl, rg, lg in skipped:
    print(f"  {name}  radar_gaps={rg} label_gaps={lg}")

print()
print("=" * 84)
print("cases where the label series has MORE frames than the radar series")
print("=" * 84)
extra = [c for c in cases if c[2] > c[1]]
print(f"count: {len(extra)}")
for name, n_rad, n_lbl, rg, lg in extra:
    print(f"  {name}  radar={n_rad} label={n_lbl} extra={n_lbl - n_rad}")

print()
print("=" * 84)
print("cases where the radar series is SHORT (the window spans more than 11 steps)")
print("=" * 84)
# A case with fewer radar frames than 11 cannot have a full window at all;
# one with more than 11 but a short count relative to its time span has holes.
for name, n_rad, n_lbl, rg, lg in cases:
    index = scan_case(DATA / name)
    times = list(index.radar_times)
    if len(times) < 2:
        continue
    span = minutes_between(times[-1], times[0])
    expected = span // 6 + 1
    if expected - len(times) > 0:
        print(f"  {name}: span {span} min -> expected {expected} frames, "
              f"have {len(times)}  (holes: {expected - len(times)})")
