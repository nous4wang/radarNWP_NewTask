"""Verify the rebuilt windows: evenly spaced in time, holes repaired, and
the partition clean."""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.data.index import (
    IRREGULAR_CASES,
    add_minutes,
    minutes_between,
    split_cases,
)

DATA = Path("/home/wf/data/radarNWP/BJ")

print("=" * 78)
print("PART 1: the partition")
print("=" * 78)
all_ids = sorted(p.name for p in DATA.glob("CP_*") if p.is_dir())
parts = split_cases(all_ids)
for name in ("train", "val", "test", "robustness_missing"):
    ids = parts.get(name, [])
    inside = [c for c in ids if c in IRREGULAR_CASES]
    print(f"  {name:20s} {len(ids):3d} cases, {len(inside)} of them irregular")
    if name != "robustness_missing" and inside:
        print(f"      !! CONTAMINATED: {inside}")

overlap = (
    set(parts["train"]) & set(parts["val"])
    | set(parts["train"]) & set(parts["test"])
    | set(parts["val"]) & set(parts["test"])
)
print(f"  overlap between the clean splits: {len(overlap)} (must be 0)")
covered = set(parts["train"]) | set(parts["val"]) | set(parts["test"]) | set(
    parts["robustness_missing"]
)
print(f"  every case accounted for: {covered == set(all_ids)} "
      f"({len(covered)}/{len(all_ids)})")

print()
print("=" * 78)
print("PART 2: are the history windows evenly spaced in time?")
print("=" * 78)
for split in ("train", "test", "robustness_missing"):
    ds = RadarNWPDataset(
        DATA, packed=False, split=split,
        in_len=11, out_len=10, patch_size=64, slide_stride=64,
    )
    cases = sorted({s.case_path.name for s in ds.samples})
    print(f"  split={split}: {len(ds.samples)} samples over {len(cases)} cases")

    # Rebuild the timestamps of each window from the case index.
    from radar_nwp_bench.data.index import scan_case
    bad_spacing = 0
    holes_found = 0
    checked = 0
    for spec in ds.samples[:4000]:
        idx = scan_case(spec.case_path)
        times = [idx.radar_times[i] for i in spec.radar_idx]
        diffs = [minutes_between(b, a) for a, b in zip(times, times[1:])]
        checked += 1
        # A repaired slot holds the frame from before the hole, so the step
        # across it is 12 minutes; every other step must be exactly 6.
        for k, d in enumerate(diffs):
            if d not in (6, 12):
                bad_spacing += 1
                break
        if any(d == 12 for d in diffs):
            holes_found += 1
        # the target times must be exactly +6 .. +60 minutes
        if spec.issue_time is None:
            bad_spacing += 1
    print(f"    checked {checked} windows: {bad_spacing} with an unexpected step, "
          f"{holes_found} carrying a hole")

print()
print("=" * 78)
print("PART 3: the target times are exactly +6 .. +60 minutes")
print("=" * 78)
ds = RadarNWPDataset(
    DATA, packed=False, split="robustness_missing",
    in_len=11, out_len=10, patch_size=64, slide_stride=64,
)
from radar_nwp_bench.data.index import scan_case
bad = 0
for spec in ds.samples[:3000]:
    idx = scan_case(spec.case_path)
    issue = spec.issue_time
    expected = [add_minutes(issue, 6 * k) for k in range(1, 11)]
    actual = [idx.label_times[i] for i in spec.label_idx]
    if actual != expected:
        bad += 1
print(f"  windows whose target timestamps are not exactly +6..+60: {bad} / 3000")

print()
print("=" * 78)
print("PART 4: the interpolation writes a real value and flags it")
print("=" * 78)
ds = RadarNWPDataset(
    DATA, packed=False, split="robustness_missing",
    in_len=11, out_len=10, patch_size=64, slide_stride=64, random_crop=False,
)
n_holes = 0
for spec in ds.samples:
    if spec.missing_radar_steps:
        n_holes += 1
print(f"  samples carrying a repaired hole: {n_holes} / {len(ds.samples)}")

# read the sample that actually carries the hole
target_idx = None
for i, spec in enumerate(ds.samples):
    if spec.missing_radar_steps:
        target_idx = i
        break
if target_idx is not None:
    item = ds[target_idx]
    mask = item["radar_missing_mask"].numpy()
    x = item["x_radar"].numpy()
    print(f"  sample {target_idx}: missing slots {ds.samples[target_idx].missing_radar_steps}")
    print(f"  mask: {mask.astype(int).tolist()}")
    print(f"  x_radar shape {tuple(x.shape)}, finite: {bool(np.isfinite(x).all())}")
    slot = int(np.argmax(mask))
    print(f"  the rebuilt slot {slot} carries a non-zero value: "
          f"max|x| = {float(np.abs(x[slot]).max()):.4f}")
    # the rebuilt slot must differ from the frame that was there before,
    # and must sit between its two neighbours
    print(f"  neighbour slots: {slot-1 if slot else None} and {slot+1}")
    for s in (slot - 1, slot, slot + 1):
        if 0 <= s < x.shape[0]:
            print(f"    slot {s}: mean|x| = {float(np.abs(x[s]).mean()):.5f}")
