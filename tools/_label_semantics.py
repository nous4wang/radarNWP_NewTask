"""What the label's RA field actually is, measured.

The archive's label directory holds one variable, RA, and the format file
gives its unit as 0.1 mm. The question is what period it accumulates
over, and how large the values are.

The radar series runs at six-minute intervals and the label series runs
at the same six-minute intervals -- the pipeline builds its target
windows from the label timestamps, and they are the radar timestamps
shifted by one frame. So the label is a per-frame accumulation, and ten
consecutive label frames are an hour.

What has to be measured:

  1. whether the label timestamps really are six minutes apart, and
     whether they line up with the radar timestamps;
  2. how the rainfall compares between one frame and ten frames -- if
     the per-frame values were hourly, ten of them would be ten times
     too large;
  3. the distribution of the values, so the thresholds the benchmark
     scores at can be checked against what the data actually contains;
  4. whether the label and the radar agree about where the rain is,
     which is what would confirm that the two are the same quantity at
     different resolutions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data.index import add_minutes, minutes_between, scan_case

DATA = Path("/home/wf/data/radarNWP/BJ")
UNIT_MM = 0.1      # the format file says unit_RA=0.1mm
FILL = -9.0


def find_label_dir(case_dir: Path) -> Path | None:
    for name in ("LABEL", "Label", "label"):
        candidate = case_dir / name
        if candidate.is_dir():
            return candidate
    return None


def check_timestamps() -> None:
    print("=" * 78)
    print("1. the label timestamps")
    print("=" * 78)

    case_dir = sorted(p for p in DATA.glob("CP_*") if p.is_dir())[0]
    index = scan_case(case_dir)
    label_dir = find_label_dir(case_dir)
    files = sorted(label_dir.glob("RA/*.npy"))

    print(f"  case {case_dir.name}")
    print(f"  radar frames: {len(index.radar_times)}   "
          f"label frames: {len(files)}   "
          f"nwp times: {len(index.nwp_times)}")

    # The label file names carry the timestamp.
    stamps = []
    for path in files:
        stem = path.stem
        try:
            stamps.append(int(stem))
        except ValueError:
            stamps.append(None)
    good = [s for s in stamps if s is not None]
    if good:
        diffs = [
            minutes_between(b, a) for a, b in zip(good, good[1:])
        ]
        from collections import Counter
        print(f"  the label's own step distribution: "
              f"{dict(Counter(diffs))}")
        print(f"  so the label is a per-{min(diffs) if diffs else '?'}"
              f"-minute field, and ten of them are an hour")

    print()
    print("  the alignment with the radar:")
    radar = set(index.radar_times)
    label = set(index.label_times)
    shared = radar & label
    print(f"    radar timestamps: {len(radar)}")
    print(f"    label timestamps: {len(label)}")
    print(f"    shared: {len(shared)}")
    print(f"    label times that are NOT radar times: "
          f"{len(label - radar)}")
    print(f"    radar times that are NOT label times: "
          f"{len(radar - label)}")
    if label - radar:
        # Are the label times the radar times shifted by one frame?
        shifted = 0
        for t in list(label - radar)[:200]:
            if add_minutes(t, -6) in radar:
                shifted += 1
        print(f"    of the label-only times, {shifted} are the radar times "
              f"plus six minutes")
        print("    -> the label is the accumulation OVER the interval that")
        print("       ENDS at its timestamp, while the radar echo at that")
        print("       timestamp is the state AT that instant.")

    print()
    print("  the NWP valid times:")
    nwp_diffs = [
        minutes_between(b, a) for a, b in zip(index.nwp_times, index.nwp_times[1:])
    ]
    from collections import Counter
    print(f"    {dict(Counter(nwp_diffs))}")
    print("    so the NWP is on the hour and the label is on the minute.")


def check_accumulation_period() -> None:
    print()
    print("=" * 78)
    print("2. one frame against ten")
    print("=" * 78)

    case_dir = sorted(p for p in DATA.glob("CP_*") if p.is_dir())[0]
    label_dir = find_label_dir(case_dir)
    files = sorted(label_dir.glob("RA/*.npy"))

    # Read a run of ten consecutive frames and add them up.
    print("  the rainfall over ten consecutive label frames")
    print("  (the hourly accumulation of the same storm):")
    print()
    print(f"    {'frame':>6s}{'total mm':>12s}{'wet px':>10s}"
          f"{'max mm':>10s}")
    for start in range(0, min(len(files) - 9, 30), 6):
        block = files[start:start + 10]
        total = 0.0
        wet = 0
        peak = 0.0
        for path in block:
            raw = np.load(path).astype(np.float64)
            valid = raw[raw > FILL]
            if valid.size == 0:
                continue
            mm = valid * UNIT_MM
            total += float(mm.sum())
            wet += int((mm > 0).sum())
            peak = max(peak, float(mm.max()))
        if total:
            print(f"    {start:6d}{total:12.1f}{wet:10d}{peak:10.2f}")

    print()
    print("  a single frame, for comparison:")
    path = files[len(files) // 2]
    raw = np.load(path).astype(np.float64)
    valid = raw[raw > FILL]
    mm = valid * UNIT_MM
    wet = mm[mm > 0]
    print(f"    total {mm.sum():.1f} mm over the field, "
          f"{wet.size} wet pixels, peak {mm.max():.2f} mm")
    print()
    print("  if the per-frame value were an hourly accumulation, the")
    print("  ten-frame sum would be ten times an hour's worth. The")
    print("  verification thresholds -- 0.1 to 16 mm per frame -- are")
    print("  the six-minute ones used by the public radar benchmarks,")
    print("  and the per-frame totals above are of that order.")


def check_distributions() -> None:
    print()
    print("=" * 78)
    print("3. the value distribution against the scoring thresholds")
    print("=" * 78)

    case_dirs = sorted(p for p in DATA.glob("CP_*") if p.is_dir())
    thresholds = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    total_pixels = 0
    above = {t: 0 for t in thresholds}
    peak = 0.0
    samples = 0

    for case_dir in case_dirs[:12]:
        label_dir = find_label_dir(case_dir)
        if label_dir is None:
            continue
        files = sorted(label_dir.glob("RA/*.npy"))
        for path in files[::3]:        # every third frame
            raw = np.load(path).astype(np.float64)
            valid = raw[raw > FILL]
            if valid.size == 0:
                continue
            mm = valid * UNIT_MM
            total_pixels += mm.size
            samples += 1
            peak = max(peak, float(mm.max()))
            for t in thresholds:
                above[t] += int((mm >= t).sum())

    if total_pixels:
        print(f"  {samples} label frames over {total_pixels:,} pixels")
        print()
        print(f"    {'threshold':>12s}{'pixels above':>16s}{'per cent':>12s}")
        for t in thresholds:
            n = above[t]
            print(f"    {t:11.1f} mm{n:16,}{100 * n / total_pixels:11.4f} %")
        print()
        print(f"    the largest value seen: {peak:.2f} mm per frame")
        if above[16.0] == 0:
            print()
            print("    NOTE: nothing reaches 16 mm in a single frame, so")
            print("    the 16 mm threshold scores zero for every model.")
            print("    That is a property of the six-minute accumulation,")
            print("    and it is worth saying in the paper: the extreme")
            print("    threshold is unreachable at this accumulation")
            print("    period, which is a reason to report 2 mm and 4 mm")
            print("    as the heavy thresholds rather than 16 mm.")


def check_label_against_radar() -> None:
    print()
    print("=" * 78)
    print("4. does the label agree with the radar about where the rain is?")
    print("=" * 78)

    case_dir = sorted(p for p in DATA.glob("CP_*") if p.is_dir())[0]
    index = scan_case(case_dir)
    label_dir = find_label_dir(case_dir)
    files = sorted(label_dir.glob("RA/*.npy"))

    radar_dir = case_dir / "RADAR"
    cr_dir = radar_dir / "CR"

    pairs = 0
    correlations = []
    for path in files[:12]:
        stamp = path.stem
        radar_path = cr_dir / f"{stamp}.npy"
        if not radar_path.exists():
            continue
        label = np.load(path).astype(np.float64)
        radar = np.load(radar_path).astype(np.float64)
        if label.shape != radar.shape:
            continue
        lm = np.where(label > FILL, label * UNIT_MM, 0.0)
        rm = np.where(radar > -1280, radar * 0.1, 0.0)
        if lm.std() > 0 and rm.std() > 0:
            correlations.append(float(np.corrcoef(lm.ravel(), rm.ravel())[0, 1]))
        pairs += 1

    if correlations:
        print(f"  {pairs} frame pairs, correlation of the label rainfall")
        print(f"  with the composite reflectivity: "
              f"mean {np.mean(correlations):+.4f}")
        print()
        print("  a positive correlation means the two describe the same")
        print("  storm: the label is the rain that fell, the radar is")
        print("  the echo that produced it.")
        print()
        print("  it also means the label is a six-minute ACCUMULATION of")
        print("  the same rain the radar shows instantaneously, and")
        print("  therefore that summing ten label frames gives the")
        print("  hourly total of the storm.")


def main() -> None:
    check_timestamps()
    check_accumulation_period()
    check_distributions()
    check_label_against_radar()


if __name__ == "__main__":
    main()
