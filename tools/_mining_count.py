"""How many patches would the extreme mining actually add?

The proposal keeps every patch that reaches 5 mm and adds it twice, on the
premise that the extreme slices are diluted by the sliding window. The
measurement is straightforward: classify every patch of the training set
and count.

The labels are read straight from the archive, and the patches are the
ones the reader actually walks, so the count is the count that matters.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data.index import scan_case
from radar_nwp_bench.data.transforms import clean_label_rain

DATA = Path("/home/wf/data/radarNWP/BJ")
SIZE = 64
STRIDE = 64


def main() -> None:
    print("=" * 78)
    print("the patches of the training set, classified the way the")
    print("proposal's mining classifies them")
    print("=" * 78)

    counts: Counter = Counter()
    n_cases = 0
    n_windows = 0

    for case_dir in sorted(p for p in DATA.glob("CP_*") if p.is_dir()):
        label_dir = case_dir / "LABEL" / "RA"
        if not label_dir.is_dir():
            continue
        index = scan_case(case_dir)
        files = sorted(label_dir.glob("*.npy"))
        if len(files) < 21:
            continue
        n_cases += 1

        # The frames of the case, in order.
        frames = []
        for path in files:
            frames.append(clean_label_rain(np.load(path)))
        field = np.stack(frames)

        # The windows the reader walks: ten target frames after an issue
        # time, the history before it.
        for r in range(10, len(field) - 10):
            window = field[r + 1: r + 11].max(axis=0)   # (H, W)
            n_windows += 1
            for y0 in range(0, window.shape[0] - SIZE + 1, STRIDE):
                for x0 in range(0, window.shape[1] - SIZE + 1, STRIDE):
                    patch = window[y0:y0 + SIZE, x0:x0 + SIZE]
                    peak = float(patch.max())
                    if peak >= 10.0:
                        counts["extreme 10mm"] += 1
                    elif peak >= 5.0:
                        counts["extreme 5mm"] += 1
                    elif peak >= 2.0:
                        counts["heavy 2mm"] += 1
                    elif peak >= 0.5:
                        counts["moderate 0.5mm"] += 1
                    elif peak >= 0.05:
                        counts["light"] += 1
                    else:
                        counts["dry"] += 1

    total = sum(counts.values())
    print(f"  {n_cases} cases, {n_windows} windows, {total:,} patches")
    print()
    print(f"    {'class':>18s}{'patches':>12s}{'per cent':>10s}")
    print("    " + "-" * 40)
    for name in ("extreme 10mm", "extreme 5mm", "heavy 2mm",
                 "moderate 0.5mm", "light", "dry"):
        n = counts.get(name, 0)
        print(f"    {name:>18s}{n:12,}{100 * n / max(total, 1):9.3f} %")

    extreme = counts.get("extreme 5mm", 0) + counts.get("extreme 10mm", 0)
    print()
    print(f"  the patches that reach 5 mm: {extreme:,} "
          f"({100 * extreme / max(total, 1):.4f} % of the set)")
    print()
    print("  the proposal keeps every one of them twice, so it adds")
    print(f"  {extreme:,} samples to a set of {total:,}: "
          f"{100 * extreme / max(total, 1):.4f} % more data.")
    print()
    if total:
        share = extreme / total
        if share < 0.01:
            print("  That is too small to move a score. The premise -- that")
            print("  the extreme slices are diluted -- is right, and the")
            print("  remedy does not follow: they are not diluted, they")
            print("  are simply few. The training set has")
            print(f"  {total:,} patches and {extreme:,} extreme ones;")
            print("  duplicating the rare ones twice cannot change what")
            print("  the model sees at the scale that matters.")
        else:
            print("  That is large enough to matter.")


if __name__ == "__main__":
    main()
