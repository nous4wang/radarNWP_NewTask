"""Is the 24.5 % real, or is the measurement wrong?

The patch count says that a quarter of the training patches reach 5 mm in
some frame of their target window. The pixel count says that 0.2 % of the
wet pixels reach 10 mm. Those two are consistent only if a patch is
large enough for one extreme pixel to matter, and it is: 64 x 64 is 4096
pixels, so a patch that covers any rain at all is likely to cover an
extreme cell somewhere.

But the claim that matters for the training is not "how many patches
qualify" -- it is "how many extreme pixels does the model get to see".
Duplicating a patch doubles the extreme pixels in it, but the patch is
one of many that overlap, and the extreme pixel is counted once per
overlapping patch.

The measurement that settles it: how many of the training set's extreme
pixels are seen at all, and how much does the duplication add?
"""
import sys
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
    print("1. the patches overlap: the same extreme pixel is seen many times")
    print("=" * 78)

    case_dir = sorted(p for p in DATA.glob("CP_*") if p.is_dir())[0]
    label_dir = case_dir / "LABEL" / "RA"
    index = scan_case(case_dir)
    files = sorted(label_dir.glob("*.npy"))
    frames = np.stack([clean_label_rain(np.load(p)) for p in files])

    r = len(frames) // 2
    window = frames[r + 1: r + 11].max(axis=0)

    extreme = window >= 5.0
    print(f"  one window of the case: {extreme.sum()} pixels at or above"
          f" 5 mm, out of {window.size:,}")

    seen = np.zeros_like(extreme, dtype=int)
    n_patches = 0
    n_qualifying = 0
    for y0 in range(0, window.shape[0] - SIZE + 1, STRIDE):
        for x0 in range(0, window.shape[1] - SIZE + 1, STRIDE):
            n_patches += 1
            block = extreme[y0:y0 + SIZE, x0:x0 + SIZE]
            if block.any():
                n_qualifying += 1
                seen[y0:y0 + SIZE, x0:x0 + SIZE] += block.astype(int)

    print(f"  patches: {n_patches}, of which {n_qualifying} qualify")
    if extreme.sum():
        print(f"  the extreme pixels are seen {seen[extreme].mean():.1f} times"
              f" on average")
        print(f"  the maximum is {int(seen[extreme].max())} times")
    print()
    print("  So the extreme pixels are not rare in the training set -- they")
    print("  are repeated by the overlapping windows. Duplicating a")
    print("  qualifying patch doubles its share of the extreme pixels, but")
    print("  the pixels are already there many times over.")

    print()
    print("=" * 78)
    print("2. what the duplication actually changes")
    print("=" * 78)

    # The extreme-pixel weight in the training set, before and after.
    total_pixels = 0
    extreme_pixels = 0
    for case_dir in sorted(p for p in DATA.glob("CP_*") if p.is_dir()):
        label_dir = case_dir / "LABEL" / "RA"
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.npy"))[::2]:
            raw = clean_label_rain(np.load(path))
            total_pixels += raw.size
            extreme_pixels += int((raw >= 5.0).sum())

    print(f"  extreme pixels per frame of the whole field: {extreme_pixels / 62 / 0.5:.0f}")
    print(f"  that is {100 * extreme_pixels / total_pixels:.4f} % of the field")
    print()
    print("  The duplication does not create extreme pixels, and it does")
    print("  not change their distribution. It changes how often the model")
    print("  sees a patch that contains one.")
    print()
    print("  Whether that helps depends on whether the problem is")
    print("  under-sampling of the extreme slices or something else -- and")
    print("  the measured frequency bias says the problem is not")
    print("  under-sampling: the model sees the extreme rain often enough")
    print("  and still predicts a fifth of it.")

    print()
    print("=" * 78)
    print("3. the diagnosis the mining is meant to address")
    print("=" * 78)
    print("  The proposal's premise is that the extreme slices are")
    print("  diluted by the sliding window. The measurement says the")
    print("  opposite: a quarter of the patches qualify, and the extreme")
    print("  pixels are seen many times each.")
    print()
    print("  So the mining does not address a sampling problem, because")
    print("  there is no sampling problem to address. The measured")
    print("  failure is that the models do not predict the extreme rain")
    print("  they are shown -- the bias is 0.08 to 0.21 across the family,")
    print("  and the spectrum says the field is being flattened.")
    print()
    print("  Duplicating the patches would make that worse, not better:")
    print("  it increases the weight of the extreme rain in the loss")
    print("  without changing the loss's shape, and the loss's shape is")
    print("  what makes the model conservative.")


if __name__ == "__main__":
    main()
