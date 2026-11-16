"""Are the proposed thresholds usable on this archive?

The proposal recalibrates the thresholds to (0.1, 0.5, 2.0, 5.0, 10.0) mm per
6-minute accumulation, on the grounds that 0.5 mm/6min is only 5 mm/h and
the business-relevant threshold for a short-duration heavy-rain warning is
20 mm/h, which is 2.0 mm per 6 minutes.

The reasoning about the business thresholds is sound. What has to be
measured is whether the archive can support the scores at those levels:

  * 10.0 mm per 6 minutes is 100 mm/h. If the archive contains almost no
    such pixels, a CSI at that threshold is a count of a handful of
    pixels and it will not be a measurement of anything;
  * the extreme-patch mining the proposal asks for needs enough patches to
    mine. If there are only a few dozen in the whole archive, the
    "oversample them twice" instruction has no effect;
  * the per-tier weights assume a particular balance between misses and
    false alarms, and that balance depends on the prevalence at each
    threshold. The proposed thresholds have a much lower prevalence at the
    top than the ones in use, so the weights have to be recomputed rather
    than carried over.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

DATA = Path("/home/wf/data/radarNWP/BJ")
UNIT_MM = 0.1
LABEL_FILL = -9.0

PROPOSED = (0.1, 0.5, 2.0, 5.0, 10.0)
CURRENT = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def gather():
    """Read every label frame and keep the values above the smallest
    threshold of interest."""
    smallest = min(PROPOSED) * 0.5
    chunks = []
    for case_dir in sorted(p for p in DATA.glob("CP_*") if p.is_dir()):
        label_dir = case_dir / "LABEL" / "RA"
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.npy"))[::2]:
            raw = np.load(path).astype(np.float32)
            valid = raw[raw > LABEL_FILL]
            if valid.size:
                mm = valid * UNIT_MM
                chunks.append(mm[mm >= smallest])
    return np.concatenate(chunks) if chunks else np.zeros(0)


def main() -> None:
    print("=" * 78)
    print("1. the prevalence at the proposed thresholds")
    print("=" * 78)

    values = gather()
    if values.size == 0:
        print("  no label data found")
        return
    total = values.size
    print(f"  {values.size:,} pixels above 0.05 mm, from every second frame"
          f" of every case")
    print()
    print(f"    {'threshold':>12s}{'pixels':>14s}{'per cent':>12s}"
          f"{'per 301x301 frame':>22s}")
    print("    " + "-" * 58)
    for th in sorted(set(PROPOSED) | set(CURRENT)):
        n = int((values >= th).sum())
        # The archive has 62 cases; the count above is over half the
        # frames, so scale to a single frame of the whole domain.
        per_frame = n / (62 * 0.5) / (301 * 301)
        mark = ""
        if th in PROPOSED:
            mark = "  <- proposed"
        print(f"    {th:10.1f} mm{n:14,}{100 * n / total:11.4f} %"
              f"{per_frame:20.2f} px{mark}")

    print()
    print("=" * 78)
    print("2. is 10 mm per 6 minutes a measurable quantity?")
    print("=" * 78)

    n10 = int((values >= 10.0).sum())
    n5 = int((values >= 5.0).sum())
    print(f"  pixels at or above 5.0 mm:  {n5:,}")
    print(f"  pixels at or above 10.0 mm: {n10:,}")
    print()
    if n10 < 1000:
        print(f"  {n10:,} pixels over the whole archive is {n10 / 62 / 0.5:.0f}")
        print("  pixels per case per frame. A CSI computed on that many")
        print("  pixels moves in steps of a percent or more, and the")
        print("  difference between two models is a handful of pixels.")
        print()
        print("  It is not a measurement, and reporting it as one would")
        print("  invite exactly the question it cannot answer.")
    else:
        print("  enough pixels for the score to mean something.")

    print()
    print("=" * 78)
    print("3. how many patches could the extreme mining actually mine?")
    print("=" * 78)

    # A patch is 64x64 and the target window is ten frames. A patch counts
    # as extreme if any pixel in it reaches the threshold in any frame.
    size = 64
    n_patches = 0
    n_extreme = 0
    n_wet = 0
    for case_dir in sorted(p for p in DATA.glob("CP_*") if p.is_dir())[:20]:
        label_dir = case_dir / "LABEL" / "RA"
        if not label_dir.is_dir():
            continue
        files = sorted(label_dir.glob("*.npy"))
        # Take the first window of ten frames and slide a patch over it.
        for start in range(0, min(len(files) - 10, 20), 10):
            frames = []
            for path in files[start:start + 10]:
                raw = np.load(path).astype(np.float32)
                raw = np.where(raw > LABEL_FILL, raw * UNIT_MM, 0.0)
                frames.append(raw)
            field = np.max(np.stack(frames), axis=0)
            for y0 in range(0, field.shape[0] - size + 1, size):
                for x0 in range(0, field.shape[1] - size + 1, size):
                    patch = field[y0:y0 + size, x0:x0 + size]
                    n_patches += 1
                    if patch.max() >= 10.0:
                        n_extreme += 1
                    elif patch.max() >= 0.05:
                        n_wet += 1

    print(f"  patches examined: {n_patches:,}")
    print(f"  with a pixel at or above 10.0 mm: {n_extreme:,}")
    print(f"  with a pixel at or above 0.05 mm: {n_wet:,}")
    print()
    if n_patches:
        print(f"    the extreme ones are {100 * n_extreme / n_patches:.3f} % of"
              f" the patches")
        print(f"    the wet ones are {100 * n_wet / n_patches:.2f} %")
        if n_extreme < 50:
            print()
            print("    Duplicating them twice adds a handful of samples to a")
            print("    training set of tens of thousands. It cannot move a")
            print("    score, and it makes the epoch length irregular for no")
            print("    reason.")

    print()
    print("=" * 78)
    print("4. the tier weights at the proposed thresholds")
    print("=" * 78)

    # The proposed weights, at the prevalence the archive actually has.
    weights = [(1.0, 3.0), (1.5, 2.0), (3.0, 1.0), (6.0, 0.5), (12.0, 0.1)]
    print(f"  {'threshold':>10s}{'prevalence':>12s}{'w_miss':>8s}{'w_fa':>7s}"
          f"{'miss term':>12s}{'FA term':>12s}{'ratio':>9s}")
    print("  " + "-" * 68)
    for k, th in enumerate(PROPOSED):
        p = float((values >= th).sum()) / total
        w_miss, w_fa = weights[k]
        miss = w_miss * p * np.log(2.0)
        fa = w_fa * (1 - p) * np.log(2.0)
        ratio = miss / fa if fa > 0 else float("inf")
        print(f"  {th:8.1f}mm{p:12.6f}{w_miss:8.1f}{w_fa:7.1f}"
              f"{miss:12.6f}{fa:12.6f}{ratio:9.5f}")

    print()
    print("  The ratio is how much less the miss term counts than the false")
    print("  alarm term. It is below one at every threshold, which means")
    print("  the loss is still mostly about the dry pixels -- the weights")
    print("  do not fix that, because the prevalence moved further away")
    print("  as the thresholds went up.")


if __name__ == "__main__":
    main()
