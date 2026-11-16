"""Why does the dataset's first target frame have a maximum of 0.10 mm
when the archive's file for the same timestamp has 4.10 mm?

The units are verified -- the same file read through
``clean_label_rain`` and by hand gives the same numbers -- so the
difference has to come from *which* pixels the dataset hands over, or
from the target being normalised somewhere.

The candidate explanations:

  1. the dataset normalises the target and the reader has to invert it;
  2. the dataset's ``y_rain`` is a different variable than the label;
  3. the crop the dataset takes happens to sit on a dry part of the
     storm, and the archive's maximum is elsewhere;
  4. the target has been scaled by something.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.data.index import decode_stamp, scan_case
from radar_nwp_bench.data.transforms import clean_label_rain

DATA = Path("/home/wf/data/radarNWP/BJ")


def stamp_text(stamp: int) -> str:
    month_day, hour_min = decode_stamp(stamp)
    return (f"{month_day // 100:02d}{month_day % 100:02d}-"
            f"{hour_min // 100:02d}{hour_min % 100:02d}")


def main() -> None:
    ds = RadarNWPDataset(
        DATA, packed=False, split="train",
        in_len=11, out_len=10, patch_size=64, random_crop=False,
        slide_stride=64, max_samples=200,
    )
    spec = ds.samples[0]
    index = scan_case(spec.case_path)
    label_dir = spec.case_path / "LABEL" / "RA"

    print("=" * 78)
    print("the sample the dataset hands over")
    print("=" * 78)
    print(f"  case {spec.case_path.name}")
    print(f"  issue time {spec.issue_time} "
          f"({stamp_text(spec.issue_time)})")
    print(f"  the target frames' timestamps:")
    for i, idx in enumerate(spec.label_idx):
        if i < 3 or i >= len(spec.label_idx) - 1:
            print(f"    frame {i}: {index.label_times[idx]} "
                  f"({stamp_text(index.label_times[idx])})")
        elif i == 3:
            print("    ...")

    item = ds[0]
    y = item["y_rain"].numpy()
    print()
    print(f"  y_rain: shape={y.shape} min={y.min():.4f} max={y.max():.4f}")

    # The first target frame, against the archive's own file.
    first_stamp = index.label_times[spec.label_idx[0]]
    path = label_dir / f"CP_Label_RA_00001_{stamp_text(first_stamp)}.npy"
    if not path.exists():
        candidates = sorted(label_dir.glob(f"*{stamp_text(first_stamp)}*"))
        path = candidates[0] if candidates else None
    if path is None:
        print("  no archive file for the first target frame")
        return

    direct = clean_label_rain(np.load(path))
    print()
    print(f"  the archive's file {path.name}:")
    print(f"    full field: max={direct.max():.2f} mm, "
          f"wet={(direct > 0).sum()} px, mean={direct.mean():.5f}")

    # The crop the dataset took.
    print()
    print("  where the dataset's crop sits:")
    print(f"    patch_origin (y, x) = {item['patch_origin'].tolist()}")
    y0, x0 = int(item["patch_origin"][0]), int(item["patch_origin"][1])
    sz = 64
    crop = direct[y0:y0 + sz, x0:x0 + sz]
    print(f"    the archive's values in that window: max={crop.max():.2f} mm, "
          f"wet={(crop > 0).sum()} px, mean={crop.mean():.5f}")
    print(f"    the dataset's target frame 0:        max={y[0, 0].max():.2f} mm, "
          f"wet={(y[0, 0] > 0).sum()} px, mean={y[0, 0].mean():.5f}")

    print()
    if np.allclose(crop, y[0, 0], atol=1e-4):
        print("  -> IDENTICAL. The dataset's target IS the archive's label,")
        print("     cropped. The earlier 0.10 vs 4.10 was a mismatch between")
        print("     the file I compared against and the frame the dataset")
        print("     actually used.")
    else:
        print("  -> they differ. Checking the normalisation:")
        ratio = np.divide(
            y[0, 0], crop,
            out=np.zeros_like(crop), where=np.abs(crop) > 1e-6,
        )
        nonzero = ratio[np.abs(ratio) > 1e-6]
        if nonzero.size:
            print(f"     the ratio y/crop: min={nonzero.min():.4f} "
                  f"max={nonzero.max():.4f} median={np.median(nonzero):.4f}")
        print(f"     the target's max {y[0,0].max():.4f} vs the crop's "
              f"{crop.max():.4f}")
        if np.allclose(np.log1p(crop), y[0, 0], atol=1e-3):
            print("     -> the target is log1p(mm); the reader has to invert it.")
        else:
            print("     -> not the log transform either. Checking the "
                  "tile stitching:")
            print(f"        the target's origin {y0, x0}, the archive's "
                  f"field is {direct.shape}")


if __name__ == "__main__":
    main()
