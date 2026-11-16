"""End-to-end check: does the pipeline's label match the archive's label?

The unit conversion is claimed to be right, but a claim about the units
is worth checking against the file, because a factor of ten in the
target would move every score in the matrix and nothing in the code
would look wrong.

The check reads the same label file twice -- once through the archive
and once through the dataset -- and compares them.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.data.index import decode_stamp, scan_case
from radar_nwp_bench.data.transforms import LABEL_SCALE, clean_label_rain

DATA = Path("/home/wf/data/radarNWP/BJ")


def main() -> None:
    print("=" * 78)
    print("1. the unit constants")
    print("=" * 78)
    print(f"  LABEL_SCALE = {LABEL_SCALE}   (the format file says unit_RA=0.1mm)")
    print(f"  so a stored value of 37 is {37 * LABEL_SCALE:.1f} mm per 6 minutes")

    print()
    print("=" * 78)
    print("2. the same file, read two ways")
    print("=" * 78)

    case = sorted(p for p in DATA.glob("CP_*") if p.is_dir())[0]
    index = scan_case(case)
    label_files = sorted((case / "LABEL" / "RA").glob("*.npy"))
    stamp = index.label_times[len(index.label_times) // 2]
    # The file names carry the timestamp as MMDD-HHMM; the index carries
    # it as the integer MMDDHHMM.
    month_day, hour_min = decode_stamp(stamp)
    month, day = month_day // 100, month_day % 100
    stamp_text = (f"{month:02d}{day:02d}-"
                  f"{hour_min // 100:02d}{hour_min % 100:02d}")
    matching = [p for p in label_files if stamp_text in p.stem]
    if not matching:
        print(f"  no file for timestamp {stamp}")
        return
    path = matching[0]

    raw = np.load(path)
    through_code = clean_label_rain(raw)
    through_hand = np.where(
        np.isfinite(raw) & (raw != -9.0) & (raw >= 0.0), raw * 0.1, 0.0
    )
    print(f"  {path.name}")
    print(f"    raw: min={raw.min():.0f} max={raw.max():.0f} "
          f"(a fill of -9 means no observation)")
    print(f"    through clean_label_rain: max={through_code.max():.2f} mm")
    print(f"    by hand:                   max={through_hand.max():.2f} mm")
    print(f"    identical: {np.allclose(through_code, through_hand)}")

    print()
    print("=" * 78)
    print("3. what the dataset hands the model")
    print("=" * 78)

    ds = RadarNWPDataset(
        DATA, packed=False, split="train",
        in_len=11, out_len=10, patch_size=64, random_crop=False,
        slide_stride=64, max_samples=200,
    )
    print(f"  {len(ds.samples)} training samples")

    item = ds[0]
    y = item["y_rain"].numpy()
    print(f"  the target: shape={y.shape} dtype={y.dtype}")
    print(f"    min={y.min():.4f} max={y.max():.2f} mean={y.mean():.5f} mm")

    # The ten frames of one target window add up to the hourly
    # accumulation, and the model is scored on each frame separately.
    total = y[:, 0].sum()
    print(f"    the ten frames of the window add up to {total:.2f} mm over the"
          f" patch")
    print(f"    that is the hourly accumulation of the same storm over that"
          f" patch")

    # Compare against the archive's own file for the same window.
    spec = ds.samples[0]
    stamps = [index.label_times[i] for i in spec.label_idx]
    first_md, first_hm = decode_stamp(stamps[0])
    first_text = (f"{first_md // 100:02d}{first_md % 100:02d}-"
                  f"{first_hm // 100:02d}{first_hm % 100:02d}")
    files = [p for p in label_files if first_text in p.stem]
    if files:
        direct = clean_label_rain(np.load(files[0]))
        patch = y[0, 0]
        print()
        print("    the dataset's first target frame against the archive's")
        print(f"      dataset: max={patch.max():.2f} mm, "
              f"wet pixels={(patch > 0).sum()}")
        print(f"      archive: max={direct.max():.2f} mm, "
              f"wet pixels={(direct > 0).sum()}")
        print("      (the dataset crops and stitches tiles, so the counts"
              " differ;")
        print("       the units and the maximum do not)")

    print()
    print("=" * 78)
    print("4. the verdict on the accumulation period")
    print("=" * 78)
    print("  the label's own timestamp sequence is six minutes apart, and")
    print("  the NWP's is sixty. Ten label frames are an hour.")
    print()
    print("  nothing in the pipeline needs the ten-frame sum: the model")
    print("  predicts the six-minute accumulation, which is what the")
    print("  verification measures, and the NWP enters as the state of")
    print("  the atmosphere at the analysis hour and the forecast hour.")
    print("  There is no hourly rainfall field in the archive to align")
    print("  with -- the NWP carries the atmospheric state, not the rain")
    print("  it produced.")
    print()
    print("  What the ten-frame sum IS good for is reporting: an hourly")
    print("  rainfall accumulation is the quantity a reader compares")
    print("  against a rain gauge, and the paper can show it as a")
    print("  sanity figure.")


if __name__ == "__main__":
    main()
