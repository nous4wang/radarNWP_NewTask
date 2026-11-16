"""What the archive actually contains, measured rather than read off a
format file.

The question that prompted this: the label's RA is a six-minute
accumulation and the NWP is hourly, so does anything in the archive
carry an hourly accumulated rainfall? The format file lists the
variables, but the format file is a description of the whole archive,
not of the cases in it, and it does not say what the units mean for the
fields that are not obviously one thing or another.

Specifically:

  1. Which NWP variables are actually present in the cases, and does
     the archive carry an accumulated-rainfall field at all?
  2. What does the label's RA look like -- is it a six-minute
     accumulation, and how does its magnitude compare with the hourly
     total that the same storm produces?
  3. What is PWAT, the one NWP field whose unit is millimetres? It is
     precipitable water, not rainfall, but a reader of the format file
     could reasonably mistake it for one, and the pipeline has it as an
     input.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data.index import NWP_VARS, RADAR_VARS, scan_case

DATA = Path("/home/wf/data/radarNWP/BJ")


def list_nwp_variables() -> None:
    print("=" * 78)
    print("1. which NWP variables are actually in the cases?")
    print("=" * 78)

    case_dirs = sorted(p for p in DATA.glob("CP_*") if p.is_dir())
    if not case_dirs:
        print(f"  no cases under {DATA}")
        return

    found: Counter = Counter()
    for case_dir in case_dirs[:5]:
        nwp_dir = case_dir / "NWP"
        if not nwp_dir.is_dir():
            continue
        for var_dir in nwp_dir.iterdir():
            found[var_dir.name] += 1

    print(f"  found {len(found)} distinct NWP variables in the first five cases")
    print()
    print("  the format file lists 36. The variables the pipeline reads:")
    print(f"    {len(NWP_VARS)}")
    missing = sorted(set(found) - set(NWP_VARS))
    print()
    print("  present in the archive but NOT read by the pipeline:")
    for name in missing:
        print(f"    {name}")
    print()
    print("  read by the pipeline but NOT present:")
    for name in sorted(set(NWP_VARS) - set(found)):
        print(f"    {name}")

    # Is there an accumulated-rainfall field anywhere?
    rainfall_like = [
        name for name in found
        if any(
            token in name.upper()
            for token in ("RA", "RAIN", "PCP", "PRE", "ACC", "PRCP")
        )
    ]
    print()
    if rainfall_like:
        print("  accumulated-rainfall-like variables:")
        for name in rainfall_like:
            print(f"    {name}")
    else:
        print("  NO accumulated-rainfall variable in the NWP fields.")
        print("  The archive's NWP carries the state of the atmosphere, not")
        print("  the rainfall it produced.")

    # PWAT: millimetres, but what kind?
    print()
    if "PWAT" in found:
        sample = sorted(
            (case_dirs[0] / "NWP" / "PWAT").glob("*.npy")
        )[:1]
        if sample:
            raw = np.load(sample[0]).astype(np.float64)
            valid = raw[raw > -9999.0]
            if valid.size:
                print(f"  PWAT: min={valid.min():.2f} max={valid.max():.2f} "
                      f"mean={valid.mean():.2f} mm")
                print("  precipitable water -- the water vapour in the column.")
                print("  A storm of 10 mm would need most of that column to")
                print("  fall, so PWAT bounds the rainfall but is not it.")


def check_label_semantics() -> None:
    print()
    print("=" * 78)
    print("2. the label: is RA a six-minute accumulation, and of what size?")
    print("=" * 78)

    case_dirs = sorted(p for p in DATA.glob("CP_*") if p.is_dir())
    if not case_dirs:
        return

    case = case_dirs[0]
    index = scan_case(case)
    label_dir = case / "Label"
    if not label_dir.is_dir():
        print(f"  no Label directory under {case}")
        return

    files = sorted(label_dir.glob("*.npy"))
    print(f"  {case.name}: {len(files)} label frames, "
          f"{len(index.label_times)} timestamps")

    # The label's own unit, from the format file, is 0.1 mm. So the stored
    # integers are tenths of a millimetre.
    for path in files[:3]:
        raw = np.load(path)
        print(f"    {path.name}: shape={raw.shape} dtype={raw.dtype} "
              f"min={raw.min()} max={raw.max()}")

    # The interesting number: the sum over ten consecutive frames, which
    # is the hourly total the same storm produces.
    print()
    print("  the rainfall in the first ten label frames, in mm:")
    total = 0.0
    for path in files[:10]:
        raw = np.load(path).astype(np.float64)
        valid = raw[raw > -9.0]        # -9 is the fill code
        if valid.size:
            total += float(valid.sum()) / 100.0 * 0.1   # unit is 0.1 mm
    if total:
        print(f"    the hourly accumulation over the whole 301x301 field: "
              f"{total:.1f} mm")
        print("    that is the quantity an hourly rainfall field would hold.")
        print("    The label's per-frame values are a tenth of it at most.")

    # What the pipeline sees: the label of a single frame.
    print()
    print("  the label of one frame, in mm:")
    path = files[len(files) // 2]
    raw = np.load(path).astype(np.float64)
    valid = raw[raw > -9.0]
    if valid.size:
        mm = valid / 10.0     # unit is 0.1 mm
        wet = mm[mm > 0.0]
        print(f"    wet pixels: {wet.size} of {mm.size} "
              f"({100 * wet.size / mm.size:.2f} %)")
        if wet.size:
            print(f"    their rainfall: mean={wet.mean():.4f} "
                  f"max={wet.max():.2f} mm")
        print(f"    the field's mean including the dry pixels: "
              f"{mm.mean():.5f} mm")


def compare_scales() -> None:
    print()
    print("=" * 78)
    print("3. the scales, side by side")
    print("=" * 78)

    print("  label RA          0.1 mm per count, per 6-minute frame")
    print("                    a heavy cell is a few mm per frame")
    print("                    an hour is ten frames, so the hourly")
    print("                    accumulation is ten times the per-frame value")
    print()
    print("  NWP fields        mostly state variables (CAPE, humidity,")
    print("                    geopotential height, winds)")
    print("                    PWAT is in mm but is precipitable water")
    print("                    NWP valid times are whole hours")
    print()
    print("  what this means for the model:")
    print("    the model predicts the 6-minute accumulation, which is what")
    print("    the verification measures, and the NWP enters as the state")
    print("    of the atmosphere at the analysis hour and the forecast")
    print("    hour. There is no hourly rainfall field to align with.")
    print()
    print("    the quantity the model is asked to predict has a natural")
    print("    upper bound: the water vapour in the column. PWAT is")
    print("    that bound, and it is already one of the inputs.")


def main() -> None:
    list_nwp_variables()
    check_label_semantics()
    compare_scales()


if __name__ == "__main__":
    main()
