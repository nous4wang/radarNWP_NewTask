"""Does the label agree with the radar, physically?

The label's RA and the radar's CR describe the same storm at the same
instants. The question is whether the label is a six-minute
accumulation of the rain the radar shows instantaneously, which is what
makes ten label frames an hour.

The test is the Z-R relation: the rainfall that a given reflectivity
implies, against the rainfall the label actually records at that pixel.
If the label were an hourly accumulation, the two would disagree by a
factor of about ten.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

DATA = Path("/home/wf/data/radarNWP/BJ")
UNIT_MM = 0.1
LABEL_FILL = -9.0
RADAR_NODATA = -32768.0
RADAR_NOECHO = -1280.0


def stamp_of(path: Path) -> str:
    """Pull the timestamp out of a file name.

    The label is named ``CP_Label_RA_00001_0517-1518`` and the radar
    ``CP_RADA_00001_0517-1518_SA_CR``, so the timestamp is the field
    that holds a hyphen -- the last one for the label and the third from
    the end for the radar.
    """
    fields = path.stem.split("_")
    for field in fields:
        if "-" in field:
            return field
    return fields[-1]


def main() -> None:
    case = sorted(p for p in DATA.glob("CP_*") if p.is_dir())[0]
    label_files = sorted((case / "LABEL" / "RA").glob("*.npy"))
    radar_files = sorted((case / "RADAR" / "CR").glob("*.npy"))
    print(f"case {case.name}: {len(label_files)} label frames, "
          f"{len(radar_files)} radar frames")

    lm = {stamp_of(p): p for p in label_files}
    rm = {stamp_of(p): p for p in radar_files}
    shared = sorted(set(lm) & set(rm))
    print(f"shared timestamps: {len(shared)}")

    correlations = []
    for stamp in shared[:15]:
        lab = np.load(lm[stamp]).astype(np.float64)
        rad = np.load(rm[stamp]).astype(np.float64)
        lmm = np.where(lab > LABEL_FILL, lab * UNIT_MM, 0.0)
        rmm = np.where(rad > RADAR_NOECHO, rad * 0.1, 0.0)
        if lmm.std() > 0 and rmm.std() > 0:
            correlations.append(
                float(np.corrcoef(lmm.ravel(), rmm.ravel())[0, 1])
            )
    print(f"label-vs-radar correlation over {len(correlations)} frames: "
          f"mean {np.mean(correlations):+.4f}")

    print()
    print("the label against the radar at the same instant, by echo")
    print("intensity (this is the Z-R relation, measured):")
    print()
    print(f"  {'radar':>14s}{'pixels':>10s}{'label mean':>13s}"
          f"{'label max':>12s}{'Z-R implies':>14s}")
    print("  " + "-" * 63)

    # The Marshall-Palmer relation, the standard one for the Z-R of
    # convective rain in the tropics.
    def z_to_r(dbz: np.ndarray) -> np.ndarray:
        z = 10 ** (dbz / 10.0)
        return (z / 200.0) ** (1.0 / 1.6)

    # Accumulate over all the shared frames.
    sums: dict[int, list[float]] = {}
    maxima: dict[int, float] = {}
    for stamp in shared[:30]:
        lab = np.load(lm[stamp]).astype(np.float64)
        rad = np.load(rm[stamp]).astype(np.float64)
        lmm = np.where(lab > LABEL_FILL, lab * UNIT_MM, 0.0)
        rmm = np.where(rad > RADAR_NOECHO, rad * 0.1, 0.0)
        for dbz in range(15, 60, 5):
            mask = (rmm >= dbz) & (rmm < dbz + 5)
            if mask.sum() > 0:
                sums.setdefault(dbz, []).append(float(lmm[mask].mean()))
                maxima[dbz] = max(
                    maxima.get(dbz, 0.0), float(lmm[mask].max())
                )

    for dbz in sorted(sums):
        mean_label = float(np.mean(sums[dbz]))
        implied = float(z_to_r(np.array([float(dbz) + 2.5]))) * 6.0
        print(f"  {dbz:6d}-{dbz + 5:<7d}{len(sums[dbz]):10d}"
              f"{mean_label:13.2f}{maxima[dbz]:12.2f}{implied:14.2f}")

    print()
    print("  the last column is what the Marshall-Palmer relation gives")
    print("  for that echo over six minutes. The measured column is")
    print("  what the label actually records.")
    print()
    print("  if the measured column were about a tenth of the implied")
    print("  one, the label would be a six-minute accumulation. If it")
    print("  were about the same, the label would be an hour's worth.")


if __name__ == "__main__":
    main()
