"""Physical cleaning and normalisation of RADAR / LABEL / NWP fields.

The raw archives use sentinel-coded integer ranges (see docs/DATA_AUDIT.md §4).
Every function here is a *pure* array -> array transform so it can be unit-tested
against the sentinel tables without touching the filesystem.
"""
from __future__ import annotations

import numpy as np

# --- RADAR sentinel table (FORMAT/RADAR_Format_File_AI/*.txt) ------------------
RADAR_NODATA = -32768
RADAR_NOECHO = -1280
RADAR_DATA_NOECHO = -32767
RADAR_SCALE = 10.0

#: Variables stored in 0.1 dBZ and physically bounded below by the no-echo floor.
DBZ_VARS = ("CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70", "CR", "HBR")
#: Variables stored in other physical units (0.1 km, 0.1 kg/m2).
OTHER_RADAR_VARS = ("ET", "VIL")

#: HBR (and any future variable) uses a distinct low-end encoding that sits far
#: below the -1280 no-echo marker. Anything below this floor is a code, not a value.
RADAR_LOW_ENCODING_FLOOR = -1000.0

# --- LABEL / NWP ---------------------------------------------------------------
LABEL_FILL = -9
LABEL_SCALE = 0.1  # raw unit -> mm
NWP_FILL = -9999.0

#: Per-variable normalisation constants for the NWP fields.
#:
#: The NWP variables are physical quantities whose units differ from one
#: another -- a precipitable water in millimetres, a CAPE in J/kg, a
#: divergence in 1/s -- and their ranges differ accordingly. Measured on the
#: archive, they span six orders of magnitude:
#:
#:     DVG850   -0.002 .. 0.002      (divergence, 1/s)
#:     KI         16.9 .. 48.4       (K index)
#:     PWAT       32.7 .. 91.4       (mm)
#:     HT0      4268   .. 6028       (m)
#:     LFC          0  .. 11768       (m)
#:
#: Fed raw, the large fields (LFC, HT10, LCL, CAPE) are three orders of
#: magnitude above the small ones (the divergence fields), so the first
#: convolution of any model that concatenates the NWP onto the radar sees
#: essentially only the large fields, and the NWP encoder's own backbone is
#: swamped. The radar fields do not have this problem: they are normalised
#: by :func:`radar_log_transform` before the network sees them.
#:
#: Each variable is therefore centred and scaled by its own measured mean
#: and standard deviation, so every channel enters the network on the same
#: footing. The constants were measured over the whole archive (all 62
#: cases, all hours) rather than guessed, and they are frozen here so that
#: the pipeline stays deterministic and does not depend on which cases a
#: given run happens to read.
#:
#: The mean and the standard deviation are in the variable's own physical
#: unit; the scaling is what matters, and a standard deviation of zero
#: (a constant field) would divide by zero, so it is floored.
NWP_NORM: dict[str, tuple[float, float]] = {
    "CAPE": (441.075, 761.534),
    "DVG200": (4.16887e-06, 0.000107876),
    "DVG850": (-1.47369e-07, 0.000162850),
    "DVG925": (-4.58141e-06, 0.000184689),
    "HT0": (4343.730, 569.850),
    "HT10": (6100.060, 650.879),
    "HTw0": (3907.890, 534.739),
    "KI": (32.2255, 5.61110),
    "LCL": (1291.830, 846.887),
    "LFC": (1967.320, 1897.530),
    "LI300": (2.30904, 5.17984),
    "LI500": (-0.163357, 3.43495),
    "muLCL": (1789.950, 871.478),
    "PE": (32.1884, 13.1882),
    "PWAT": (44.5295, 11.2221),
    "Q1000": (13.2628, 3.48524),
    "Q700": (6.65188, 2.03302),
    "Q850": (11.0057, 2.89025),
    "RH1000": (66.5551, 18.4542),
    "RH500": (59.0031, 26.1183),
    "RH700": (69.2702, 18.8317),
    "TdSfc850": (13.1992, 5.62254),
    "TTdMax74": (12.5071, 8.19587),
    "TTdMean74": (7.18667, 4.93034),
    "WS500": (10.5132, 5.28743),
    "WS700": (7.17117, 3.87863),
    "WS925": (3.75907, 2.74068),
}

#: A version stamp for the input pipeline.
#:
#: It changes whenever the numbers the network sees change. A checkpoint
#: trained under one stamp cannot be evaluated under another: the model
#: expects the distribution it was trained on, and feeding it a differently
#: scaled field produces a number that looks like a score and is not one.
#: The training run records the stamp next to its checkpoint, and the
#: evaluation compares the two.
#:
#: ``norm-v2`` re-measured the constants over the whole archive with the
#: fill pixels handled the way the pipeline handles them. The first set
#: overstated the divergence fields' standard deviation by a factor of 2.7
#: to 4.6 -- and the divergence is exactly the dynamical signal the
#: quantum coupling is built on, so the plugin had been reading a third of
#: the signal it was supposed to read.
NWP_PIPELINE_VERSION = "norm-v2"



def clean_radar_field(values: np.ndarray, var_name: str) -> np.ndarray:
    """Decode one RADAR field into physical units.

    Args:
        values: raw int16 field as stored on disk.
        var_name: one of the ten RADAR variable names.

    Returns:
        float32 array in physical units (dBZ / km / kg m-2), with no-echo and
        nodata mapped to 0.
    """
    raw = values.astype(np.float32)
    valid = np.isfinite(raw) & (raw != RADAR_NODATA) & (raw != RADAR_DATA_NOECHO)
    # The no-echo marker and the low-end encoding both mean "nothing here".
    no_signal = (raw == RADAR_NOECHO) | (raw < RADAR_LOW_ENCODING_FLOOR)
    out = np.where(no_signal, 0.0, raw / RADAR_SCALE)
    if var_name in DBZ_VARS:
        # dBZ floor of the archive; anything below is a code, not a measurement.
        valid &= out >= -30.0
    else:
        # ET (0.1 km) and VIL (0.1 kg/m2) are non-negative by construction.
        valid &= out >= 0.0
    return np.where(valid, out, 0.0).astype(np.float32)


def clean_label_rain(values: np.ndarray) -> np.ndarray:
    """Decode one LABEL/RA field into millimetres of 6-minute accumulated rain.

    Raw unit is 0.1 mm; ``-9`` marks fill (no observation) and is mapped to 0.
    """
    raw = values.astype(np.float32)
    valid = np.isfinite(raw) & (raw != LABEL_FILL) & (raw >= 0.0)
    out = np.where(valid, raw * LABEL_SCALE, 0.0)
    return out.astype(np.float32)


def clean_nwp_field(values: np.ndarray) -> np.ndarray:
    """Decode one NWP field. Fill code ``-9999`` becomes 0."""
    raw = values.astype(np.float32)
    valid = np.isfinite(raw) & (raw > NWP_FILL)
    return np.where(valid, raw, 0.0).astype(np.float32)


def normalize_nwp_field(values: np.ndarray, var_name: str) -> np.ndarray:
    """Centre and scale one NWP field by its own measured statistics.

    The NWP variables are physical quantities in different units, and their
    ranges differ by six orders of magnitude (the divergence fields are
    around 1e-3, the LFC level around 1e4). Fed raw, the large fields
    dominate the first convolution of every model that uses them, and the
    small ones -- the divergence fields, which carry the dynamical
    signal -- contribute nothing.

    The radar fields are normalised the same way before they reach the
    network; this puts the NWP on the same footing.

    Args:
        values: physical-unit field, any shape.
        var_name: one of the NWP variable names.

    Returns:
        float32 array centred on zero with a unit standard deviation, or
        the input unchanged when the variable has no normalisation
        constant.
    """
    out = values.astype(np.float32)
    stats = NWP_NORM.get(var_name)
    if stats is None:
        return out
    mean, std = stats
    return ((out - mean) / max(std, 1e-6)).astype(np.float32)


def radar_log_transform(values: np.ndarray, var_name: str) -> np.ndarray:
    """Compress the dynamic range of a radar field for network input.

    dBZ fields are converted to a bounded, near-linear scale. Reflectivity spans
    roughly [-10, 65] dBZ; a shifted log1p keeps the no-echo floor at 0 while
    giving the tail a resolution of ~1 dBZ.

    Args:
        values: physical-unit field (any shape).
        var_name: the RADAR variable the field belongs to, which selects the
            normalisation.
    """
    out = values.astype(np.float32)
    if var_name in DBZ_VARS:
        # dBZ may be slightly negative (down to the -30 dBZ floor); clamp
        # before log1p, which would otherwise yield NaN.
        return np.log1p(np.clip(out, 0.0, None)) / 6.0
    # ET / VIL are non-negative by construction after cleaning.
    return np.log1p(np.clip(out, 0.0, None)) / 4.0


def normalize_rain(values: np.ndarray) -> np.ndarray:
    """Scale rainfall into a well-conditioned regression target.

    Rainfall is strongly zero-inflated and heavy-tailed, so a log1p transform
    both stabilises the loss and keeps the relative error meaningful for light
    rain. Inverse is :func:`denormalize_rain`.
    """
    return np.log1p(np.clip(values, 0.0, None)).astype(np.float32)


def denormalize_rain(values: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize_rain`."""
    return np.expm1(np.clip(values, None, 20.0)).astype(np.float32)
