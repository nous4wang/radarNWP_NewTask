"""Tests for the sentinel decoding and normalisation transforms.

The raw archive encodes "nothing here" as magic integers.  These tests pin the
decoding table from docs/DATA_AUDIT.md section 4, because a silent change
there would corrupt every downstream field without any visible error.
"""
from __future__ import annotations

import numpy as np

from radar_nwp_bench.data.transforms import (
    LABEL_FILL,
    NWP_FILL,
    RADAR_DATA_NOECHO,
    RADAR_LOW_ENCODING_FLOOR,
    RADAR_NODATA,
    RADAR_NOECHO,
    clean_label_rain,
    clean_nwp_field,
    clean_radar_field,
    denormalize_rain,
    normalize_rain,
    radar_log_transform,
)


def _field(*values: float) -> np.ndarray:
    """A 1xN int16 field holding the given raw codes."""
    return np.array([values], dtype=np.int16)


# ---------------------------------------------------------------------------
# RADAR sentinels
# ---------------------------------------------------------------------------
def test_radar_no_echo_code_becomes_zero():
    raw = _field(RADAR_NOECHO)
    out = clean_radar_field(raw, "CAP20")
    assert out[0, 0] == 0.0


def test_radar_nodata_code_becomes_zero():
    raw = _field(RADAR_NODATA)
    out = clean_radar_field(raw, "CAP20")
    assert out[0, 0] == 0.0


def test_radar_data_no_echo_code_becomes_zero():
    raw = _field(RADAR_DATA_NOECHO)
    out = clean_radar_field(raw, "CAP20")
    assert out[0, 0] == 0.0


def test_radar_sentinels_are_zero_for_every_variable():
    codes = (RADAR_NODATA, RADAR_NOECHO, RADAR_DATA_NOECHO)
    for var in ("CAP20", "CAP70", "CR", "ET", "HBR", "VIL"):
        out = clean_radar_field(_field(*codes), var)
        assert np.all(out == 0.0), var


def test_hbr_below_the_low_encoding_floor_becomes_zero():
    """HBR carries a low-end encoding well below the no-echo marker."""
    raw = _field(-28635, -28630, -28290)
    out = clean_radar_field(raw, "HBR")
    assert np.all(out == 0.0)


def test_hbr_low_encoding_floor_boundary():
    """-1000 is the floor; anything strictly below it is a code."""
    assert RADAR_LOW_ENCODING_FLOOR == -1000.0
    out = clean_radar_field(_field(-1001), "HBR")
    assert out[0, 0] == 0.0


def test_radar_value_is_scaled_by_ten():
    raw = _field(525)
    out = clean_radar_field(raw, "CAP20")
    assert out[0, 0] == 52.5


def test_radar_scaling_applies_to_every_variable():
    for var in ("CAP20", "CR", "HBR", "ET", "VIL"):
        out = clean_radar_field(_field(525), var)
        assert out[0, 0] == 52.5, var


def test_radar_decode_returns_float32():
    out = clean_radar_field(_field(525, RADAR_NOECHO), "CAP20")
    assert out.dtype == np.float32


def test_radar_decode_preserves_shape():
    raw = np.arange(-5, 5, dtype=np.int16).reshape(2, 5)
    out = clean_radar_field(raw, "CAP20")
    assert out.shape == raw.shape


def test_radar_values_below_the_dbz_floor_are_dropped():
    """A dBZ field below -30 dBZ is a code, not a measurement."""
    raw = _field(-400, -300, 300)
    out = clean_radar_field(raw, "CAP20")
    # -400 raw -> -40 dBZ -> below the floor -> 0.
    assert out[0, 0] == 0.0
    # -300 raw -> -30 dBZ -> exactly at the floor -> kept.
    assert out[0, 1] == -30.0
    assert out[0, 2] == 30.0


def test_radar_negative_values_are_dropped_for_non_dbz_variables():
    """ET and VIL are non-negative by construction."""
    out = clean_radar_field(_field(-50), "ET")
    assert out[0, 0] == 0.0


def test_radar_mixed_field_decodes_each_element_independently():
    raw = _field(RADAR_NOECHO, RADAR_NODATA, RADAR_DATA_NOECHO, -28635, 525)
    out = clean_radar_field(raw, "CAP20")
    assert np.allclose(out, np.array([[0.0, 0.0, 0.0, 0.0, 52.5]], dtype=np.float32))


# ---------------------------------------------------------------------------
# LABEL sentinels
# ---------------------------------------------------------------------------
def test_label_fill_code_becomes_zero():
    assert LABEL_FILL == -9
    out = clean_label_rain(_field(LABEL_FILL))
    assert out[0, 0] == 0.0


def test_label_value_is_scaled_by_a_tenth():
    """Raw unit is 0.1 mm, so 49 raw is 4.9 mm."""
    out = clean_label_rain(_field(49))
    assert abs(float(out[0, 0]) - 4.9) < 1e-6


def test_label_negative_values_become_zero():
    out = clean_label_rain(_field(-1, -50))
    assert np.all(out == 0.0)


def test_label_zero_stays_zero():
    out = clean_label_rain(_field(0))
    assert out[0, 0] == 0.0


def test_label_returns_float32():
    out = clean_label_rain(_field(49))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# NWP sentinels
# ---------------------------------------------------------------------------
def test_nwp_fill_code_becomes_zero():
    assert NWP_FILL == -9999.0
    out = clean_nwp_field(np.array([[NWP_FILL]], dtype=np.float32))
    assert out[0, 0] == 0.0


def test_nwp_regular_values_are_kept():
    values = np.array([[12.5, 0.0, -3.25]], dtype=np.float32)
    out = clean_nwp_field(values)
    assert np.allclose(out, values)


def test_nwp_returns_float32():
    out = clean_nwp_field(np.array([[1.0]], dtype=np.float32))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# normalise / denormalise
# ---------------------------------------------------------------------------
def test_normalize_rain_round_trip():
    values = np.array([0.0, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0], dtype=np.float32)
    restored = denormalize_rain(normalize_rain(values))
    assert np.allclose(restored, values, rtol=1e-5, atol=1e-5)


def test_normalize_rain_is_zero_at_zero():
    assert normalize_rain(np.array([0.0], dtype=np.float32))[0] == 0.0


def test_normalize_rain_clips_negative_values_to_zero():
    out = normalize_rain(np.array([-5.0, 0.0], dtype=np.float32))
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_denormalize_rain_clips_large_log_values():
    """The inverse is clipped so that a runaway log value cannot explode."""
    out = denormalize_rain(np.array([1e6], dtype=np.float32))
    assert np.isfinite(out).all()


def test_normalize_rain_returns_float32():
    out = normalize_rain(np.array([1.0], dtype=np.float32))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# radar_log_transform
# ---------------------------------------------------------------------------
def test_radar_log_transform_never_returns_nan_for_negative_inputs():
    negative = np.array([-1000.0, -300.0, -30.0, -1e-6, 0.0], dtype=np.float32)
    for var in ("CAP20", "CAP70", "CR", "ET", "HBR", "VIL"):
        out = radar_log_transform(negative, var)
        assert np.isfinite(out).all(), var
        assert not np.isnan(out).any(), var


def test_radar_log_transform_is_non_negative():
    values = np.array([-50.0, 0.0, 10.0, 65.0], dtype=np.float32)
    for var in ("CAP20", "ET", "VIL"):
        out = radar_log_transform(values, var)
        assert (out >= 0.0).all(), var


def test_radar_log_transform_clamps_negatives_to_zero():
    """A negative input clips to zero, and log1p(0) is zero."""
    out = radar_log_transform(np.array([-25.0], dtype=np.float32), "CAP20")
    assert out[0] == 0.0


def test_radar_log_transform_is_monotonic_in_the_input():
    values = np.array([0.0, 10.0, 20.0, 40.0, 65.0], dtype=np.float32)
    out = radar_log_transform(values, "CAP20")
    assert np.all(np.diff(out) > 0)


def test_radar_log_transform_returns_float32():
    out = radar_log_transform(np.array([10.0], dtype=np.float32), "CAP20")
    assert out.dtype == np.float32
