"""Tests for the RADAR/NWP grid registration.

The NWP grid (167x167 @ 0.03 deg) has to be registered onto the RADAR grid
(301x301 @ 0.01 deg).  The numpy implementation is the reference; the torch
implementation must agree with it so that a model trained on one sees the
same numbers as a model trained on the other.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from radar_nwp_bench.data.coordinates import (
    NWP_SHAPE,
    NWP_TO_RADAR_SCALE,
    RADAR_SHAPE,
    GridSpec,
    bilinear_upsample_center_crop,
    parse_grid_file,
    resample_nwp_to_radar_torch,
)


def _ramp_field() -> np.ndarray:
    """A field whose value is an affine function of the row and column."""
    rows, cols = np.meshgrid(
        np.arange(NWP_SHAPE[0]), np.arange(NWP_SHAPE[1]), indexing="ij"
    )
    return (2.0 * rows + 3.0 * cols).astype(np.float32)


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
def test_upsample_maps_167_to_301():
    assert NWP_SHAPE == (167, 167)
    assert RADAR_SHAPE == (301, 301)
    field = np.zeros(NWP_SHAPE, dtype=np.float32)
    out = bilinear_upsample_center_crop(field)
    assert out.shape == RADAR_SHAPE


def test_upsample_keeps_the_channel_axis():
    field = np.zeros((4,) + NWP_SHAPE, dtype=np.float32)
    out = bilinear_upsample_center_crop(field)
    assert out.shape == (4,) + RADAR_SHAPE


def test_upsample_returns_float32():
    field = np.zeros(NWP_SHAPE, dtype=np.float32)
    out = bilinear_upsample_center_crop(field)
    assert out.dtype == np.float32


def test_upsample_rejects_a_field_of_the_wrong_shape():
    field = np.zeros((100, 100), dtype=np.float32)
    try:
        bilinear_upsample_center_crop(field)
    except ValueError:
        return
    raise AssertionError("a wrongly shaped field should raise ValueError")


def test_scale_factor_is_three():
    assert NWP_TO_RADAR_SCALE == 3


# ---------------------------------------------------------------------------
# value preservation
# ---------------------------------------------------------------------------
def test_the_centre_of_a_constant_field_is_preserved():
    field = np.full(NWP_SHAPE, 3.25, dtype=np.float32)
    out = bilinear_upsample_center_crop(field)
    centre = out[150, 150]
    assert abs(float(centre) - 3.25) < 1e-5


def test_a_constant_field_stays_constant_everywhere():
    field = np.full(NWP_SHAPE, 7.0, dtype=np.float32)
    out = bilinear_upsample_center_crop(field)
    assert np.allclose(out, 7.0, atol=1e-5)


def test_a_linear_ramp_maps_to_the_expected_centre_value():
    """A linear ramp is reproduced exactly by bilinear interpolation.

    The 501x501 intermediate is cropped to its central 301x301, so output
    pixel (i, j) samples the source at ((i + 100.5) / 3 - 0.5).
    """
    out = bilinear_upsample_center_crop(_ramp_field())
    centre = out[150, 150]

    src_row = (150 + 100.5) / 3.0 - 0.5
    src_col = (150 + 100.5) / 3.0 - 0.5
    expected = 2.0 * src_row + 3.0 * src_col
    assert abs(float(centre) - expected) < 1e-3


def test_a_linear_ramp_is_reproduced_across_the_whole_field():
    out = bilinear_upsample_center_crop(_ramp_field())
    for i, j in ((0, 0), (0, 150), (150, 0), (150, 150), (300, 300)):
        src_row = (i + 100.5) / 3.0 - 0.5
        src_col = (j + 100.5) / 3.0 - 0.5
        expected = 2.0 * src_row + 3.0 * src_col
        assert abs(float(out[i, j]) - expected) < 1e-3, (i, j)


def test_the_upsample_is_monotonic_along_a_ramp():
    out = bilinear_upsample_center_crop(_ramp_field())
    assert np.all(np.diff(out[150, :]) > 0)
    assert np.all(np.diff(out[:, 150]) > 0)


# ---------------------------------------------------------------------------
# numpy vs torch
# ---------------------------------------------------------------------------
def test_torch_resample_agrees_with_numpy():
    rng = np.random.default_rng(0)
    field = rng.random((1,) + NWP_SHAPE).astype(np.float32)

    numpy_out = bilinear_upsample_center_crop(field[0])
    torch_out = resample_nwp_to_radar_torch(torch.from_numpy(field)).numpy()[0]

    assert torch_out.shape == numpy_out.shape
    assert np.max(np.abs(numpy_out - torch_out)) < 1e-4


def test_torch_resample_agrees_with_numpy_on_a_smooth_field():
    rows, cols = np.meshgrid(
        np.arange(NWP_SHAPE[0]), np.arange(NWP_SHAPE[1]), indexing="ij"
    )
    field = (
        100.0 + 50.0 * np.sin(rows / 20.0) + 30.0 * np.cos(cols / 25.0)
    ).astype(np.float32)[None]

    numpy_out = bilinear_upsample_center_crop(field[0])
    torch_out = resample_nwp_to_radar_torch(torch.from_numpy(field)).numpy()[0]

    assert np.max(np.abs(numpy_out - torch_out)) < 1e-4


def test_torch_resample_handles_a_batch():
    rng = np.random.default_rng(1)
    field = rng.random((3, 5) + NWP_SHAPE).astype(np.float32)
    out = resample_nwp_to_radar_torch(torch.from_numpy(field))
    assert out.shape == (3, 5) + RADAR_SHAPE


def test_torch_resample_handles_a_single_channel_stack():
    rng = np.random.default_rng(2)
    field = rng.random((5,) + NWP_SHAPE).astype(np.float32)
    out = resample_nwp_to_radar_torch(torch.from_numpy(field))
    assert out.shape == (5,) + RADAR_SHAPE


def test_torch_resample_preserves_a_constant_field():
    field = torch.full((1,) + NWP_SHAPE, 2.5, dtype=torch.float32)
    out = resample_nwp_to_radar_torch(field).numpy()[0]
    assert np.allclose(out, 2.5, atol=1e-5)


def test_torch_resample_is_differentiable():
    field = torch.zeros((1,) + NWP_SHAPE, requires_grad=True)
    out = resample_nwp_to_radar_torch(field)
    out.sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


# ---------------------------------------------------------------------------
# GridSpec / parse_grid_file
# ---------------------------------------------------------------------------
_GRID_HEADER = """StartLon = 115.0
StartLat = 40.0
EndLon = 121.01
EndLat = 34.99
resolution_X = 0.03
resolution_Y = 0.03
nx = 167
ny = 167
"""


def test_parse_grid_file_reads_the_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grid.txt"
        path.write_text(_GRID_HEADER, encoding="utf-8")

        spec = parse_grid_file(path)
        assert isinstance(spec, GridSpec)
        assert spec.start_lon == 115.0
        assert spec.start_lat == 40.0
        assert spec.end_lon == 121.01
        assert spec.end_lat == 34.99
        assert spec.res_x == 0.03
        assert spec.res_y == 0.03
        assert spec.nx == 167
        assert spec.ny == 167


def test_parse_grid_file_rejects_a_header_with_missing_keys():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grid.txt"
        path.write_text("StartLon = 115.0\n", encoding="utf-8")
        try:
            parse_grid_file(path)
        except ValueError:
            return
        raise AssertionError("an incomplete header should raise ValueError")


def test_grid_spec_centre_and_pixel_lookup():
    spec = GridSpec(
        start_lon=115.0,
        start_lat=40.0,
        end_lon=121.01,
        end_lat=34.99,
        res_x=0.03,
        res_y=0.03,
        nx=167,
        ny=167,
    )
    assert abs(spec.center_lon - 118.005) < 1e-9
    assert abs(spec.center_lat - 37.495) < 1e-9

    lon, lat = spec.lonlat_at(0, 0)
    assert abs(lon - (115.0 + 0.5 * 0.03)) < 1e-9
    assert abs(lat - (40.0 - 0.5 * 0.03)) < 1e-9


def test_grid_spec_contains_uses_the_extent():
    outer = GridSpec(115.0, 40.0, 121.01, 34.99, 0.03, 0.03, 167, 167)
    inner = GridSpec(116.0, 39.0, 119.0, 36.0, 0.01, 0.01, 301, 301)
    assert outer.contains(inner)
    assert not inner.contains(outer)
