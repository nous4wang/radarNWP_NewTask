"""Tests for the tile stitcher.

The stitcher is what makes a sliding-window evaluation meaningful. Scoring
the tiles independently would double-count every storm in the overlap strip
and score the halves of a storm that straddles a tile boundary as misses, so
the tiles have to be reassembled into the whole field first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.metrics import PatchStitcher  # noqa: E402


def _field(shape=(3, 20, 20), seed=0):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def test_a_non_overlapping_tiling_restores_the_original_field():
    """The stitched forecast must equal the source field exactly."""
    field = _field()
    stitcher = PatchStitcher((20, 20), n_frames=3, n_positions=4)
    origins = [(0, 0), (0, 10), (10, 0), (10, 10)]
    for i, (y, x) in enumerate(origins):
        tile = field[:, y:y + 10, x:x + 10]
        stitcher.add(0, i, (y, x), tile, tile)
    pred, truth = stitcher.finalize(0)
    assert np.array_equal(pred, field)
    assert np.array_equal(truth, field)


def test_the_overlap_is_averaged_not_taken_from_one_tile():
    """Two tiles disagreeing over the same pixels must be averaged."""
    stitcher = PatchStitcher((10, 10), n_frames=1, n_positions=2)
    a = np.zeros((1, 10, 10), dtype=np.float32)
    b = np.zeros((1, 10, 10), dtype=np.float32)
    b[0, 5:, :] = 4.0
    stitcher.add(0, 0, (0, 0), a, a)
    stitcher.add(0, 1, (0, 0), b, a)
    pred, _ = stitcher.finalize(0)
    assert abs(pred[0, 7, 7] - 2.0) < 1e-6, "the overlap must be the mean"
    assert abs(pred[0, 0, 0] - 0.0) < 1e-6


def test_the_truth_is_taken_once_and_is_consistent():
    """Every tile of a window shows the same underlying observation."""
    field = _field((1, 10, 10), seed=2)
    stitcher = PatchStitcher((10, 10), n_frames=1, n_positions=2)
    stitcher.add(0, 0, (0, 0), field, field)
    stitcher.add(0, 1, (0, 0), field, field)
    _, truth = stitcher.finalize(0)
    assert np.allclose(truth, field)


def test_uncovered_pixels_are_an_error_not_silence():
    """An uncovered pixel must not be scored as 'no rain'.

    Treating it as dry would flatter the forecast, because every storm in the
    uncovered strip would be a miss that is silently forgiven.
    """
    stitcher = PatchStitcher((20, 20), n_frames=1, n_positions=2)
    tile = np.zeros((1, 10, 10), dtype=np.float32)
    stitcher.add(0, 0, (0, 0), tile, tile)
    stitcher.add(0, 1, (5, 5), tile, tile)
    try:
        stitcher.finalize(0)
    except ValueError as exc:
        assert "never covered" in str(exc)
    else:
        raise AssertionError("uncovered pixels must raise")


def test_a_window_is_only_complete_once_every_tile_is_in():
    stitcher = PatchStitcher((20, 20), n_frames=1, n_positions=4)
    tile = np.zeros((1, 10, 10), dtype=np.float32)
    for i in range(3):
        stitcher.add(0, i, (0, 0), tile, tile)
    assert not stitcher.is_complete(0)
    stitcher.add(0, 3, (0, 0), tile, tile)
    assert stitcher.is_complete(0)


def test_a_tile_that_does_not_fit_is_rejected():
    stitcher = PatchStitcher((10, 10), n_frames=1, n_positions=1)
    tile = np.zeros((1, 20, 20), dtype=np.float32)
    try:
        stitcher.add(0, 0, (0, 0), tile, tile)
    except ValueError as exc:
        assert "does not fit" in str(exc)
    else:
        raise AssertionError("an oversized tile must raise")


def test_releasing_a_window_frees_its_accumulators():
    stitcher = PatchStitcher((10, 10), n_frames=1, n_positions=1)
    tile = np.zeros((1, 10, 10), dtype=np.float32)
    stitcher.add(0, 0, (0, 0), tile, tile)
    assert stitcher.pending == [0]
    stitcher.release(0)
    assert stitcher.pending == []


def test_the_real_sliding_walk_covers_the_whole_grid():
    """The 64/64 walk over the 301x301 grid must leave no pixel uncovered.

    This is the property the stitcher relies on: if the walk missed a strip,
    that strip would have to be reported as an error rather than scored.
    """
    from radar_nwp_bench.data.dataset import RadarNWPDataset

    class _Fake(RadarNWPDataset):
        def __init__(self, size, stride):
            self.patch_size = size
            self.slide_stride = stride
            self.random_crop = False
            self._patch_origins = ()
            self.set_grid_shape(301, 301)

    for size, stride in ((32, 32), (64, 64), (128, 128), (64, 32)):
        ds = _Fake(size, stride)
        stitcher = PatchStitcher((301, 301), n_frames=1,
                                 n_positions=ds.n_patch_positions)
        for i, (y, x) in enumerate(ds.patch_origins):
            tile = np.zeros((1, size, size), dtype=np.float32)
            stitcher.add(0, i, (y, x), tile, tile)
        # finalize must not raise: every pixel is covered
        pred, _ = stitcher.finalize(0)
        assert pred.shape == (1, 301, 301)
