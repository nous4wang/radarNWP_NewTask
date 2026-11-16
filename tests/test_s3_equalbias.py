"""Unit tests for the S3 equal-bias control tool.

The tool is shared by E41 (the output-parameterisation arm), E42 (the
loss-geometry arm) and E43 (the low-rank + sparse arm), so a defect in
it propagates into every S3 verdict that was ever read from it.

The defect these tests pin down: the two-level bootstrap derived the
number of windows from ``len(c[0])``, where ``c`` is a ``(n_win, 3)``
array of per-window contingency counts.  ``c[0]`` is ONE window's three
counts, so ``len(c[0])`` is always 3 and the resampling drew three
windows out of 538.  The interval therefore described a three-window
sampling distribution, and the point estimate could fall outside its own
interval.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "_e41_s3_equalbias.py"


def _load():
    spec = importlib.util.spec_from_file_location("_e41_s3_equalbias", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _load()


def _counts(n_win: int, hits: float, misses: float, fa: float) -> np.ndarray:
    return np.tile(np.array([hits, misses, fa], dtype=np.float64), (n_win, 1))


def test_bootstrap_reports_the_real_window_count():
    """The tool must report the number of windows it resampled.

    The bug wrote "n_windows": 3 for a 538-window evaluation.
    """
    a = [_counts(538, 10.0, 20.0, 5.0)]
    b = [_counts(538, 12.0, 18.0, 4.0)]
    out = s3.two_level_bootstrap(a, b, np.random.default_rng(1))
    assert out["n_windows"] == 538, out["n_windows"]


def test_bootstrap_interval_contains_the_point_estimate():
    """A percentile interval must contain the statistic it describes.

    With three windows the resampled distribution had a different centre
    from the full-sample estimate, which is how the defect showed up in
    the E43 output.
    """
    rng = np.random.default_rng(7)
    a = [_counts(538, 10.0, 20.0, 5.0)]
    b = [_counts(538, 12.0, 18.0, 4.0)]
    out = s3.two_level_bootstrap(a, b, rng)
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"], out


def test_bootstrap_sees_a_real_difference():
    """A genuine difference must be detected, not washed out."""
    rng = np.random.default_rng(11)
    a = [_counts(538, 40.0, 10.0, 2.0)]
    b = [_counts(538, 10.0, 40.0, 2.0)]
    out = s3.two_level_bootstrap(a, b, rng)
    assert out["point"] > 0.1, out
    assert not out["crosses_zero"], out


def test_bootstrap_finds_no_difference_between_identical_arms():
    """Identical arms must give a point of exactly zero."""
    rng = np.random.default_rng(13)
    a = [_counts(538, 25.0, 25.0, 5.0)]
    b = [_counts(538, 25.0, 25.0, 5.0)]
    out = s3.two_level_bootstrap(a, b, rng)
    assert abs(out["point"]) < 1e-12, out


def test_bootstrap_window_resampling_actually_moves_the_estimate():
    """Resampling windows must produce a non-degenerate interval.

    A degenerate interval would mean the window level was not sampled at
    all, which is the deeper failure the three-window bug introduced.
    """
    rng = np.random.default_rng(17)
    # Windows with wildly different truth: some all-wet, some all-dry.
    a = [np.vstack([_counts(269, 30.0, 1.0, 1.0), _counts(269, 1.0, 30.0, 1.0)])]
    b = [np.vstack([_counts(269, 20.0, 2.0, 1.0), _counts(269, 2.0, 20.0, 1.0)])]
    out = s3.two_level_bootstrap(a, b, rng)
    assert out["ci_hi"] - out["ci_lo"] > 0.0, out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
