"""Tests for the precipitation verification scores.

All scores operate on accumulation in millimetres and are computed per
threshold.  These tests pin the contingency-table arithmetic, the two
reference cases every score must reproduce (a perfect forecast and a dry
one), and the shape of the verification report.
"""
from __future__ import annotations

import numpy as np

from radar_nwp_bench.metrics.precipitation import (
    DEFAULT_THRESHOLDS,
    PrecipitationVerifier,
    VerificationCounts,
    contingency_counts,
    fractions_skill_score,
    verify,
)


def _rain_block(size: int = 40, lo: int = 15, hi: int = 25) -> np.ndarray:
    """A square block of 5 mm on a dry background."""
    field = np.zeros((size, size), dtype=np.float32)
    field[lo:hi, lo:hi] = 5.0
    return field


def _shift_field(field: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Advect a field by a (possibly fractional) number of pixels.

    Bilinear interpolation is used so that a fractional displacement produces
    the smeared footprint a real advection produces, rather than a blocky
    one-pixel jump.
    """
    h, w = field.shape
    rows = np.arange(h) - dy
    cols = np.arange(w) - dx
    r0 = np.floor(rows).astype(int)
    c0 = np.floor(cols).astype(int)
    fr = (rows - r0)[:, None]
    fc = (cols - c0)[None, :]

    def sample(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return field[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)]

    return (
        sample(r0[:, None], c0[None, :]) * (1 - fr) * (1 - fc)
        + sample(r0[:, None], c0[None, :] + 1) * (1 - fr) * fc
        + sample(r0[:, None] + 1, c0[None, :]) * fr * (1 - fc)
        + sample(r0[:, None] + 1, c0[None, :] + 1) * fr * fc
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# contingency_counts
# ---------------------------------------------------------------------------
def test_contingency_counts_on_hand_built_arrays():
    pred = np.array([[True, True, False, False]])
    truth = np.array([[True, False, True, False]])
    counts = contingency_counts(pred, truth, 0.5)

    assert counts.hits == 1.0
    assert counts.misses == 1.0
    assert counts.false_alarms == 1.0
    assert counts.correct_negatives == 1.0
    assert counts.total == 4.0


def test_contingency_counts_all_hits():
    field = np.full((4, 4), 5.0)
    counts = contingency_counts(field, field, 0.5)
    assert counts.hits == 16.0
    assert counts.misses == 0.0
    assert counts.false_alarms == 0.0
    assert counts.correct_negatives == 0.0


def test_contingency_counts_all_misses():
    counts = contingency_counts(np.zeros((4, 4)), np.ones((4, 4)), 0.5)
    assert counts.hits == 0.0
    assert counts.misses == 16.0
    assert counts.false_alarms == 0.0
    assert counts.correct_negatives == 0.0


def test_contingency_counts_all_correct_negatives():
    counts = contingency_counts(np.zeros((4, 4)), np.zeros((4, 4)), 0.5)
    assert counts.hits == 0.0
    assert counts.misses == 0.0
    assert counts.false_alarms == 0.0
    assert counts.correct_negatives == 16.0


def test_contingency_counts_threshold_is_inclusive():
    """A pixel exactly at the threshold counts as rain."""
    field = np.array([[0.5]])
    counts = contingency_counts(field, field, 0.5)
    assert counts.hits == 1.0


def test_contingency_counts_returns_a_verification_counts():
    counts = contingency_counts(np.ones((2, 2)), np.ones((2, 2)), 0.1)
    assert isinstance(counts, VerificationCounts)


def test_verification_counts_add_accumulates():
    a = contingency_counts(np.ones((2, 2)), np.ones((2, 2)), 0.5)
    b = contingency_counts(np.ones((2, 2)), np.ones((2, 2)), 0.5)
    a.add(b)
    assert a.hits == 8.0
    assert a.total == 8.0


# ---------------------------------------------------------------------------
# CSI
# ---------------------------------------------------------------------------
def test_csi_of_a_perfect_forecast_is_one():
    field = _rain_block()
    counts = contingency_counts(field, field, 0.5)
    assert counts.csi() == 1.0


def test_csi_of_a_dry_forecast_is_zero():
    """A forecast with no rain at all scores zero against observed rain."""
    truth = _rain_block()
    counts = contingency_counts(np.zeros_like(truth), truth, 0.5)
    assert counts.csi() == 0.0


def test_csi_is_one_for_every_threshold_on_a_perfect_forecast():
    """A perfect forecast scores 1 at every threshold it actually crosses."""
    field = np.full((8, 8), 100.0, dtype=np.float32)
    for thr in DEFAULT_THRESHOLDS:
        assert contingency_counts(field, field, thr).csi() == 1.0, thr


def test_csi_is_nan_when_the_threshold_is_never_crossed():
    field = np.full((8, 8), 0.5, dtype=np.float32)
    counts = contingency_counts(field, field, 16.0)
    assert np.isnan(counts.csi())


def test_csi_is_nan_when_nothing_is_forecast_or_observed():
    counts = contingency_counts(np.zeros((4, 4)), np.zeros((4, 4)), 0.5)
    assert np.isnan(counts.csi())


def test_bias_of_a_perfect_forecast_is_one():
    field = _rain_block()
    assert contingency_counts(field, field, 0.5).bias() == 1.0


def test_pod_and_far_of_a_perfect_forecast():
    """A perfect forecast detects every event (POD 1) and raises none (FAR 0)."""
    field = _rain_block()
    counts = contingency_counts(field, field, 0.5)
    assert counts.pod() == 1.0
    assert counts.far() == 0.0


def test_pod_of_a_dry_forecast_is_zero():
    truth = _rain_block()
    counts = contingency_counts(np.zeros_like(truth), truth, 0.5)
    assert counts.pod() == 0.0


def test_as_dict_contains_the_standard_scores():
    counts = contingency_counts(np.ones((2, 2)), np.ones((2, 2)), 0.5)
    entry = counts.as_dict()
    for key in ("csi", "ets", "hss", "bias", "pod", "far", "hits", "misses", "false_alarms"):
        assert key in entry, key


# ---------------------------------------------------------------------------
# HSS -- the properties that pin the formula down
# ---------------------------------------------------------------------------
def test_hss_of_a_perfect_forecast_is_one():
    """A perfect forecast scores exactly 1 whatever the rain base rate."""
    for hits, correct_negatives in ((100, 900), (10, 990), (900, 100)):
        counts = VerificationCounts(hits, 0, 0, correct_negatives)
        assert counts.hss() == 1.0, (hits, correct_negatives)


def test_hss_of_a_climatological_forecast_is_zero():
    """A forecast that knows only the base rate has no skill."""
    # H*CN == M*FA, which is exactly the climatological case
    counts = VerificationCounts(10, 90, 90, 810)
    assert counts.hss() == 0.0


def test_hss_of_a_no_skill_forecast_is_zero():
    counts = VerificationCounts(25, 25, 25, 25)
    assert counts.ets() == 0.0
    assert counts.hss() == 0.0


def test_hss_of_a_completely_wrong_forecast_is_minus_one():
    counts = VerificationCounts(0, 50, 50, 0)
    assert counts.hss() == -1.0


def test_hss_is_bounded_between_minus_one_and_one():
    rng = np.random.default_rng(0)
    for _ in range(50):
        counts = VerificationCounts(
            float(rng.integers(0, 500)),
            float(rng.integers(0, 500)),
            float(rng.integers(0, 500)),
            float(rng.integers(0, 500)),
        )
        if counts.total > 0:
            assert -1.0 <= counts.hss() <= 1.0


# ---------------------------------------------------------------------------
# FSS
# ---------------------------------------------------------------------------
def test_fss_of_identical_fields_is_one():
    field = _rain_block()
    assert fractions_skill_score(field, field, 0.5) == 1.0


def test_fss_of_a_shifted_field_is_lower_than_one():
    """A displaced storm must be penalised by the fractions skill score."""
    truth = _rain_block()
    shifted = _shift_field(truth, 0.5, 0.5)

    perfect = fractions_skill_score(truth, truth, 0.5)
    displaced = fractions_skill_score(shifted, truth, 0.5)
    assert displaced < perfect
    assert displaced < 1.0
    assert displaced > 0.0


def test_fss_decreases_as_the_displacement_grows():
    """A larger displacement is worse than a smaller one."""
    truth = _rain_block()
    near = fractions_skill_score(_shift_field(truth, 0.5, 0.5), truth, 0.5)
    far = fractions_skill_score(_shift_field(truth, 2.5, 2.5), truth, 0.5)
    assert far <= near


def test_fss_of_a_dry_forecast_against_rain_is_zero():
    truth = _rain_block()
    assert fractions_skill_score(np.zeros_like(truth), truth, 0.5) == 0.0


def test_fss_of_two_dry_fields_is_one():
    dry = np.zeros((16, 16), dtype=np.float32)
    assert fractions_skill_score(dry, dry, 0.5) == 1.0


def test_fss_is_symmetric_for_identical_fields():
    field = _rain_block()
    assert fractions_skill_score(field, field, 0.5) == fractions_skill_score(
        field, field, 0.5
    )


def _naive_fss(pred: np.ndarray, truth: np.ndarray, thr: float, radius: int) -> float:
    """A direct, loop-based reference implementation of the FSS.

    The production implementation uses an integral image for speed; this one
    recomputes every neighbourhood explicitly, so the two must agree exactly.
    """
    p = (np.asarray(pred) >= thr).astype(float)
    t = (np.asarray(truth) >= thr).astype(float)
    h, w = p.shape
    frac_obs = np.zeros((h, w))
    frac_fcst = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            ys = range(max(0, i - radius), min(h, i + radius + 1))
            xs = range(max(0, j - radius), min(w, j + radius + 1))
            n = a = b = 0
            for y in ys:
                for x in xs:
                    n += 1
                    a += t[y, x]
                    b += p[y, x]
            frac_obs[i, j] = a / n
            frac_fcst[i, j] = b / n
    numer = ((frac_obs - frac_fcst) ** 2).sum()
    denom = (frac_obs**2 + frac_fcst**2).sum()
    return 1.0 if denom == 0 else 1.0 - numer / denom


def test_fast_fss_matches_the_naive_reference():
    """The integral-image implementation must reproduce the direct one."""
    rng = np.random.default_rng(3)
    for shape, thr, radius in (
        ((32, 32), 1.0, 3),
        ((20, 25), 1.0, 2),
        ((17, 17), 1.0, 4),
        ((40, 40), 0.5, 3),
    ):
        obs = rng.random(shape) * 3.0
        fcst = rng.random(shape) * 3.0
        fast = fractions_skill_score(fcst, obs, thr, radius=radius)
        naive = _naive_fss(fcst, obs, thr, radius=radius)
        assert abs(fast - naive) < 1e-9, (shape, radius, fast, naive)


def test_fss_of_a_displaced_storm_stays_high_at_a_coarse_scale():
    """FSS is scale-sensitive: a small displacement is forgiven, not punished.

    This is the property that separates FSS from a contingency-table score.
    """
    truth = _rain_block()
    shifted = _shift_field(truth, 1.0, 1.0)
    assert fractions_skill_score(shifted, truth, 0.5) > 0.85


def test_fss_stays_within_zero_and_one():
    rng = np.random.default_rng(5)
    for _ in range(30):
        obs = rng.random((24, 24)) * 3.0
        fcst = rng.random((24, 24)) * 3.0
        value = fractions_skill_score(fcst, obs, 1.0)
        assert -1.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# PrecipitationVerifier
# ---------------------------------------------------------------------------
def _sample_batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pred = rng.random((2, 3, 16, 16)).astype(np.float32)
    truth = rng.random((2, 3, 16, 16)).astype(np.float32)
    return pred, truth


def test_verifier_report_has_the_expected_top_level_keys():
    verifier = PrecipitationVerifier(thresholds=(0.1, 1.0))
    pred, truth = _sample_batch()
    verifier.update(pred, truth)
    report = verifier.report()

    assert "n_samples" in report
    assert "thresholds" in report
    assert "lead_csi" in report
    assert report["n_samples"] == 2


def test_verifier_report_has_one_entry_per_threshold():
    verifier = PrecipitationVerifier(thresholds=(0.1, 1.0, 4.0))
    pred, truth = _sample_batch()
    verifier.update(pred, truth)
    report = verifier.report()

    assert set(report["thresholds"]) == {"0.1", "1.0", "4.0"}


def test_verifier_threshold_entries_have_the_expected_keys():
    verifier = PrecipitationVerifier(thresholds=(0.1,))
    pred, truth = _sample_batch()
    verifier.update(pred, truth)
    entry = verifier.report()["thresholds"]["0.1"]

    for key in (
        "csi",
        "ets",
        "hss",
        "bias",
        "pod",
        "far",
        "fss",
        "hits",
        "misses",
        "false_alarms",
    ):
        assert key in entry, key


def test_verifier_reports_csi_per_lead_time():
    verifier = PrecipitationVerifier(thresholds=(0.1,))
    pred, truth = _sample_batch()
    verifier.update(pred, truth)
    lead_csi = verifier.report()["lead_csi"]

    # The lead is keyed by its string form, so the report reads the same
    # before and after a round trip through JSON -- which is where these
    # numbers end up.
    assert set(lead_csi) == {"0", "1", "2"}
    for lead, per_threshold in lead_csi.items():
        assert "0.1" in per_threshold


def test_verifier_accumulates_over_updates():
    verifier = PrecipitationVerifier(thresholds=(0.1,))
    pred, truth = _sample_batch()
    verifier.update(pred, truth)
    verifier.update(pred, truth)
    report = verifier.report()

    assert report["n_samples"] == 4


def test_verifier_accepts_a_five_dimensional_batch():
    verifier = PrecipitationVerifier(thresholds=(0.1,))
    pred, truth = _sample_batch()
    verifier.update(pred[:, :, None], truth[:, :, None])
    assert verifier.report()["n_samples"] == 2


def test_verifier_rejects_mismatched_shapes():
    verifier = PrecipitationVerifier(thresholds=(0.1,))
    pred, _ = _sample_batch()
    truth = np.zeros((2, 3, 8, 8), dtype=np.float32)
    try:
        verifier.update(pred, truth)
    except ValueError:
        return
    raise AssertionError("mismatched shapes should raise ValueError")


def test_verifier_summary_mentions_every_threshold():
    verifier = PrecipitationVerifier(thresholds=(0.1, 1.0))
    pred, truth = _sample_batch()
    verifier.update(pred, truth)
    summary = verifier.summary()

    assert "CSI" in summary
    assert "FSS" in summary
    assert "0.10" in summary
    assert "1.00" in summary


def test_verify_helper_returns_a_report():
    pred, truth = _sample_batch()
    report = verify(pred, truth, thresholds=(0.1,))
    assert "thresholds" in report
    assert report["n_samples"] == 2


def test_default_thresholds_are_the_public_benchmark_set():
    assert DEFAULT_THRESHOLDS == (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def test_the_verifier_falls_back_to_the_default_thresholds():
    """A caller with no thresholds in mind passes None.

    The evaluation does exactly that when the command line does not
    name any, and the verifier used to iterate over the None and fail at
    the first sample -- which is a long way from the mistake.
    """
    from radar_nwp_bench.metrics.precipitation import (
        DEFAULT_THRESHOLDS,
        PrecipitationVerifier,
    )

    verifier = PrecipitationVerifier(thresholds=None)
    assert verifier.thresholds == DEFAULT_THRESHOLDS
    assert set(verifier.counts) == set(DEFAULT_THRESHOLDS)
