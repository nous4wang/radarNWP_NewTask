"""The neighbourhood fractions term added to the score-aware objective.

The term is only worth having if it does what it claims: give a forecast
that is in the right place but slightly displaced a better score than a
pixel-wise loss would, and still punish a forecast that is in the wrong
place. These tests check that behaviour directly rather than checking the
code against itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pytest_compat import pytest  # noqa: E402

from radar_nwp_bench.metrics.losses import SoftFSSPrecipLoss  # noqa: E402


class _ConstantBase:
    """A stand-in for the score-aware objective.

    It returns a number that does not depend on the field, so whatever
    the test measures comes from the fractions term alone.
    """

    def __call__(self, pred, target, model=None, **extra):
        value = pred.new_zeros(())
        return value, {"total": 0.0}


def _loss(scales=(3, 5), alpha_fss: float = 0.4) -> SoftFSSPrecipLoss:
    return SoftFSSPrecipLoss(
        _ConstantBase(),
        thresholds=(0.1, 0.5, 2.0, 4.0),
        scales=scales,
        alpha_fss=alpha_fss,
    )


def _field(seed: int = 0, size: int = 24) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 1, size, size, generator=g) * 4.0


def test_a_perfect_forecast_scores_a_full_skill():
    """The term has to be able to reach zero.

    If the forecast equals the observation the two fraction fields are
    the same field and the ratio is zero. This is the check that catches
    an asymmetric softening: with only the forecast softened, a perfect
    forecast scores about 0.52 and the constant offset drowns the
    objective the term is added to.
    """
    truth = _field()
    loss = _loss()
    _, components = loss(truth.clone(), truth)
    assert components["fss_loss"] == pytest.approx(0.0, abs=1e-6)
    assert components["fss_skill"] == pytest.approx(1.0, abs=1e-6)


def test_a_displaced_forecast_is_penalised_less_than_a_wrong_one():
    """The whole point is that a displaced storm is not a wrong storm.

    A field shifted by a pixel has to score better than a field that is
    uncorrelated with the truth, and it has to score worse than the truth
    itself.
    """
    truth = _field()
    loss = _loss()

    _, perfect = loss(truth.clone(), truth)
    _, shifted = loss(torch.roll(truth, shifts=1, dims=-1), truth)
    _, unrelated = loss(_field(seed=99), truth)

    assert perfect["fss_loss"] < shifted["fss_loss"] < unrelated["fss_loss"]


def test_the_penalty_grows_with_the_displacement():
    """A larger displacement has to cost more.

    If it did not, the term would be rewarding a shift rather than
    tolerating a small one.
    """
    truth = _field()
    loss = _loss()
    _, one = loss(torch.roll(truth, shifts=1, dims=-1), truth)
    _, three = loss(torch.roll(truth, shifts=3, dims=-1), truth)
    assert one["fss_loss"] < three["fss_loss"]


def test_the_term_is_off_when_its_weight_is_zero():
    """The control arm: with no weight the base objective is untouched.

    This is what shows the term is doing something. Without this arm a
    difference between the two runs could be the base objective drifting
    rather than the new term.
    """
    truth = _field()
    _, components = _loss(alpha_fss=0.0)(torch.rand(1, 1, 24, 24) * 4.0, truth)
    assert "fss_loss" not in components
    assert components["total"] == 0.0


def test_the_gradient_reaches_the_forecast():
    """A term that does not train is worse than no term."""
    truth = _field()
    pred = (torch.rand(1, 1, 24, 24) * 4.0).requires_grad_(True)
    value, _ = _loss()(pred, truth)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


def test_the_boundary_does_not_penalise_a_uniform_field():
    """The edge has to be handled the same way as the interior.

    A constant field is its own neighbourhood mean everywhere, so the
    fraction of the neighbourhood that is rain is the indicator itself
    wherever it is. If the edge were padded with zeros, the fraction
    would fall off near the border and the term would push the forecast
    to be wet at the edges.
    """
    truth = torch.full((1, 1, 16, 16), 3.0)
    loss = _loss(scales=(3,), alpha_fss=1.0)
    _, components = loss(truth.clone(), truth)
    assert components["fss_loss"] == pytest.approx(0.0, abs=1e-6)


def test_the_threshold_weights_are_looked_up_by_value():
    """A different set of thresholds must not inherit another tier's weight.

    The same defect was found in the tier weights of the hierarchical
    loss: with positional indexing the 2.0 mm threshold was given the
    weights meant for 1.0 mm and the heaviest tier's weights were never
    used at all.
    """
    loss = SoftFSSPrecipLoss(
        _ConstantBase(), thresholds=(0.1, 0.5, 2.0, 4.0), alpha_fss=0.4
    )
    assert loss.threshold_weights[2.0] == 4.0
    assert loss.threshold_weights[4.0] == 6.0
    assert loss.threshold_weights[0.1] == 1.0
    # an unknown threshold gets the middle of the table, not the first
    other = SoftFSSPrecipLoss(
        _ConstantBase(), thresholds=(3.3,), alpha_fss=0.4
    )
    assert other.threshold_weights[3.3] == 2.5


def test_the_training_script_exposes_the_new_loss():
    """The switch has to be reachable from the command line."""
    source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "soft_fss" in source
    assert "--alpha-fss" in source
    assert "--fss-scales" in source
