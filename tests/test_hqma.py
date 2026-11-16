"""The properties the hybrid quantum manifold architecture claims.

The head is built so that the exceedance probabilities are nested by
arithmetic rather than by training. If that ever stops holding, the
whole premise of the head -- that it can commit to the heavy bin rather
than hedging towards the middle -- is gone.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch


def _load_module(name: str, script: str):
    import importlib.util

    path = REPO_ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# the nested exceedance probabilities
# ---------------------------------------------------------------------------
def test_the_exceedance_probabilities_are_nested():
    """P(R >= 0.1) >= P(R >= 0.5) >= ... holds by construction."""
    from radar_nwp_bench.models.common import QuantumHierarchicalRainHead

    head = QuantumHierarchicalRainHead(in_channels=8, hidden_dim=16)
    feat = torch.randn(4, 8, 12, 12)

    _, cum_probs = head(feat)

    assert cum_probs.shape == (4, 5, 12, 12)
    # The first exceedance probability is the largest, the last the
    # smallest, and no pair of them is out of order.
    for k in range(cum_probs.shape[1] - 1):
        assert torch.all(cum_probs[:, k] >= cum_probs[:, k + 1] - 1e-6), (
            f"the probabilities are not nested at bin {k}"
        )
    # A probability is a probability.
    assert torch.all(cum_probs >= -1e-6)
    assert torch.all(cum_probs <= 1.0 + 1e-6)


def test_the_head_predicts_a_non_negative_rain_rate():
    from radar_nwp_bench.models.common import QuantumHierarchicalRainHead

    head = QuantumHierarchicalRainHead(in_channels=8, hidden_dim=16)
    pred, _ = head(torch.randn(2, 8, 10, 10))
    assert pred.shape == (2, 1, 10, 10)
    assert torch.all(pred >= 0.0)
    assert torch.isfinite(pred).all()


def test_the_head_can_predict_more_rain_than_a_flat_field():
    """The head must be able to commit to the heavy bin.

    If the expectation over the bins always landed near the middle of the
    range, the head would be doing what the regression head does, and the
    whole reason for replacing it would be gone.
    """
    from radar_nwp_bench.models.common import QuantumHierarchicalRainHead

    torch.manual_seed(0)
    head = QuantumHierarchicalRainHead(in_channels=4, hidden_dim=32)
    # Push the intensity offsets up so the heavy bin dominates. The last
    # layer of the offset head is a Softplus, so the shift goes into the
    # convolution before it.
    with torch.no_grad():
        conv = head.intensity_offsets[0]
        conv.weight.fill_(0.0)
        conv.bias.fill_(0.0)
        conv.bias[-1] = 3.0
    pred, _ = head(torch.randn(2, 4, 8, 8))
    assert float(pred.mean()) > 1.0, (
        "the head cannot reach the heavy bin, so it cannot lift the bias"
    )


# ---------------------------------------------------------------------------
# the balanced loss
# ---------------------------------------------------------------------------
def test_the_balanced_loss_penalises_a_dry_forecast():
    """The loss has to be able to tell a dry forecast from a wet one."""
    from radar_nwp_bench.models.common import BalancedExtremePrecipLoss

    loss_fn = BalancedExtremePrecipLoss()
    target = torch.full((1, 2, 1, 8, 8), 3.0)

    dry = torch.zeros(1, 2, 1, 8, 8)
    wet = torch.full((1, 2, 1, 8, 8), 3.0)

    dry_loss = loss_fn(dry, target)
    wet_loss = loss_fn(wet, target)
    assert wet_loss < dry_loss, (
        "the loss does not reward a forecast that predicts the rain"
    )


def test_the_balanced_loss_is_finite_and_differentiable():
    from radar_nwp_bench.models.common import BalancedExtremePrecipLoss

    loss_fn = BalancedExtremePrecipLoss()
    # A leaf tensor, so the gradient lands on it directly.
    pred = (torch.rand(2, 2, 1, 8, 8) * 4.0).requires_grad_(True)
    target = torch.rand(2, 2, 1, 8, 8) * 4.0
    value = loss_fn(pred, target)
    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


def test_the_balanced_loss_rewards_hitting_the_heavy_rain():
    """The extreme threshold must carry more weight than the light one."""
    from radar_nwp_bench.models.common import BalancedExtremePrecipLoss

    loss_fn = BalancedExtremePrecipLoss(thresholds=(0.1, 4.0))
    # A forecast that gets the light rain right and the heavy rain wrong.
    light_right = torch.full((1, 1, 1, 4, 4), 0.3)
    # A forecast that gets both right.
    both_right = torch.full((1, 1, 1, 4, 4), 5.0)
    heavy = torch.full((1, 1, 1, 4, 4), 5.0)

    only_light = loss_fn(light_right, heavy)
    both = loss_fn(both_right, heavy)
    assert both < only_light


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def test_the_hybrid_model_is_registered_and_builds():
    from radar_nwp_bench.models import build_model, list_models

    assert "simvp_v2_hqma" in list_models()
    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    assert model.num_parameters() > 0


def test_the_hybrid_model_forecasts_the_right_shape():
    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.all(out >= 0.0)
    assert torch.isfinite(out).all()


def test_the_hybrid_model_runs_without_the_nwp():
    """The ablation has to be able to withhold the NWP."""
    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    out = model(x_radar, None)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()


def test_the_hybrid_model_uses_the_nwp():
    """The NWP branch must actually reach the output."""
    from radar_nwp_bench.models import build_model

    torch.manual_seed(0)
    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32, requires_grad=True)

    out = model(x_radar, x_nwp)
    out.sum().backward()
    assert x_nwp.grad is not None
    assert float(x_nwp.grad.abs().sum()) > 0.0, (
        "the NWP does not reach the output, so the model ignores it"
    )


def test_the_hybrid_model_refuses_a_mismatched_nwp():
    """A slice count the model was not built for must fail loudly.

    The shallow concatenation fixes the input width, so a model built for
    two slices cannot read one. Failing with a message that says so is
    better than a shape error deeper in the stack, and better than
    silently padding the field.
    """
    import pytest

    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 1, 27, 32, 32)
    with pytest.raises(ValueError, match="NWP slices"):
        model(x_radar, x_nwp)


def test_the_hybrid_model_can_be_built_for_one_slice():
    """The model has to be usable under the other alignment policies."""
    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
        nwp_slices=1,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 1, 27, 32, 32)
    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------
def test_the_loss_can_be_selected_from_the_command_line():
    train = _load_module("rnb_tr_loss", "train.py")
    parser = train.build_argparser()
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x"]
    )
    assert args.loss == "standard"
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x", "--loss", "balanced"]
    )
    assert args.loss == "balanced"


def test_the_balanced_criterion_fits_the_training_loop():
    """The loop expects a loss and a dictionary of components."""
    train = _load_module("rnb_tr_crit", "train.py")
    import argparse

    args = argparse.Namespace(loss="balanced", alpha_csi=0.5, physics_weight=0.0)
    criterion = train._build_criterion(args)
    import torch

    pred = torch.rand(2, 2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 2, 1, 8, 8)
    value, components = criterion(pred, target, model=None)
    assert torch.isfinite(value)
    assert "total" in components
    value.backward()
