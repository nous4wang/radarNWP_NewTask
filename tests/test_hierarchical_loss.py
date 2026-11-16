"""The properties the hierarchical loss and the six-bin head claim.

Three things have to hold, and each of them failed at least once during
development.

1. The alignment. The exceedance probabilities have to correspond to the
   thresholds they are supervised against. With one bin per threshold and
   no bin for the open-ended top interval, the first tail sum is
   ``P(R >= 0)`` -- identically one -- and every threshold is off by one
   against its probability. The six-bin head fixes that, and the
   five-bin head is kept so the difference can be measured.

2. The capability check. The loss has to find the probabilities on a
   model that has them and to degrade to the plain regression on one that
   does not, without anybody saying which model it is looking at.

3. The contract. The model's forward pass returns a single tensor, in
   training and in evaluation alike, because the evaluation script, the
   tests and the distributed path all depend on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# the alignment
# ---------------------------------------------------------------------------
def test_the_six_bin_head_aligns_the_probabilities_with_the_thresholds():
    """The tail sums must answer the questions the thresholds ask."""
    from radar_nwp_bench.models.common import QuantumHierarchicalRainHead

    head = QuantumHierarchicalRainHead(in_channels=4, hidden_dim=16,
                                       num_bins=6)
    _, cum = head(torch.randn(2, 4, 8, 8))

    assert cum.shape == (2, 5, 8, 8), (
        f"expected one probability per threshold, got {tuple(cum.shape)}"
    )
    # The first probability is P(R >= 0.1), which can be anything except
    # one; the old five-bin head had P(R >= 0) there, which is always one.
    assert float(cum[:, 0].max()) < 1.0 - 1e-6, (
        "the first probability is still P(R >= 0) -- the alignment is off"
    )
    # And the nesting still holds.
    for k in range(cum.shape[1] - 1):
        assert torch.all(cum[:, k] >= cum[:, k + 1] - 1e-6)


def test_the_five_bin_head_keeps_the_misalignment():
    """The comparison arm has to reproduce the old behaviour exactly."""
    from radar_nwp_bench.models.common import QuantumHierarchicalRainHead

    head = QuantumHierarchicalRainHead(in_channels=4, hidden_dim=16,
                                       num_bins=5)
    _, cum = head(torch.randn(2, 4, 8, 8))
    assert cum.shape == (2, 5, 8, 8)
    # P(R >= 0) is one, so the first probability cannot move.
    assert torch.allclose(
        cum[:, 0], torch.ones_like(cum[:, 0]), atol=1e-6
    ), "the five-bin head is supposed to keep the first tier dead"


def test_the_head_publishes_its_probabilities():
    """The loss reads them from the model, so the head has to keep them."""
    from radar_nwp_bench.models.common import QuantumHierarchicalRainHead

    head = QuantumHierarchicalRainHead(in_channels=4, hidden_dim=16)
    assert head.last_cum_probs is None
    head(torch.randn(2, 4, 8, 8))
    assert head.last_cum_probs is not None
    assert head.last_cum_probs.shape == (2, 5, 8, 8)


# ---------------------------------------------------------------------------
# the capability check
# ---------------------------------------------------------------------------
class _WithProbabilities(nn.Module):
    """A stand-in for the hybrid model: it publishes the probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.last_cum_probs = torch.rand(2, 1, 5, 8, 8).sort(
            dim=2, descending=True
        ).values


class _WithoutProbabilities(nn.Module):
    """A stand-in for an ordinary baseline: it has no probabilities."""


def test_the_loss_finds_the_probabilities_on_the_model():
    from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss

    loss_fn = HierarchicalPrecipLoss(alpha_prob=0.3)
    model = _WithProbabilities()
    pred = torch.rand(2, 1, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 1, 8, 8)

    value, components = loss_fn(pred, target, model=model)
    assert "prob_loss" in components, (
        "the loss did not find the probabilities on the model"
    )
    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None


def test_the_loss_degrades_on_a_model_without_probabilities():
    """An ordinary baseline must still train under this loss."""
    from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss

    loss_fn = HierarchicalPrecipLoss(alpha_prob=0.3)
    model = _WithoutProbabilities()
    pred = torch.rand(2, 1, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 1, 8, 8)

    value, components = loss_fn(pred, target, model=model)
    assert "prob_loss" not in components
    assert "reg_loss" in components
    assert torch.isfinite(value)
    value.backward()
    assert pred.grad is not None


def test_the_loss_finds_the_probabilities_on_a_submodule():
    """A model that keeps them on its head rather than on itself."""
    from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_cum_probs = torch.rand(2, 1, 5, 8, 8).sort(
                dim=2, descending=True
            ).values

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quantum_povm_head = Head()

    loss_fn = HierarchicalPrecipLoss(alpha_prob=0.3)
    pred = torch.rand(2, 1, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 1, 8, 8)
    _, components = loss_fn(pred, target, model=Model())
    assert "prob_loss" in components


def test_the_loss_without_a_model_is_the_plain_regression():
    from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss

    loss_fn = HierarchicalPrecipLoss(alpha_prob=0.3)
    pred = torch.rand(2, 1, 1, 8, 8)
    target = torch.rand(2, 1, 1, 8, 8)
    value, components = loss_fn(pred, target, model=None)
    assert "prob_loss" not in components
    assert torch.isfinite(value)


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------
def test_the_model_returns_a_single_tensor_in_both_modes():
    """The evaluation script and the tests depend on this."""
    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)

    model.train()
    out = model(x_radar, x_nwp)
    assert isinstance(out, torch.Tensor), (
        "the training forward pass must return a single tensor"
    )
    assert out.shape == (2, 10, 1, 32, 32)

    model.eval()
    with torch.no_grad():
        out = model(x_radar, x_nwp)
    assert isinstance(out, torch.Tensor), (
        "the evaluation forward pass must return a single tensor"
    )
    assert out.shape == (2, 10, 1, 32, 32)


def test_the_model_publishes_the_probabilities_with_the_right_shape():
    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    model(x_radar, x_nwp)

    probs = model.last_cum_probs
    assert probs is not None
    assert probs.shape == (2, 10, 5, 32, 32), (
        f"the probabilities have the wrong shape: {tuple(probs.shape)}"
    )


def test_the_loss_reads_the_probabilities_off_the_real_model():
    """The two halves have to find each other."""
    from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss
    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    pred = model(x_radar, x_nwp)
    target = torch.rand(2, 10, 1, 32, 32)

    loss_fn = HierarchicalPrecipLoss(alpha_prob=0.3)
    value, components = loss_fn(pred, target, model=model)
    assert "prob_loss" in components, (
        "the loss did not find the probabilities the model published"
    )
    value.backward()


def test_the_model_runs_with_the_five_bin_head():
    """The comparison arm has to build."""
    from radar_nwp_bench.models import build_model

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
        head_bins=5,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert model.last_cum_probs.shape == (2, 10, 5, 32, 32)


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------
def test_the_probability_term_penalises_a_dry_forecast():
    from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss

    loss_fn = HierarchicalPrecipLoss(alpha_prob=0.3)
    model = _WithProbabilities()
    target = torch.full((2, 1, 1, 8, 8), 5.0)

    # The model says the rain is unlikely everywhere.
    dry = _WithProbabilities()
    dry.last_cum_probs = torch.full((2, 1, 5, 8, 8), 0.01)
    # The model says the heavy rain is certain everywhere.
    heavy = _WithProbabilities()
    heavy.last_cum_probs = torch.full((2, 1, 5, 8, 8), 0.99)

    dry_loss = loss_fn(torch.zeros(2, 1, 1, 8, 8), target, model=dry)[0]
    heavy_loss = loss_fn(torch.zeros(2, 1, 1, 8, 8), target, model=heavy)[0]
    assert heavy_loss < dry_loss, (
        "the loss does not reward a model that expects the heavy rain"
    )


def test_the_focal_modulation_is_a_switch():
    """It has to change the loss, otherwise it is not measuring anything."""
    from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss

    plain = HierarchicalPrecipLoss(alpha_prob=0.3, use_focal=False)
    focal = HierarchicalPrecipLoss(alpha_prob=0.3, use_focal=True)

    model = _WithProbabilities()
    pred = torch.rand(2, 1, 1, 8, 8)
    target = torch.rand(2, 1, 1, 8, 8)

    a = plain(pred, target, model=model)[0]
    b = focal(pred, target, model=model)[0]
    assert not torch.allclose(a, b), (
        "the focal modulation changed nothing"
    )


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------
def _load_train():
    import importlib.util

    path = REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("rnb_hier_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_hier_train"] = module
    spec.loader.exec_module(module)
    return module


def test_the_hierarchical_loss_is_selectable():
    train = _load_train()
    parser = train.build_argparser()
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x"]
    )
    assert args.loss == "standard"
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x",
         "--loss", "hierarchical"]
    )
    assert args.loss == "hierarchical"


def test_the_focal_switch_has_both_positions():
    """A flag that can only be turned on cannot be turned off."""
    train = _load_train()
    parser = train.build_argparser()
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x"]
    )
    assert args.use_focal is False
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x", "--use-focal"]
    )
    assert args.use_focal is True
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x", "--no-focal"]
    )
    assert args.use_focal is False


def test_the_head_bins_flag_is_read():
    train = _load_train()
    parser = train.build_argparser()
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x"]
    )
    assert args.head_bins == 0
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x", "--head-bins", "5"]
    )
    assert args.head_bins == 5


def test_the_criterion_degrades_for_a_model_without_probabilities():
    """The build has to work for a baseline that has no head at all."""
    import argparse

    train = _load_train()
    args = argparse.Namespace(
        loss="hierarchical", alpha_prob=0.3, use_focal=False,
        alpha_csi=0.5, physics_weight=0.0,
    )
    criterion = train._build_criterion(args)
    import torch

    pred = torch.rand(2, 2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 2, 1, 8, 8)
    value, components = criterion(pred, target, model=None)
    assert "prob_loss" not in components
    value.backward()
