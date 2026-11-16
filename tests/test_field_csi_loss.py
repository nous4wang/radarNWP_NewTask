"""The properties the field-wide tier loss claims.

The reason this loss exists is measured, not guessed: the per-pixel form
of the same idea loses to the plain soft-CSI loss on every score, because
the false-alarm term outweighs the miss term by seventeen to forty-two
times and the dry pixels are ninety-seven per cent of the field.

The field-wide form is the fix. It counts over the whole field and takes
the ratio, which is the score itself, and it uses the model's own
exceedance probability as the soft mask, so the model can commit to the
heavy rain where it is confident without being charged the full false
alarm for it.

These tests check that the two forms really are different, that the
field-wide one prefers the right behaviour, and that it degrades to the
plain regression for a model that has no probabilities.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch


def _model_with(probs: torch.Tensor):
    """A stand-in for the hybrid model, publishing the probabilities."""

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_cum_probs = probs

    return Model()


def test_the_field_loss_degrades_without_probabilities():
    from radar_nwp_bench.metrics.losses import HierarchicalFieldCSILoss

    loss_fn = HierarchicalFieldCSILoss(alpha_prob=0.3)
    pred = torch.rand(2, 1, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 1, 8, 8)
    value, components = loss_fn(pred, target, model=None)
    assert "tier_loss" not in components
    assert torch.isfinite(value)
    value.backward()


def test_the_field_loss_uses_the_probabilities():
    from radar_nwp_bench.metrics.losses import HierarchicalFieldCSILoss

    loss_fn = HierarchicalFieldCSILoss(alpha_prob=0.3)
    pred = torch.rand(2, 1, 1, 8, 8)
    target = torch.rand(2, 1, 1, 8, 8)

    # A model that expects heavy rain everywhere.
    heavy = _model_with(torch.full((2, 1, 4, 8, 8), 0.9))
    # A model that expects no rain anywhere.
    dry = _model_with(torch.full((2, 1, 4, 8, 8), 0.01))

    heavy_loss, heavy_components = loss_fn(pred, target, model=heavy)
    dry_loss, _ = loss_fn(pred, target, model=dry)
    assert "tier_loss" in heavy_components
    assert torch.isfinite(heavy_loss) and torch.isfinite(dry_loss)
    assert heavy_loss != dry_loss, (
        "the loss does not see the difference between the two models"
    )


def test_the_field_and_pixel_forms_differ():
    """The whole point is that they are not the same computation."""
    from radar_nwp_bench.metrics.losses import HierarchicalFieldCSILoss

    torch.manual_seed(0)
    probs = torch.rand(2, 1, 4, 8, 8)
    model = _model_with(probs)
    pred = torch.rand(2, 1, 1, 8, 8)
    target = torch.rand(2, 1, 1, 8, 8)

    field = HierarchicalFieldCSILoss(alpha_prob=0.3, field_ratio=True)
    pixel = HierarchicalFieldCSILoss(alpha_prob=0.3, field_ratio=False)

    a = field(pred, target, model=model)[0]
    b = pixel(pred, target, model=model)[0]
    assert not torch.allclose(a, b), (
        "the field-wide form and the per-pixel form are the same"
    )


def test_the_field_loss_rewards_finding_the_heavy_rain():
    """The model that puts the heavy rain where the truth is must win.

    This is the property the per-pixel form fails to have: with a binary
    mask, committing to the heavy rain is always charged the full false
    alarm. With a probability for the soft mask, a confident commitment
    is charged in proportion to how confident it is.
    """
    from radar_nwp_bench.metrics.losses import HierarchicalFieldCSILoss

    loss_fn = HierarchicalFieldCSILoss(
        thresholds=(0.1, 0.5, 2.0, 4.0), alpha_prob=1.0
    )

    # The truth: the top-right quadrant is heavy rain, the rest is dry.
    target = torch.zeros(1, 1, 1, 8, 8)
    target[0, 0, 0, 4:, 4:] = 5.0

    # A model that finds it: high probability on the heavy quadrant.
    good = torch.zeros(1, 1, 4, 8, 8)
    good[0, 0, :, 4:, 4:] = 0.95
    good[0, 0, :, :4, :4] = 0.05

    # A model that misses it but says rain everywhere else.
    bad = torch.zeros(1, 1, 4, 8, 8)
    bad[0, 0, :, 4:, 4:] = 0.05
    bad[0, 0, :, :4, :4] = 0.60

    good_loss = loss_fn(torch.zeros(1, 1, 1, 8, 8), target,
                        model=_model_with(good))[0]
    bad_loss = loss_fn(torch.zeros(1, 1, 1, 8, 8), target,
                       model=_model_with(bad))[0]
    assert good_loss < bad_loss, (
        "the loss does not prefer the model that finds the heavy rain"
    )


def test_the_field_loss_is_differentiable_through_the_probabilities():
    from radar_nwp_bench.metrics.losses import HierarchicalFieldCSILoss

    loss_fn = HierarchicalFieldCSILoss(alpha_prob=0.3)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_cum_probs = torch.rand(
                2, 1, 4, 8, 8, requires_grad=True
            )

    model = Model()
    pred = torch.rand(2, 1, 1, 8, 8)
    target = torch.rand(2, 1, 1, 8, 8)
    value, components = loss_fn(pred, target, model=model)
    assert "tier_loss" in components
    value.backward()
    assert model.last_cum_probs.grad is not None
    assert float(model.last_cum_probs.grad.abs().sum()) > 0.0


def test_the_loss_is_selectable_from_the_command_line():
    import importlib.util

    path = REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("rnb_field_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_field_train"] = module
    spec.loader.exec_module(module)

    parser = module.build_argparser()
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x",
         "--loss", "field_csi"]
    )
    assert args.loss == "field_csi"
    args = parser.parse_args(
        ["--model", "simvp_v2_hqma", "--data-root", "x",
         "--tier-thresholds", "0.1", "0.5", "2.0"]
    )
    assert args.tier_thresholds == [0.1, 0.5, 2.0]


def test_the_criterion_builds_for_the_field_loss():
    import argparse
    import importlib.util

    path = REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("rnb_field_train2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_field_train2"] = module
    spec.loader.exec_module(module)

    args = argparse.Namespace(
        loss="field_csi", alpha_prob=0.3, tier_thresholds=None,
        per_pixel=False, alpha_csi=0.5, physics_weight=0.0,
        use_focal=False,
    )
    criterion = module._build_criterion(args)
    pred = torch.rand(2, 2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 2, 1, 8, 8)
    value, components = criterion(pred, target, model=None)
    assert "tier_loss" not in components
    value.backward()
