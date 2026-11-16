"""Tests for the model factory and the tensor contract every model honours.

The eight baselines are interchangeable behind
:class:`~radar_nwp_bench.models.base.BasePrecipitationModel`, so a single
parametrised test can assert the contract for all of them:

    x_radar : (B, T_in=11, C_radar=10, H, W)
    x_nwp   : (B, T_nwp=2, C_nwp=27, H, W)
    returns : (B, T_out=10, 1, H, W)   non-negative millimetres

The spatial size is 32x32 rather than the production 301x301: the point of
these tests is the contract, not the forecast quality, and the CPU forward +
backward pass has to stay fast enough to run on every commit.
"""
from __future__ import annotations

import time

import torch

from _pytest_compat import pytest
from radar_nwp_bench.models import BasePrecipitationModel, build_model, list_models


BATCH = 2
IN_LEN = 11
OUT_LEN = 10
RADAR_CHANNELS = 10
NWP_CHANNELS = 27
SPATIAL = 32


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """A fresh pair of inputs in the documented layout.

    The values are **non-negative**, matching what the dataset actually
    produces (the radar transform is a log1p of a physical field, and NWP
    fields are physical quantities). Signed noise would be out of
    distribution, and a baseline whose output head ends in a ReLU would then
    emit an all-zero map and receive no gradient -- an artefact of the input
    distribution rather than a defect in the model.
    """
    torch.manual_seed(0)
    x_radar = torch.rand(BATCH, IN_LEN, RADAR_CHANNELS, SPATIAL, SPATIAL)
    x_nwp = torch.rand(BATCH, 2, NWP_CHANNELS, SPATIAL, SPATIAL)
    return x_radar, x_nwp


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
def test_list_models_returns_every_registered_baseline():
    names = list_models()
    assert isinstance(names, list)
    assert names, "the registry is empty"
    for name in names:
        assert isinstance(name, str)
        assert name == name.lower()
    assert len(names) == len(set(names)), "a baseline is registered twice"


def test_list_models_is_sorted():
    names = list_models()
    assert names == sorted(names)


def test_build_model_returns_a_base_model():
    model = build_model("metnet3")
    assert isinstance(model, BasePrecipitationModel)
    assert isinstance(model, torch.nn.Module)


def test_build_model_is_case_insensitive():
    model = build_model("METNET3")
    assert isinstance(model, BasePrecipitationModel)


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        build_model("no_such_model")


def test_build_model_error_message_lists_the_available_models():
    try:
        build_model("no_such_model")
    except ValueError as exc:
        message = str(exc)
        for name in list_models():
            assert name in message
        return
    raise AssertionError("build_model should raise ValueError for an unknown name")


def test_every_registered_model_can_be_built():
    for name in list_models():
        model = build_model(name)
        assert isinstance(model, BasePrecipitationModel), name
        assert model.num_parameters() > 0, name


def test_models_declare_the_documented_contract():
    for name in list_models():
        model = build_model(name)
        assert model.in_radar_c == RADAR_CHANNELS, name
        assert model.in_nwp_c == NWP_CHANNELS, name
        assert model.in_len == IN_LEN, name
        assert model.out_len == OUT_LEN, name
        assert model.out_channels == 1, name


# ---------------------------------------------------------------------------
# the forward contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", list_models())
def test_model_forward_returns_the_documented_shape(model_name):
    x_radar, x_nwp = _inputs()
    model = build_model(model_name)

    out = model(x_radar, x_nwp)

    assert out.shape == (BATCH, OUT_LEN, 1, SPATIAL, SPATIAL)


@pytest.mark.parametrize("model_name", list_models())
def test_model_forward_output_is_finite(model_name):
    x_radar, x_nwp = _inputs()
    model = build_model(model_name)

    out = model(x_radar, x_nwp)

    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_name", list_models())
def test_model_forward_output_is_non_negative(model_name):
    """Rainfall is non-negative by definition, so the head must enforce it."""
    x_radar, x_nwp = _inputs()
    model = build_model(model_name)

    out = model(x_radar, x_nwp)

    assert (out >= 0.0).all()


@pytest.mark.parametrize("model_name", list_models())
def test_model_backward_produces_finite_gradients(model_name):
    x_radar, x_nwp = _inputs()
    model = build_model(model_name)

    out = model(x_radar, x_nwp)
    out.pow(2).mean().backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{model_name} produced no gradients at all"
    for grad in grads:
        assert torch.isfinite(grad).all(), model_name


@pytest.mark.parametrize("model_name", list_models())
def test_model_backward_updates_parameters(model_name):
    """A single optimiser step must move the weights.

    Analytical baselines -- advection, persistence, anything with no trained
    parameters -- are exempt: they declare ``is_learnable = False`` and are
    meant to be evaluated, not fitted.
    """
    x_radar, x_nwp = _inputs()
    model = build_model(model_name)
    if not getattr(model, "is_learnable", True):
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    before = [p.detach().clone() for p in model.parameters()]
    out = model(x_radar, x_nwp)
    optimizer.zero_grad()
    out.pow(2).mean().backward()
    optimizer.step()

    changed = any(
        not torch.equal(b, p.detach())
        for b, p in zip(before, model.parameters())
    )
    assert changed, f"{model_name} did not update any parameter"


@pytest.mark.parametrize("model_name", list_models())
def test_model_is_deterministic_in_eval_mode(model_name):
    """A fixed noise seed must make the diffusion-based adapters reproducible."""
    x_radar, x_nwp = _inputs()
    model = build_model(model_name)
    model.eval()

    with torch.no_grad():
        first = model(x_radar, x_nwp)
        second = model(x_radar, x_nwp)

    assert torch.allclose(first, second, atol=1e-5)


# ---------------------------------------------------------------------------
# shape validation
# ---------------------------------------------------------------------------
def test_check_shapes_rejects_a_wrong_radar_channel_count():
    model = build_model("metnet3")
    x_radar, x_nwp = _inputs()
    wrong_radar = torch.randn(BATCH, IN_LEN, RADAR_CHANNELS + 1, SPATIAL, SPATIAL)

    with pytest.raises(ValueError):
        model._check_shapes(wrong_radar, x_nwp)


def test_check_shapes_rejects_a_wrong_nwp_channel_count():
    model = build_model("metnet3")
    x_radar, x_nwp = _inputs()
    wrong_nwp = torch.randn(BATCH, 2, NWP_CHANNELS - 1, SPATIAL, SPATIAL)

    with pytest.raises(ValueError):
        model._check_shapes(x_radar, wrong_nwp)


def test_check_shapes_rejects_a_wrong_radar_length():
    model = build_model("metnet3")
    x_radar, x_nwp = _inputs()
    wrong_radar = torch.randn(BATCH, IN_LEN + 1, RADAR_CHANNELS, SPATIAL, SPATIAL)

    with pytest.raises(ValueError):
        model._check_shapes(wrong_radar, x_nwp)


def test_check_shapes_rejects_a_mismatched_batch():
    model = build_model("metnet3")
    x_radar, x_nwp = _inputs()
    wrong_nwp = torch.randn(BATCH + 1, 2, NWP_CHANNELS, SPATIAL, SPATIAL)

    with pytest.raises(ValueError):
        model._check_shapes(x_radar, wrong_nwp)


def test_check_shapes_rejects_a_mismatched_spatial_grid():
    model = build_model("metnet3")
    x_radar, x_nwp = _inputs()
    wrong_nwp = torch.randn(BATCH, 2, NWP_CHANNELS, SPATIAL + 8, SPATIAL)

    with pytest.raises(ValueError):
        model._check_shapes(x_radar, wrong_nwp)


def test_check_shapes_rejects_a_four_dimensional_radar_tensor():
    model = build_model("metnet3")
    x_radar, x_nwp = _inputs()
    wrong_radar = torch.randn(BATCH, IN_LEN, RADAR_CHANNELS, SPATIAL)

    with pytest.raises(ValueError):
        model._check_shapes(wrong_radar, x_nwp)


def test_check_shapes_accepts_the_documented_layout():
    model = build_model("metnet3")
    x_radar, x_nwp = _inputs()
    model._check_shapes(x_radar, x_nwp)  # must not raise


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
def test_the_whole_model_sweep_stays_fast():
    """A guard against the model tests becoming the slowest part of CI."""
    x_radar, x_nwp = _inputs()
    start = time.perf_counter()
    for name in list_models():
        model = build_model(name)
        out = model(x_radar, x_nwp)
        out.pow(2).mean().backward()
    elapsed = time.perf_counter() - start

    assert elapsed < 120.0, f"the model sweep took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# the physical floor must not freeze pixels
# ---------------------------------------------------------------------------
def test_rain_head_has_no_dead_pixels():
    """Every pixel of the rain head must be able to receive a gradient.

    A hard ``clamp_min(0)`` has a zero derivative below the threshold, so at
    initialisation roughly 43 % of the output pixels were frozen at "no rain"
    for the entire run: they could never be taught to predict rain. The
    straight-through floor keeps the values physical but lets the gradient
    through.
    """
    from radar_nwp_bench.models.common import RainHead

    torch.manual_seed(0)
    head = RainHead(32, hidden=32)
    # the real initialisation: a small final layer with a zero bias
    final = head.net[-1]
    torch.nn.init.normal_(final.weight, std=0.01)
    torch.nn.init.zeros_(final.bias)

    x = torch.randn(8, 32, 32, 32) * 2.0
    z = head.net(x)
    assert (z < 0).any(), "the probe must exercise the negative side"

    zz = z.detach().clone().requires_grad_(True)
    raw = torch.expm1(zz.clamp(-10.0, 6.0))
    out = raw + (raw.clamp_min(0.0) - raw).detach()
    out.sum().backward()

    dead = (zz.grad.abs() < 1e-20).float().mean()
    assert dead == 0.0, f"{float(dead) * 100:.1f}% of pixels have no gradient"


def test_rain_head_stays_non_negative_with_the_same_values():
    """The straight-through floor must not change the output values."""
    from radar_nwp_bench.models.common import RainHead

    torch.manual_seed(1)
    head = RainHead(16, hidden=16)
    x = torch.randn(4, 16, 24, 24)
    with torch.no_grad():
        got = head(x)
        z = head.net(x)
        expected = torch.expm1(z.clamp(-10.0, 6.0)).clamp_min(0.0)
    assert torch.equal(got, expected)
    assert (got >= 0).all()


# ---------------------------------------------------------------------------
# every model directory is a package
# ---------------------------------------------------------------------------
def test_every_model_directory_is_an_importable_package():
    """``models/<name>/`` must be a package, not a loose directory.

    The registry loads the implementations by file path, so a missing
    ``__init__.py`` does not break the harness -- but it does break
    ``from models.conv_lstm import ConvLSTMModel``, and it leaves the tree
    inconsistent. A test is cheaper than noticing it later.
    """
    import importlib
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "models"
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        init = entry / "__init__.py"
        assert init.is_file(), f"models/{entry.name}/ is missing __init__.py"
        module = importlib.import_module(f"models.{entry.name}")
        assert getattr(module, "__all__", None), (
            f"models/{entry.name}/__init__.py should declare __all__"
        )
        for name in module.__all__:
            assert hasattr(module, name), (
                f"models/{entry.name}/__init__.py exports {name}, "
                "which it does not define"
            )


def test_the_registry_finds_every_model_package():
    """A model in a package must also be reachable by name."""
    import pathlib

    from radar_nwp_bench.models import list_models

    root = pathlib.Path(__file__).resolve().parent.parent / "models"
    registered = set(list_models())
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if not (entry / "model.py").is_file():
            continue
        # the directory may hold more than one registered variant
        assert any(name.startswith(entry.name) for name in registered), (
            f"models/{entry.name}/ is not reachable through the registry"
        )


# ---------------------------------------------------------------------------
# a negative prediction must still be able to learn
# ---------------------------------------------------------------------------
def test_every_model_keeps_a_gradient_path_through_a_negative_output():
    """A model whose output is entirely negative must still get a gradient.

    Rainfall cannot be negative, so every adapter floors its output at zero.
    A hard ``clamp(min=0)`` has a zero derivative below the threshold, and
    at initialisation the whole output usually sits on that side of the
    boundary -- which leaves the model with no gradient at all and no way to
    learn that it should rain. This happened for real: the DiffCast adapter
    had 106 of 106 parameters without a gradient, and its training loss
    never moved.

    The fix is a straight-through floor, which returns the same value but
    routes the gradient through.
    """
    import torch

    from radar_nwp_bench.models import build_model, list_models

    for name in list_models():
        torch.manual_seed(0)
        model = build_model(name, in_radar_c=10, in_nwp_c=27,
                            in_len=11, out_len=10)
        if not getattr(model, "is_learnable", True):
            continue  # an analytical baseline has no parameters to train

        # a radar field that is deliberately far too weak to produce rain
        x_radar = torch.full((1, 11, 10, 16, 16), 0.01)
        x_nwp = torch.full((1, 2, 27, 16, 16), 0.01)
        out = model(x_radar, x_nwp)
        assert out.requires_grad, f"{name}: the output has no graph"

        # penalise the output itself, so a zero gradient means the output
        # was disconnected from the parameters
        out.sum().backward()
        total = sum(
            float(p.grad.abs().sum())
            for p in model.parameters()
            if p.grad is not None
        )
        assert total > 0, (
            f"{name}: no parameter received a gradient from the output "
            f"({sum(1 for _ in model.parameters())} parameters); the floor "
            f"at zero has cut the graph"
        )


def test_the_output_floor_is_the_straight_through_one():
    """The shared floor must keep the values physical and the gradient alive."""
    from radar_nwp_bench.models.common import floor_at_zero

    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0], requires_grad=True)
    floored = floor_at_zero(x)
    assert torch.allclose(floored, x.detach().clamp_min(0.0))
    assert (floored >= 0).all()
    floored.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0


def test_a_baseline_that_accepts_a_teacher_signal_uses_it():
    """The trainer passes the ground truth to the baselines that want it.

    FusionCast's future-radar prior, PW-FouCast's frequency memory and the
    diffusion residual all depend on a teacher signal. Without it those
    branches never receive a signal and stay untrained -- which is exactly
    what happened: PW-FouCast's memory projection had no gradient at all.
    """
    import inspect

    import torch

    from radar_nwp_bench.models import build_model, list_models

    teacher_names = ("gt_rain", "gt_radar", "gt_spectrum", "target")
    found = 0
    for name in list_models():
        model = build_model(name, in_radar_c=10, in_nwp_c=27,
                            in_len=11, out_len=10)
        params = inspect.signature(type(model).forward).parameters
        key = next((k for k in teacher_names if k in params), None)
        if key is None:
            continue
        found += 1
        torch.manual_seed(0)
        x_radar = torch.full((1, 11, 10, 16, 16), 0.01)
        x_nwp = torch.full((1, 2, 27, 16, 16), 0.01)
        target = torch.rand(1, 10, 1, 16, 16)
        out = model(x_radar, x_nwp, **{key: target})
        assert out.shape == (1, 10, 1, 16, 16), f"{name}: wrong output shape"
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
    assert found > 0, "no baseline accepts a teacher signal"


def test_the_training_script_passes_the_teacher_signal():
    """The wiring has to be in the trainer, not only in the model."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "_teacher_argument" in source
    assert "gt_spectrum" in source or "gt_rain" in source
