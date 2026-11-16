"""Unit tests for the E45 gate arms (MASTER 4.47).

Two arms, each differing from the control in exactly one thing: where
the gate's value comes from.

* ``amp_monotone_gate`` -- the gate is a monotone increasing function
  of the residual, ``sigmoid((resid - thr) / T)``.  Because the mask is
  an increasing function of the residual and of nothing else,
  ``corr(mask, resid) >= 0`` holds by CONSTRUCTION rather than by
  training.
* ``amp_hard_support`` -- the gate is the indicator ``1[resid > thr]``
  with a straight-through gradient: the value is hard, the gradient is
  soft.

**S0 gate: with both switches off every arm must reproduce
``simvp_v2`` bit for bit** (``max_abs_diff == 0.0``).  The check is on
the OUTPUT with the SAME weights, not on the weights themselves: two
independently initialised models obviously hold different numbers, and
comparing them would say nothing.  A difference of more than the
mechanism would void the comparison, so this is checked here rather than
discovered on the cluster.

The arms are checked on the properties the pre-registration rests on
and on nothing else: the S0 bit equality, the monotonicity of the gate
in the residual, the non-negativity of ``corr(mask, resid)``, the
liveness of the straight-through gradient, and the output contract.
No training, no dataset, no CSI numbers.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from radar_nwp_bench.models.registry import build_model

_REPO = Path(__file__).resolve().parents[1]

#: The switch that turns each arm's mechanism off, per arm.
_SWITCHES = {
    "amp_monotone_gate": ("low_rank", "monotone_gate"),
    "amp_hard_support": ("low_rank", "hard_gate"),
}


def _config(name: str) -> dict:
    from radar_nwp_bench.config import load_config

    cfg = load_config(name)
    return dict(cfg.get("model") or {})


def _control_kwargs(kwargs: dict) -> dict:
    """The subset of the arm's kwargs that ``simvp_v2`` accepts.

    The arms take two arguments the control does not (``r_low`` and
    ``gate_temp``), so the control is built from the trunk arguments
    only.
    """
    from radar_nwp_bench.models.registry import model_class

    accepted = set(inspect.signature(model_class("simvp_v2").__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _inputs(seed: int = 4242, batch: int = 2, size: int = 64):
    """A batch in the shape the models actually receive.

    The radar is ``(B, T_in=11, C=10, H, W)`` and the NWP is
    ``(B, T_nwp=2, C=27, H, W)`` -- the NWP's time axis has two frames
    and its channel count is 27, which is the convention the dataset
    writes and ``resolve_nwp`` expects.
    """
    g = torch.Generator().manual_seed(seed)
    x_radar = torch.rand(batch, 11, 10, size, size, generator=g)
    x_nwp = torch.rand(batch, 2, 27, size, size, generator=g)
    return x_radar, x_nwp


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """The Pearson correlation of two tensors, over all their entries."""
    x = a.reshape(-1).double()
    y = b.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


# --------------------------------------------------------------------- S0


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_s0_bit_equivalence_with_simvp_v2(name):
    """With the mechanism off, the arm IS simvp_v2.

    The control is given the arm's weights, so any difference in the
    output comes from the computation graph alone.  The arm owns one
    extra head -- the gate's threshold -- which the control does not
    have; it is not read on the S0 path, but it does consume random
    numbers at construction time, so the shared weights are copied
    across rather than compared.
    """
    switches = _SWITCHES[name]
    kwargs = _config(name)
    kwargs.update({key: False for key in switches})

    control = build_model("simvp_v2", **_control_kwargs(kwargs))
    armed = build_model(name, **kwargs)

    # The two must have the same parameters under the same names, or the
    # comparison is not a single-variable one.
    c_sd = control.state_dict()
    a_sd = armed.state_dict()
    shared = set(c_sd) & set(a_sd)
    assert shared, f"no shared parameters between simvp_v2 and {name}"
    for key in shared:
        assert c_sd[key].shape == a_sd[key].shape, (
            f"{name}: {key} has shape {tuple(a_sd[key].shape)}, "
            f"simvp_v2 has {tuple(c_sd[key].shape)}"
        )

    missing, unexpected = armed.load_state_dict(c_sd, strict=False)
    assert not unexpected, f"{name} has parameters simvp_v2 does not: {unexpected}"
    assert set(missing) == {"gate_threshold.weight", "gate_threshold.bias"}, (
        f"{name}: unexpected missing parameters {sorted(missing)}"
    )

    control.eval()
    armed.eval()

    x_radar, x_nwp = _inputs()
    with torch.no_grad():
        expected = control(x_radar, x_nwp)
        got = armed(x_radar, x_nwp)
    assert got.shape == expected.shape
    assert torch.equal(got, expected), (
        f"{name} with the mechanism off is not bit-equal to simvp_v2: "
        f"max_abs_diff={(got - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_the_switches_change_the_composition(name):
    """The mechanism must actually change something, or it is a no-op.

    The comparison is made on the LOG-MM field, which is where the
    composition lives: at random initialisation the decoder's output is
    small and negative everywhere, so ``expm1`` sends every pixel below
    zero and the millimetre field is identically zero with and without
    the mechanism.  Comparing there would be comparing two zeros, and
    the comparison would depend on the seed.
    """
    kwargs = _config(name)
    switches = _SWITCHES[name]

    armed = build_model(name, **kwargs)
    off = build_model(name, **{**kwargs, **{k: False for k in switches}})
    missing, unexpected = off.load_state_dict(armed.state_dict(), strict=False)
    assert not unexpected
    assert not [k for k in missing if not k.startswith("gate_threshold")]

    armed.eval()
    off.eval()
    x_radar, x_nwp = _inputs(seed=7)
    with torch.no_grad():
        armed(x_radar, x_nwp)
        off(x_radar, x_nwp)

    def _composed(debug):
        return debug["low"] + debug["mask"] * debug["resid"]

    a = _composed(armed.last_debug)
    b = _composed(off.last_debug)
    assert float((a - b).abs().max()) > 0.0, (
        f"{name}: the switches do not change the composition"
    )
    # With the mechanism off the composition is the plain decoder output.
    assert torch.allclose(b, off.last_debug["amp"])


# ---------------------------------------------------------- the gate itself


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_the_mask_is_monotone_in_the_residual(name):
    """The gate must be an increasing function of the residual.

    This is the property the arms are built on: for a fixed threshold
    the mask never decreases when the residual increases.  The residual
    is constructed to increase along its last axis, so a monotone mask
    is one whose sorted values never decrease.
    """
    kwargs = _config(name)
    model = build_model(name, **kwargs)

    resid = torch.linspace(-3.0, 3.0, 32).view(1, 1, 1, 32)
    thr = torch.zeros(1)
    mask = model._gate_mask(resid, thr).reshape(-1)

    assert torch.all(torch.diff(mask) >= -1e-6), (
        "the mask is not monotone in the residual: "
        f"diff={torch.diff(mask)}"
    )


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_corr_mask_resid_is_not_negative(name):
    """corr(mask, resid) >= 0 on a random input.

    The mask is an increasing function of the residual, so the
    correlation cannot be negative; it is checked with a small negative
    tolerance because the correlation is a ratio of floating point sums
    and the last bits are not exact.

    The correlation is checked over the whole batch AND frame by frame.
    The per-frame check is the one the mechanism guarantees directly:
    the threshold is a single number per (window, lead time), so within
    a frame the mask is a monotone function of the residual and no
    correlation of any sign can appear.  The batch-wide check follows
    from it here because the threshold head starts at zero, so every
    frame shares the same threshold.
    """
    kwargs = _config(name)
    model = build_model(name, **kwargs)
    model.eval()

    x_radar, x_nwp = _inputs(seed=11)
    with torch.no_grad():
        model(x_radar, x_nwp)

    debug = model.last_debug
    corr = _corr(debug["mask"], debug["resid"])
    assert corr >= -1e-6, f"{name}: corr(mask, resid) = {corr}"

    mask = debug["mask"].reshape(debug["mask"].shape[0], -1)
    resid = debug["resid"].reshape(debug["resid"].shape[0], -1)
    for i, (m, r) in enumerate(zip(mask, resid)):
        frame_corr = _corr(m, r)
        assert frame_corr >= -1e-6, (
            f"{name}: frame {i} has corr(mask, resid) = {frame_corr}"
        )


def test_hard_support_mask_is_binary_and_its_gradient_is_soft():
    """A2: the value of the mask is hard, the gradient is soft.

    The indicator function is not differentiable, so the mask carries
    the gradient of the soft gate.  The gradient is checked against the
    analytic derivative of the sigmoid, which is the whole point of the
    straight-through construction.
    """
    kwargs = _config("amp_hard_support")
    model = build_model("amp_hard_support", **kwargs)

    resid = torch.linspace(-3.0, 3.0, 32, requires_grad=True).view(1, 1, 1, 32)
    thr = torch.zeros(1)
    mask = model._gate_mask(resid, thr)

    # The value is the hard one: 0.0 or 1.0, nothing in between.
    values = set(torch.unique(mask.detach()).tolist())
    assert values <= {0.0, 1.0}, f"the mask is not binary: {values}"

    # The gradient is the soft one's.
    (grad,) = torch.autograd.grad(mask.sum(), resid, retain_graph=True)
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0, "the straight-through gradient is zero"

    z = (resid.detach() - thr.view(-1, 1, 1, 1)) / model.gate_temp
    soft = torch.sigmoid(z)
    expected = soft * (1.0 - soft) / model.gate_temp
    assert torch.allclose(grad, expected, atol=1e-6), (
        "the gradient is not the soft gate's: "
        f"max_abs_diff={(grad - expected).abs().max().item()}"
    )


def test_hard_support_mask_is_binary_in_a_real_forward():
    """A2: the mask the model actually publishes is binary."""
    kwargs = _config("amp_hard_support")
    model = build_model("amp_hard_support", **kwargs)
    model.eval()

    x_radar, x_nwp = _inputs(seed=13)
    with torch.no_grad():
        model(x_radar, x_nwp)

    values = set(torch.unique(model.last_debug["mask"]).tolist())
    assert values <= {0.0, 1.0}, f"the mask is not binary: {values}"


def test_monotone_gate_mask_is_a_function_of_the_residual_only():
    """A1: the mask is monotone in the residual, not a free convolution.

    The free convolution gate of E43/E44 was the source of the
    measured ``corr(mask, amp) = -0.858320``.  This arm does NOT
    assert a sign for ``corr(mask, amp)`` -- at random initialisation
    the amplitude and the residual are different quantities and no such
    sign follows.  What does follow is that the mask is a monotone
    function of the residual, and that is what is checked here.
    """
    kwargs = _config("amp_monotone_gate")
    model = build_model("amp_monotone_gate", **kwargs)
    model.eval()

    x_radar, x_nwp = _inputs(seed=17)
    with torch.no_grad():
        model(x_radar, x_nwp)

    debug = model.last_debug
    mask = debug["mask"].reshape(debug["mask"].shape[0], -1)
    resid = debug["resid"].reshape(debug["resid"].shape[0], -1)

    # Within a frame the threshold is a single number, so sorting the
    # residual must not decrease the mask.
    for m, r in zip(mask, resid):
        order = torch.argsort(r)
        assert torch.all(torch.diff(m[order]) >= -1e-6), (
            "the mask is not a monotone function of the residual"
        )


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_the_threshold_starts_at_zero(name):
    """The threshold head is zero-initialised, so thr starts at 0.

    Both the weight and the bias of ``gate_threshold`` are zero at
    construction, which makes the threshold exactly 0 when the training
    begins.
    """
    kwargs = _config(name)
    model = build_model(name, **kwargs)

    assert float(model.gate_threshold.weight.detach().abs().sum()) == 0.0
    assert float(model.gate_threshold.bias.detach().abs().sum()) == 0.0

    model.eval()
    x_radar, x_nwp = _inputs(seed=19)
    with torch.no_grad():
        model(x_radar, x_nwp)
    assert float(model.last_debug["thr"].abs().sum()) == 0.0


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_the_threshold_is_one_number_per_window_and_lead_time(name):
    """thr is a per-frame scalar, not a per-window one.

    The decoder body is per (window, lead time), so the globally pooled
    threshold gives one number per forecast frame.  The amplitude scale
    changes with the lead time, and a per-frame calibration fits that
    mechanism better than a single number per window.
    """
    kwargs = _config(name)
    model = build_model(name, **kwargs)
    model.eval()

    x_radar, x_nwp = _inputs(seed=23)
    with torch.no_grad():
        model(x_radar, x_nwp)

    thr = model.last_debug["thr"]
    assert thr.shape == (2 * 10,), f"thr has shape {tuple(thr.shape)}"


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_the_gate_threshold_receives_a_gradient(name):
    """The threshold is learnable, so the arm is an experiment."""
    kwargs = _config(name)
    model = build_model(name, **kwargs)
    model.train()

    x_radar, x_nwp = _inputs(seed=29)
    out = model(x_radar, x_nwp)
    out.sum().backward()

    assert model.gate_threshold.weight.grad is not None
    assert float(model.gate_threshold.weight.grad.abs().sum()) > 0.0
    assert model.gate_threshold.bias.grad is not None
    assert float(model.gate_threshold.bias.grad.abs().sum()) > 0.0


# ------------------------------------------------------------ the contract


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_output_shape_and_non_negativity(name):
    """The evaluator must not learn a new shape, and rain is >= 0."""
    kwargs = _config(name)
    model = build_model(name, **kwargs)
    model.eval()

    x_radar, x_nwp = _inputs(seed=31)
    with torch.no_grad():
        out = model(x_radar, x_nwp)

    assert out.shape == (2, 10, 1, 64, 64)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_the_debug_tensors_are_published(name):
    """The diagnostics the read-outs need are on the model.

    ``last_debug`` holds detached tensors: the training loop calls the
    model over and over, and a live graph would be kept alive for
    nothing.  They stay on the model's device, so no host copy is paid
    for on every step.
    """
    kwargs = _config(name)
    model = build_model(name, **kwargs)
    model.eval()

    x_radar, x_nwp = _inputs(seed=37)
    with torch.no_grad():
        model(x_radar, x_nwp)

    debug = model.last_debug
    assert set(debug) == {"amp", "low", "resid", "mask", "thr"}
    for key, tensor in debug.items():
        assert isinstance(tensor, torch.Tensor), key
        assert not tensor.requires_grad, f"{key} still carries a graph"
        assert tensor.device == x_radar.device, key

    b, t = 2, 10
    assert debug["amp"].shape == (b * t, 1, 64, 64)
    assert debug["low"].shape == (b * t, 1, 64, 64)
    assert debug["resid"].shape == (b * t, 1, 64, 64)
    assert debug["mask"].shape == (b * t, 1, 64, 64)
    assert debug["thr"].shape == (b * t,)
    assert torch.allclose(debug["resid"], debug["amp"] - debug["low"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
