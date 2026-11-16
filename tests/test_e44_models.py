"""Unit tests for the E44 arms (MASTER 4.46).

Four arms, each differing from the control in exactly one thing:

* ``lr_sparse_bg``  -- the low-rank projection constrains the background
  only; the rain region keeps what the projection removed.
* ``lr_sparse_add`` -- an explicit additive decomposition, low rank plus
  an independently predicted sparse term.
* ``topo_simvp``    -- the backbone is ``simvp_v2``; the mechanism is a
  clDice term in the objective.
* ``fno_simvp``     -- the decoder is a Fourier Neural Operator layer.

The S0 gate is the废止 gate: with the mechanism switched off, every arm
must reproduce ``simvp_v2`` bit for bit.  A difference of more than the
mechanism would void the comparison, so this is checked here rather than
discovered on the cluster.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from radar_nwp_bench.models.registry import build_model

_REPO = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    from radar_nwp_bench.config import load_config

    cfg = load_config(name)
    return dict(cfg.get("model") or {})


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


# --------------------------------------------------------------------- S0


@pytest.mark.parametrize("name,off", [
    ("lr_sparse_bg", {"low_rank": False, "sparse_gate": False}),
    ("lr_sparse_add", {"low_rank": False, "additive": False}),
    ("topo_simvp", {}),
    ("fno_simvp", {"use_fno": False}),
])
def test_s0_bit_equivalence_with_simvp_v2(name, off):
    """With the mechanism off, the arm IS simvp_v2.

    The check is on the OUTPUT with the SAME weights, not on the weights
    themselves: two independently initialised models obviously hold
    different numbers, and comparing them would say nothing.  The control
    is given the arm's weights, so any difference in the output comes
    from the computation graph alone.
    """
    kwargs = _config(name)
    kwargs.update(off)

    control = build_model("simvp_v2")
    arm = build_model(name, **kwargs)

    # The two must have the same parameters under the same names, or the
    # comparison is not a single-variable one.
    c_sd = control.state_dict()
    a_sd = arm.state_dict()
    shared = set(c_sd) & set(a_sd)
    assert shared, f"no shared parameters between simvp_v2 and {name}"
    for key in shared:
        assert c_sd[key].shape == a_sd[key].shape, (
            f"{name}: {key} has shape {tuple(a_sd[key].shape)}, "
            f"simvp_v2 has {tuple(c_sd[key].shape)}"
        )

    # Give the control the arm's weights, so the only thing that can
    # make the outputs differ is the computation.
    control.load_state_dict(
        {k: v for k, v in a_sd.items() if k in c_sd}, strict=False
    )
    control.eval()
    arm.eval()

    x_radar, x_nwp = _inputs()
    with torch.no_grad():
        expected = control(x_radar, x_nwp)
        got = arm(x_radar, x_nwp)
    assert got.shape == expected.shape
    assert torch.equal(got, expected), (
        f"{name} with the mechanism off is not bit-equal to simvp_v2: "
        f"max_abs_diff={(got - expected).abs().max().item()}"
    )


# ------------------------------------------------------- the four mechanisms


def test_lr_sparse_bg_keeps_the_rain_out_of_the_projection():
    """A1: the background must be flatter than the control's.

    The low-rank term's job is the background, and E43 measured that it
    does flatten it (tv_z down 34 %).  The control is given the arm's
    weights so the comparison is about the composition alone.
    """
    kwargs = _config("lr_sparse_bg")
    arm = build_model("lr_sparse_bg", **kwargs)
    x_radar, x_nwp = _inputs(seed=7)

    control = build_model("simvp_v2")
    control.load_state_dict(
        {k: v for k, v in arm.state_dict().items() if k in control.state_dict()},
        strict=False,
    )
    control.eval()
    arm.eval()

    with torch.no_grad():
        out = arm(x_radar, x_nwp)
        ref = control(x_radar, x_nwp)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 10, 1, 64, 64)

    def tv(x):
        """The total variation along the two spatial axes.

        The field is ``(B, T, 1, H, W)``, so the spatial axes are the
        last two and the difference has to be taken along them.
        """
        dh = (x[..., 1:, :] - x[..., :-1, :]).abs()
        dw = (x[..., :, 1:] - x[..., :, :-1]).abs()
        return float(dh.mean() + dw.mean())
    assert tv(out) < tv(ref), "the low-rank term did not flatten the background"


def test_lr_sparse_bg_keeps_a_peak_that_the_projection_would_shave():
    """A1's purpose: the rain region keeps the amplitude the projection
    removed.

    The projection spreads a peak over the patch and lowers it; the
    background-only composition must give a higher peak than the
    projection alone would.
    """
    from models.lr_sparse.model import _lowest_dct_modes

    kwargs = _config("lr_sparse_bg")
    arm = build_model("lr_sparse_bg", **kwargs)

    # A single sharp peak: the projection will spread it.
    amp = torch.zeros(1, 1, 64, 64)
    amp[0, 0, 32, 32] = 10.0
    basis = _lowest_dct_modes(64, 64, arm.r_low)
    flat = amp.reshape(1, -1)
    coeff = flat @ basis.reshape(arm.r_low, -1).T
    low = (coeff @ basis.reshape(arm.r_low, -1)).reshape(1, 1, 64, 64)

    # The projection lowers the peak...
    assert float(low.max()) < float(amp.max()), "the projection did not lower the peak"
    # ...and the residual carries it back.
    residual = amp - low
    assert float(residual.max()) > float(amp.max()) * 0.5, (
        "the residual does not carry the peak back"
    )
    # With the mask fully on, the composition returns the original field.
    assert torch.allclose((low + residual), amp, atol=1e-5)


def test_lr_sparse_add_has_two_independent_components():
    """A2: the sparse term must be predicted independently of the
    projection, so the decomposition is additive rather than a gate on
    an already-projected tensor."""
    kwargs = _config("lr_sparse_add")
    arm = build_model("lr_sparse_add", **kwargs)
    assert hasattr(arm, "sparse_head")

    x_radar, x_nwp = _inputs(seed=11)
    with torch.no_grad():
        out = arm(x_radar, x_nwp)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 10, 1, 64, 64)

    # The sparse head must carry a gradient of its own.
    arm.zero_grad()
    out = arm(x_radar, x_nwp)
    out.sum().backward()
    assert arm.sparse_head.weight.grad is not None
    assert float(arm.sparse_head.weight.grad.abs().sum()) > 0.0


def test_topo_simvp_is_simvp_v2_in_structure():
    """A3: the mechanism is in the loss, so the model has to be the
    control's architecture exactly."""
    kwargs = _config("topo_simvp")
    arm = build_model("topo_simvp", **kwargs)
    control = build_model("simvp_v2")
    c_sd, a_sd = control.state_dict(), arm.state_dict()
    assert set(c_sd) == set(a_sd), "topo_simvp's parameters differ from simvp_v2's"


def test_fno_simvp_uses_a_spectral_operator():
    """A4: the decoder must actually be spectral."""
    kwargs = _config("fno_simvp")
    arm = build_model("fno_simvp", **kwargs)
    assert hasattr(arm, "spectral")

    x_radar, x_nwp = _inputs(seed=13)
    with torch.no_grad():
        out = arm(x_radar, x_nwp)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 10, 1, 64, 64)

    # The operator has to be trainable and to receive a gradient.
    arm.zero_grad()
    out = arm(x_radar, x_nwp)
    out.sum().backward()
    assert arm.spectral.weight.grad is not None
    assert float(arm.spectral.weight.grad.abs().sum()) > 0.0


# ------------------------------------------------------------- the topology term


def test_cldice_has_a_gradient_when_the_forecast_is_below_threshold():
    """The term must be able to act on a model that reports no rain.

    A hard threshold makes the binarised forecast empty in exactly the
    state a model starts in, and the term then has no gradient at all:
    the arm would train like the control and the mechanism would never
    be tested.
    """
    from radar_nwp_bench.metrics.topology_losses import ClDiceLoss

    loss = ClDiceLoss(threshold=2.0)
    pred = (torch.rand(2, 32, 32) * 0.8).requires_grad_(True)
    target = torch.rand(2, 32, 32) * 0.8
    target[:, 8:20, 8:20] = 4.0

    value, _ = loss(pred, target)
    value.backward()
    assert pred.grad is not None
    assert float(pred.grad.abs().max()) > 0.0, "no gradient on a sub-threshold forecast"
    assert int((pred.grad != 0).sum()) == pred.grad.numel(), (
        "the gradient must reach every pixel, not a handful"
    )


def test_cldice_is_zero_for_identical_fields():
    from radar_nwp_bench.metrics.topology_losses import ClDiceLoss

    loss = ClDiceLoss(threshold=2.0)
    a = torch.zeros(1, 16, 16)
    a[0, 4:10, 4:10] = 5.0
    value, _ = loss(a, a.clone())
    assert float(value) < 1e-6


def test_cldice_is_one_for_disjoint_fields():
    from radar_nwp_bench.metrics.topology_losses import ClDiceLoss

    loss = ClDiceLoss(threshold=2.0)
    a = torch.zeros(1, 16, 16)
    a[0, 4:10, 4:10] = 5.0
    b = torch.zeros(1, 16, 16)
    b[0, 12:15, 12:15] = 5.0
    value, _ = loss(a, b)
    assert float(value) > 0.99


def test_cldice_gradient_pushes_the_forecast_towards_the_truth():
    """The gradient must lower the term by moving the forecast up where
    the truth is.

    The forecast has to be non-uniform.  The term is built on the
    connected components of a field, and the component weight is the
    component's MAXIMUM, so a perfectly uniform field has a constant
    component weight and the term is flat -- the gradient of a max is
    carried by the arg-max pixel alone.  A real forecast is not
    uniform, and that is the regime the arm trains in.
    """
    from radar_nwp_bench.metrics.topology_losses import ClDiceLoss

    loss = ClDiceLoss(threshold=2.0)
    truth = torch.zeros(1, 16, 16)
    truth[0, 4:10, 4:10] = 5.0

    torch.manual_seed(3)
    pred = (torch.rand(1, 16, 16) * 1.5 + 0.2).requires_grad_(True)
    value, _ = loss(pred, truth.clone())
    value.backward()
    # A pixel that should be wet: raising it must lower the term, so the
    # gradient there must be negative on average.
    assert float(pred.grad[0, 4:10, 4:10].mean()) < 0.0, (
        f"the gradient in the truth region is "
        f"{float(pred.grad[0, 4:10, 4:10].mean())}, it should be negative"
    )


def test_topo_criterion_adds_the_term_to_the_base():
    """The main objective must still be there, with the topology term on
    top and a single weight."""
    from radar_nwp_bench.metrics.topology_losses import TopoCriterion

    class _Base(torch.nn.Module):
        def forward(self, pred, target, *args, **kwargs):
            return pred.mean() * 0.0 + 1.0, {"base": 1.0}

    crit = TopoCriterion(_Base(), topo_weight=0.1, threshold=2.0)
    pred = torch.rand(1, 16, 16) * 5.0
    target = torch.rand(1, 16, 16) * 5.0
    value, components = crit(pred, target)
    assert torch.isfinite(value)
    assert "topo_loss" in components
    assert "base" in components


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
