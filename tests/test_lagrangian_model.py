"""Route R4: the advection model and the claims it has to earn.

The model carries the rainfall field forward with a motion field and
learns the growth and decay separately. Three things have to hold, and
each of them was got wrong in the proposal the model replaces.

The warp is not conservative. A bilinear resampling dilutes the dense
cells and fills the sparse ones, so the total rainfall after the warp
is not the total before it. The model therefore records the ratio, and
the test here is that it records it rather than asserting a
conservation that does not hold.

The network can ignore the advection. Without a gate the residual
absorbs the whole answer and the model becomes the baseline it was
meant to beat. The gate has to exist, and the two switches have to
produce the three arms of the ablation.

The order of the flow channels is not something the training can
discover: a model that emitted the horizontal displacement first
would advect along the wrong axis and every loss would look
perfectly reasonable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pytest_compat import pytest  # noqa: E402

from radar_nwp_bench.models import build_model, list_models  # noqa: E402
from radar_nwp_bench.models.common import make_warp_grid  # noqa: E402

MODEL = "simvp_v2_lagrangian"


def _inputs(batch: int = 2, size: int = 32):
    x_radar = torch.rand(batch, 11, 10, size, size)
    x_nwp = torch.rand(batch, 2, 27, size, size)
    return x_radar, x_nwp


def test_the_model_is_registered():
    """It has to be reachable the way every other baseline is."""
    assert MODEL in list_models()


def test_the_forecast_is_finite_and_non_negative():
    """Rain cannot be negative and cannot be nan."""
    model = build_model(MODEL)
    x_radar, x_nwp = _inputs()
    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_the_gradient_reaches_the_motion_head():
    """A motion field that is not trained is not a motion field."""
    model = build_model(MODEL)
    x_radar, x_nwp = _inputs()
    out = model(x_radar, x_nwp)
    out.sum().backward()
    assert model.flow_head[0].weight.grad is not None
    assert float(model.flow_head[0].weight.grad.abs().sum()) > 0.0
    assert model.source_sink_head[0].weight.grad is not None
    assert model.gate_head[0].weight.grad is not None


def test_the_warp_is_not_conservative_and_the_model_says_so():
    """The mass ratio has to be recorded, because it is not one.

    A warp that conserved the rainfall would keep the sum of the field.
    A bilinear resampling does not, and the model records the ratio so
    the claim can be checked instead of asserted.
    """
    model = build_model(MODEL)
    x_radar, x_nwp = _inputs()
    model(x_radar, x_nwp)
    assert model.last_mass_ratio is not None
    # the ratio is a ratio, so it is not a constant one
    assert model.last_mass_ratio != 1.0


def test_the_advection_arm_alone_still_moves_the_field():
    """With the residual off the forecast is the advection, and it differs.

    If the two arms gave the same answer the gate would be doing
    nothing and the ablation would measure nothing.
    """
    torch.manual_seed(0)
    x_radar, x_nwp = _inputs()
    full = build_model(MODEL, use_advection=True, use_source_sink=True)
    adv_only = build_model(MODEL, use_advection=True, use_source_sink=False)
    adv_only.load_state_dict(full.state_dict())

    a = full(x_radar, x_nwp)
    b = adv_only(x_radar, x_nwp)
    assert not torch.allclose(a, b)


def test_the_residual_arm_alone_is_the_gate_closed():
    """With the advection off the gate is shut and the residual is the answer."""
    model = build_model(MODEL, use_advection=False, use_source_sink=True)
    x_radar, x_nwp = _inputs()
    out = model(x_radar, x_nwp)
    assert model.last_gate_mean == 0.0
    assert torch.isfinite(out).all()


def test_the_flow_channels_are_read_as_dy_then_dx():
    """The order of the two channels is fixed, not learned.

    ``make_warp_grid`` reads the flow as (dy, dx). If the model emitted
    them the other way round, every forecast would be advected along
    the wrong axis and nothing would raise.

    A single marked pixel is used rather than a band: a band that moves
    off the edge is partly replaced by the border padding, and its
    centroid then moves the wrong way.
    """
    size = 16
    field = torch.zeros(1, 1, size, size)
    field[:, :, 8, 8] = 1.0

    for channel, axis in ((0, "row"), (1, "column")):
        flow = torch.zeros(1, 2, size, size)
        flow[:, channel] = 1.0
        grid = make_warp_grid(flow, (size, size))
        moved = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="border",
            align_corners=True,
        )
        # The output pixel samples the input one step further along, so a
        # positive displacement of one moves the mark the other way.
        row, col = torch.nonzero(moved[0, 0] > 0.5)[0].tolist()
        if channel == 0:
            assert (row, col) == (7, 8), f"the first channel moved the {axis}"
        else:
            assert (row, col) == (8, 7), f"the second channel moved the {axis}"


def test_the_model_is_no_larger_than_the_baseline_it_is_compared_with():
    """The comparison has to be about the last step, not about capacity.

    The model shares the backbone of the inversion baseline, so the two
    differ in the extrapolation and in three small heads. A model that
    grew by an order of magnitude would not be comparable.
    """
    lagrangian = build_model(MODEL)
    baseline = build_model("simvp_v2_qpe_guided")
    n_lag = sum(p.numel() for p in lagrangian.parameters())
    n_base = sum(p.numel() for p in baseline.parameters())
    assert n_lag < 1.5 * n_base
