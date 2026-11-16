"""E41 / MASTER 4.43 -- small-tensor unit tests (CPU, plus one CUDA regression).

These are NOT dataset evaluations.  No training, no remote, no CSI numbers.
They verify the two single-variable ablation components behave as E41 S0/S5
requires before any compute is spent.
"""
from __future__ import annotations

import sys

import pytest
import torch

from radar_nwp_bench.metrics.struct_losses import (
    HurdleCriterion,
    HurdleNLL,
    UOTCriterion,
    UnbalancedSinkhornDivergence,
)
from radar_nwp_bench.models import build_model

KW = dict(in_radar_c=10, in_nwp_c=27, in_len=3, out_len=2,
          hid_c=8, num_blocks=1)


def _inputs(h=16, w=16, b=2):
    g = torch.Generator().manual_seed(20261109)
    x = torch.rand(b, 3, 10, h, w, generator=g)
    n = torch.rand(b, 2, 27, h, w, generator=g)
    return x, n


def _target(b=2, h=16, w=16):
    """Zero-inflated, heavy-tailed synthetic mm field."""
    g = torch.Generator().manual_seed(20261110)
    y = torch.rand(b, 2, 1, h, w, generator=g)
    y = (y - 0.85).clamp_min(0.0) * 12.0
    return y


def test_registry_builds_both_head_modes():
    rain = build_model("struct_hurdle", head_mode="rain", **KW)
    hurdle = build_model("struct_hurdle", head_mode="hurdle", **KW)
    assert rain.head_mode == "rain"
    assert hurdle.head_mode == "hurdle"
    with pytest.raises(ValueError):
        build_model("struct_hurdle", head_mode="nope", **KW)


def test_s0_rain_mode_is_bit_identical_to_simvp_v2():
    """E41 S0 gate: same weights -> same output, max |delta| < 1e-6."""
    base = build_model("simvp_v2", **KW)
    rain = build_model("struct_hurdle", head_mode="rain", **KW)
    rain.load_state_dict(base.state_dict())
    base.eval()
    rain.eval()
    x, n = _inputs()
    with torch.no_grad():
        a = base(x, n)
        b = rain(x, n)
    assert a.shape == b.shape
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert torch.max((a - b).abs()).item() < 1e-6


@pytest.mark.parametrize("head_mode", ["rain", "hurdle"])
def test_output_contract_is_float32_nonnegative_mm(head_mode):
    m = build_model("struct_hurdle", head_mode=head_mode, **KW)
    m.eval()
    x, n = _inputs()
    out = m(x, n)
    assert out.shape == (2, 2, 1, 16, 16)
    assert out.dtype == torch.float32, "float16 would corrupt the threshold tables"
    assert torch.isfinite(out).all()
    assert (out >= 0).all(), "mm field must be physically non-negative"


def test_hurdle_output_is_occurrence_times_amount():
    m = build_model("struct_hurdle", head_mode="hurdle", **KW)
    x, n = _inputs()
    out = m(x, n)
    expect = torch.sigmoid(m.last_occ_logit) * m.last_amount
    assert torch.allclose(out, expect, atol=1e-6)
    assert m.last_occ_logit.shape == out.shape
    assert m.last_amount.shape == out.shape


def test_hurdle_nll_backprops_into_both_heads():
    m = build_model("struct_hurdle", head_mode="hurdle", **KW)
    m.train()
    x, n = _inputs()
    out = m(x, n)
    crit = HurdleCriterion()
    loss, comp = crit(out, _target(), model=m)
    loss.backward()
    assert loss.requires_grad and torch.isfinite(loss)
    assert set(("occ_bce", "amount_nll", "wet_frac", "total")) <= set(comp)
    grad = m.dec[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    # the occurrence channel really receives a gradient
    assert grad[:, :, 0, 0].abs().sum() > 0


def test_hurdle_criterion_rejects_a_plain_regression_head():
    m = build_model("struct_hurdle", head_mode="rain", **KW)
    x, n = _inputs()
    out = m(x, n)
    with pytest.raises(RuntimeError, match="head_mode='hurdle'"):
        HurdleCriterion()(out, _target(), model=m)


def test_hurdle_nll_ignores_dry_pixels_for_amount():
    """Dry targets must not dominate the conditional-amount term."""
    occ = torch.zeros(1, 1, 1, 4, 4)
    amount = torch.ones(1, 1, 1, 4, 4)        # deliberately != wet truth
    dry = torch.zeros(1, 1, 1, 4, 4)
    wet = torch.ones(1, 1, 1, 4, 4) * 5.0
    l_dry, c_dry = HurdleNLL()(occ, amount, dry)
    l_wet, c_wet = HurdleNLL()(occ, amount, wet)
    assert c_dry["amount_nll"] == 0.0, "dry pixels must not supervise amount"
    assert c_wet["amount_nll"] > 0.0
    assert l_wet.item() > l_dry.item()
    assert c_dry["occ_bce"] == pytest.approx(c_wet["occ_bce"], abs=1e-6)
    assert torch.isfinite(l_dry) and torch.isfinite(l_wet)


def test_uot_is_sensitive_to_support_displacement():
    """A shifted rain core must cost more than the aligned one."""
    div = UnbalancedSinkhornDivergence(grid=8, n_iters=20)
    t = torch.zeros(1, 1, 1, 64, 64)
    t[0, 0, 0, 20:30, 20:30] = 8.0
    aligned, _ = div(t.clone(), t)
    shifted = torch.roll(t, shifts=16, dims=-1)
    displaced, _ = div(shifted, t)
    assert displaced.item() > aligned.item()


def test_uot_is_sensitive_to_mass_underreporting():
    """BIAS << 1 must show up as a mass gap, not be normalised away."""
    div = UnbalancedSinkhornDivergence(grid=8, n_iters=20)
    t = torch.zeros(1, 1, 1, 64, 64)
    t[0, 0, 0, 20:30, 20:30] = 8.0
    _, full = div(t.clone(), t)
    _, weak = div(t * 0.3, t)
    assert weak["mass_gap"] > full["mass_gap"]


def test_uot_criterion_backprops_and_reports_components():
    m = build_model("struct_hurdle", head_mode="rain", **KW)
    m.train()
    x, n = _inputs(h=32, w=32)
    out = m(x, n)
    crit = UOTCriterion(uot_weight=0.5, grid=8, n_iters=10)
    loss, comp = crit(out, _target(h=32, w=32), model=m)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(("uot_loss", "uot_cost", "mass_gap", "total")) <= set(comp)
    assert m.dec[-1].weight.grad is not None


def test_train_argparse_builds_both_new_loss_branches():
    sys.path.insert(0, "scripts")
    import train

    ap = train.build_argparser()
    h = train._build_criterion(
        ap.parse_args(["--model", "simvp_v2", "--loss", "hurdle"])
    )
    u = train._build_criterion(
        ap.parse_args(["--model", "simvp_v2", "--loss", "uot"])
    )
    assert type(h).__name__ == "HurdleCriterion"
    assert type(u).__name__ == "UOTCriterion"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="needs CUDA to reproduce the CPU-buffer / GPU-input mix")
def test_uot_cost_buffer_follows_prediction_device():
    """scripts/train.py keeps the criterion on CPU while pred is on cuda:0."""
    div = UnbalancedSinkhornDivergence(grid=8, n_iters=5)
    assert div.cost.device.type == "cpu"
    pred = torch.rand(1, 2, 1, 32, 32, device="cuda")
    target = torch.rand(1, 2, 1, 32, 32, device="cuda")
    value, parts = div(pred, target)
    assert value.device.type == "cuda"
    assert torch.isfinite(value)
    assert torch.isfinite(torch.tensor(parts["mass_gap"]))


# ---------------------------------------------------------------------------
# E42 / MASTER 4.44 -- the UOT scale fix (F1 boundedness, F2 ratio invariance)
#
# E41's first batch found that the A2 arm's training loss exploded (single
# step up to 3.4e8, 23 % of the steps above 100) because the UOT term
# normalised by EACH SAMPLE's own truth total: a nearly dry window made
# the denominator collapse and the normalised masses reached 5e5.
#
# The fix scales by the MEDIAN truth total of the batch instead.  These
# tests are the zero-compute gate F1/F2 of E42: the term must stay
# bounded when the absolute mass collapses, and it must still express the
# under-reporting ratio when the absolute mass changes.
# ---------------------------------------------------------------------------


def test_uot_value_stays_bounded_when_truth_mass_collapses():
    """F1: an absolute mass collapse must not blow the term up.

    Before the fix the value went from 0.086 at a truth total of 1.0 to
    5029 at 1e-5.  After it, the value must stay bounded across the
    whole range.
    """
    div = UnbalancedSinkhornDivergence(grid=8, eps=0.05, tau=1.0, n_iters=10)
    torch.manual_seed(7)
    values = []
    for target_total in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        pred = torch.rand(1, 2, 1, 32, 32) * 0.5
        target = torch.rand(1, 2, 1, 32, 32) * target_total
        value, _ = div(pred, target)
        assert torch.isfinite(value)
        values.append(float(value))
    assert max(values) <= 100.0, f"UOT value diverged: {values}"
    # The spread must be bounded too: a term that swings by orders of
    # magnitude between samples is measuring the driest sample, not the
    # mechanism.
    assert max(values) / max(min(values), 1e-12) <= 100.0, values


def test_uot_still_expresses_the_under_reporting_ratio():
    """F2: the ratio must survive the fix, else the mechanism is gone.

    A forecast at a constant fraction of the truth must keep producing a
    similar value whatever the absolute amount of rain there is.
    """
    div = UnbalancedSinkhornDivergence(grid=8, eps=0.05, tau=1.0, n_iters=10)
    torch.manual_seed(11)
    values = []
    for target_total in (1.0, 1e-2, 1e-4):
        target = torch.rand(1, 2, 1, 32, 32) * target_total
        pred = target * 0.2
        value, parts = div(pred, target)
        assert torch.isfinite(value)
        values.append(float(value))
    assert max(values) / max(min(values), 1e-12) <= 2.0, values


def test_uot_mass_gap_still_separates_full_from_under_reporting():
    """The mass_gap term must still tell a full forecast from a thin one."""
    div = UnbalancedSinkhornDivergence(grid=8, eps=0.05, tau=1.0, n_iters=10)
    torch.manual_seed(13)
    t = torch.rand(1, 2, 1, 32, 32) * 0.5
    _, full = div(t.clone(), t)
    _, thin = div(t * 0.3, t)
    assert thin["mass_gap"] > full["mass_gap"]


def test_uot_value_is_bounded_on_a_realistic_dry_batch():
    """F3: a batch that mixes one wet window with dry ones must not blow up.

    This is the shape the training batch actually has: the target is 91 %
    zeros, so most windows carry almost no mass.
    """
    div = UnbalancedSinkhornDivergence(grid=16, eps=0.05, tau=1.0, n_iters=30)
    torch.manual_seed(17)
    target = torch.zeros(4, 10, 1, 64, 64)
    target[0, 0, 0, 20:40, 20:40] = 6.0      # one wet window
    pred = torch.rand(4, 10, 1, 64, 64) * 0.2  # a typical under-forecast
    value, parts = div(pred, target)
    assert torch.isfinite(value)
    assert float(value) <= 100.0, (float(value), parts)
    assert torch.isfinite(torch.tensor(parts["mass_gap"]))
