"""E39 / MASTER 4.41 -- CPU small-tensor unit tests for the IBCFM adapter.

These are **unit tests only**: tiny random tensors, CPU, no dataset, no
training run, no evaluation.  They verify the interface contract, the
backprop path, the A1/A2 single-variable mechanism switch, and the
threshold-decision field.  They deliberately produce no CSI/BIAS number.

Per MASTER 11.6, all dataset-level training and evaluation happen on the
remote CUDA host; this file is the local-code half of that agreement.
"""
from __future__ import annotations

import torch

from radar_nwp_bench.models import build_model


def _small(**overrides):
    kwargs = dict(
        in_radar_c=10,
        in_nwp_c=27,
        in_len=11,
        out_len=10,
        hid_c=8,
        unet_base=8,
        n_velocity_blocks=2,
        n_members=3,
        sample_steps=2,
    )
    kwargs.update(overrides)
    return build_model("ibcfm", **kwargs)


def _inputs(b=2, h=16, w=16):
    torch.manual_seed(0)
    return (
        torch.randn(b, 11, 10, h, w),
        torch.randn(b, 2, 27, h, w),
    )


def test_registry_and_shape_contract():
    m = _small()
    x_radar, x_nwp = _inputs()
    m.train()
    out = m(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 16, 16)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_no_nwp_ablation_resolves_at_the_seam():
    m = _small()
    x_radar, _ = _inputs()
    m.train()
    out = m(x_radar, None)
    assert out.shape == (2, 10, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_flow_loss_is_scalar_finite_and_backprops():
    m = _small()
    x_radar, x_nwp = _inputs()
    m.train()
    m(x_radar, x_nwp)
    target = torch.rand(2, 10, 1, 16, 16) * 6.0
    loss = m.compute_flow_loss(target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        p.grad for p in m.velocity.parameters()
        if p.grad is not None
    ]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_band_weights_keep_loss_scale():
    m = _small()
    target = torch.cat(
        [
            torch.zeros(1, 1, 1, 8, 8),
            torch.full((1, 1, 1, 8, 8), 0.7),
            torch.full((1, 1, 1, 8, 8), 2.5),
            torch.full((1, 1, 1, 8, 8), 5.0),
        ],
        dim=0,
    )
    w = m._band_weights(target)
    assert torch.isfinite(w).all()
    assert abs(float(w.mean()) - 1.0) < 1e-5
    assert m._band_weights(target) is not None
    m.intensity_bands = False
    assert torch.allclose(m._band_weights(target), torch.ones_like(target))


def test_random_start_is_the_only_difference_between_a1_and_a2():
    """C1: A2 draws genuine members; zero-start A1 collapses to one field."""
    x_radar, x_nwp = _inputs()

    torch.manual_seed(7)
    a2 = _small(random_start=True).eval()
    with torch.no_grad():
        a2(x_radar, x_nwp)
        members_a2 = a2.sample_members(n=3)
    assert members_a2.shape == (2, 3, 10, 1, 16, 16)
    assert not torch.allclose(members_a2[:, 0], members_a2[:, 1])

    torch.manual_seed(7)
    a1 = _small(random_start=False).eval()
    with torch.no_grad():
        a1(x_radar, x_nwp)
        members_a1 = a1.sample_members(n=3)
    assert torch.allclose(members_a1[:, 0], members_a1[:, 1], atol=1e-6)
    assert torch.allclose(members_a1[:, 1], members_a1[:, 2], atol=1e-6)


def test_fixed_seed_reproduces_members():
    x_radar, x_nwp = _inputs()
    m = _small().eval()
    torch.manual_seed(11)
    with torch.no_grad():
        m(x_radar, x_nwp)
        first = m.sample_members(n=2).clone()
    torch.manual_seed(11)
    with torch.no_grad():
        m(x_radar, x_nwp)
        second = m.sample_members(n=2).clone()
    assert torch.allclose(first, second, atol=1e-6)


def test_decision_field_is_still_a_rainfall_field():
    m = _small(decision_mode="decision", p_star=0.3, exceed_mm=2.5)
    x_radar, x_nwp = _inputs()
    m.eval()
    with torch.no_grad():
        out = m(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 16, 16)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    values = torch.unique(out)
    assert set(values.tolist()) <= {0.0, 2.5}


def test_member_and_mean_modes_keep_the_contract():
    x_radar, x_nwp = _inputs()
    for mode in ("member", "mean"):
        m = _small(decision_mode=mode).eval()
        with torch.no_grad():
            out = m(x_radar, x_nwp)
        assert out.shape == (2, 10, 1, 16, 16)
        assert torch.isfinite(out).all()
        assert (out >= 0).all()
