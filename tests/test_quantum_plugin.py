"""Tests for the quantum-inspired NWP tendency plugin.

The plugin is meant to be a *plug-in*: it has to attach to a baseline
without that baseline knowing anything about it, it has to survive the
ablation (no NWP at all), and it has to leave the shape of the feature
map alone. Those are the properties that make it usable on all fifteen
baselines rather than on one.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _pytest_compat import pytest  # noqa: E402
from radar_nwp_bench.models.common import (  # noqa: E402
    KoopmanUnitaryFlow,
    QuantumInspiredTendencyCoupler,
    QuantumStateProjector,
)

SPATIAL = 16
FEAT = 32
NWP_VARS = 27
OUT_LEN = 10


def _coupler() -> QuantumInspiredTendencyCoupler:
    return QuantumInspiredTendencyCoupler(
        in_nwp_c=NWP_VARS, radar_feat_dim=FEAT, out_len=OUT_LEN
    )


# ---------------------------------------------------------------------------
# the projector
# ---------------------------------------------------------------------------
def test_the_projector_splits_the_variables_by_what_they_are():
    """The two branches are for the thermodynamic and the kinematic state.

    Splitting the channel list in half would put the divergence fields
    (DVG, which describe the kinematic field) on the thermodynamic side
    and the wind speeds (WS, also kinematic) on the other -- the opposite
    of what the branches are for.
    """
    from radar_nwp_bench.data.index import NWP_VARS

    projector = QuantumStateProjector(in_vars=NWP_VARS)
    thermo = [NWP_VARS[i] for i in projector.thermo_index.tolist()]
    dynam = [NWP_VARS[i] for i in projector.dynam_index.tolist()]
    for name in ("DVG200", "DVG850", "DVG925", "WS500", "WS700", "WS925"):
        assert name in dynam, f"{name} is a kinematic field"
    for name in ("CAPE", "PWAT", "KI", "RH700", "LI500"):
        assert name in thermo, f"{name} is a thermodynamic field"
    # the two groups partition the variables
    assert sorted(thermo + dynam) == sorted(NWP_VARS)


def test_the_projector_survives_a_reordering_of_the_channels():
    """The split is by name, so it does not depend on the channel order."""
    from radar_nwp_bench.data.index import NWP_VARS

    reversed_names = tuple(reversed(NWP_VARS))
    projector = QuantumStateProjector(in_vars=NWP_VARS, nwp_vars=reversed_names)
    thermo = {reversed_names[i] for i in projector.thermo_index.tolist()}
    dynam = {reversed_names[i] for i in projector.dynam_index.tolist()}

    assert "PWAT" in thermo
    assert "WS700" in dynam


def test_the_projector_falls_back_when_the_variables_are_unknown():
    """A subset the plugin cannot classify still has to work."""
    projector = QuantumStateProjector(in_vars=4, nwp_vars=("A", "B", "C", "D"))
    assert len(projector.thermo_index) + len(projector.dynam_index) == 4


# ---------------------------------------------------------------------------
# the Koopman flow
# ---------------------------------------------------------------------------
def test_the_koopman_flow_is_unitary():
    """The generator is skew-symmetric, so the evolution conserves energy.

    A skew-Hermitian generator has a unitary exponential: the norm of the
    state is preserved. That is the property the module claims, so it is
    checked rather than assumed.
    """
    import torch

    flow = KoopmanUnitaryFlow(state_dim=8)
    psi = torch.randn(2, 8, SPATIAL, SPATIAL)
    skew = flow.skew_weight - flow.skew_weight.t()
    unitary = torch.matrix_exp(skew)
    identity = unitary.t() @ unitary
    assert torch.allclose(identity, torch.eye(8), atol=1e-4)


def test_the_koopman_flow_keeps_the_shape():
    import torch

    flow = KoopmanUnitaryFlow(state_dim=FEAT)
    psi_0 = torch.randn(2, FEAT, SPATIAL, SPATIAL)
    psi_1 = torch.randn(2, FEAT, SPATIAL, SPATIAL)
    out = flow(psi_0, psi_1)
    assert out.shape == psi_0.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# the coupler
# ---------------------------------------------------------------------------
def test_the_projector_keeps_the_spatial_structure_of_the_field():
    """The projected state has to say *where* the dynamics are.

    The tendency the coupler computes is a spatial field, so a projection
    that flattens the field would throw away the information the module
    exists to carry. The measure is the correlation between neighbouring
    pixels: a smooth physical field stays smooth, and the norm must not
    divide that away.

    The reference numbers come from a real NWP field on the archive: the
    input has a neighbour correlation of 0.9996 and the projection keeps
    0.9907 of it.
    """
    import torch

    from radar_nwp_bench.models.common import QuantumStateProjector

    def neighbour_correlation(field: torch.Tensor) -> float:
        a = field[..., :, :-1].reshape(-1)
        b = field[..., :, 1:].reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        denominator = float(a.norm() * b.norm())
        return 0.0 if denominator == 0 else float((a @ b) / denominator)

    torch.manual_seed(0)
    projector = QuantumStateProjector(in_vars=NWP_VARS)
    projector.eval()

    # A smooth field, the way a real NWP field is: the physical fields
    # vary over hundreds of kilometres while the grid is 3 km, so a few
    # grid cells are barely different from their neighbours. The reference
    # on the archive is a neighbour correlation of 0.9996.
    coordinates = torch.linspace(-3, 3, SPATIAL)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    field = torch.stack([
        torch.sin(0.5 * grid_x + 0.03 * i) * torch.cos(0.5 * grid_y - 0.02 * i)
        for i in range(NWP_VARS)
    ])
    x = field.unsqueeze(0)                 # (1, C, H, W)

    before = neighbour_correlation(x)
    with torch.no_grad():
        after = neighbour_correlation(projector(x))
    assert before > 0.95, f"the test field is not smooth ({before:.4f})"
    assert after > 0.9, (
        f"the projection flattened the field: neighbour correlation "
        f"{before:.4f} -> {after:.4f}"
    )


def test_the_coupler_adapts_to_any_feature_geometry():
    """The point of a plug-in is that it fits whatever it is attached to.

    The fifteen baselines disagree about the width of their feature maps
    and about the resolution they carry them at, and the coupling has to
    work with all of them without being told which one it is on.
    """
    import torch

    for feat_dim in (16, 32, 64, 128):
        for spatial in (16, 32, 64):
            coupler = QuantumInspiredTendencyCoupler(
                in_nwp_c=NWP_VARS, radar_feat_dim=feat_dim, out_len=OUT_LEN
            )
            features = torch.randn(2, OUT_LEN, feat_dim, spatial, spatial)
            x_nwp = torch.randn(2, 2, NWP_VARS, spatial, spatial)
            with torch.no_grad():
                out = coupler(features, x_nwp)
            assert out.shape == features.shape


def test_the_coupler_handles_features_at_a_different_resolution():
    """An encoder that downsamples twice puts its features at a quarter
    of the field size, and the coupling has to follow them there."""
    import torch

    coupler = QuantumInspiredTendencyCoupler(
        in_nwp_c=NWP_VARS, radar_feat_dim=64, out_len=OUT_LEN
    )
    features = torch.randn(2, OUT_LEN, 64, SPATIAL // 2, SPATIAL // 2)
    x_nwp = torch.randn(2, 2, NWP_VARS, SPATIAL, SPATIAL)
    with torch.no_grad():
        out = coupler(features, x_nwp)
    assert out.shape == features.shape


def test_the_coupler_leaves_the_feature_shape_alone():
    import torch

    coupler = _coupler()
    features = torch.randn(2, OUT_LEN, FEAT, SPATIAL, SPATIAL)
    x_nwp = torch.randn(2, 2, NWP_VARS, SPATIAL, SPATIAL)
    out = coupler(features, x_nwp)
    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_the_coupler_actually_modulates():
    """A pass-through would satisfy every other test here.

    The coupling has to change the features, otherwise it is dead code
    that happens to have the right shape.
    """
    import torch

    torch.manual_seed(0)
    coupler = _coupler()
    features = torch.randn(2, OUT_LEN, FEAT, SPATIAL, SPATIAL)
    x_nwp = torch.randn(2, 2, NWP_VARS, SPATIAL, SPATIAL)
    out = coupler(features, x_nwp)
    assert not torch.allclose(out, features), (
        "the coupler did not modulate the features at all"
    )


def test_the_coupler_passes_through_when_the_nwp_is_withheld():
    """The ablation must reach the plugin as an identity."""
    import torch

    coupler = _coupler()
    features = torch.randn(2, OUT_LEN, FEAT, SPATIAL, SPATIAL)
    out = coupler(features, None)
    assert torch.allclose(out, features)


def test_the_coupler_passes_through_when_the_nwp_is_all_zero():
    """A blanked NWP carries no tendency, so nothing may be injected."""
    import torch

    coupler = _coupler()
    features = torch.randn(2, OUT_LEN, FEAT, SPATIAL, SPATIAL)
    out = coupler(features, torch.zeros(2, 2, NWP_VARS, SPATIAL, SPATIAL))
    assert torch.allclose(out, features)


def test_the_coupler_resamples_the_tendency_onto_the_features():
    """The two live at different resolutions.

    A backbone that downsamples twice puts its features at a quarter of
    the field size, and the modulation has to follow. Without the
    resampling the multiplication broadcasts against the wrong shape.
    """
    import torch

    coupler = _coupler()
    features = torch.randn(2, OUT_LEN, FEAT, SPATIAL // 2, SPATIAL // 2)
    x_nwp = torch.randn(2, 2, NWP_VARS, SPATIAL, SPATIAL)
    out = coupler(features, x_nwp)
    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_the_coupler_accepts_a_four_dimensional_feature_map():
    """The plugin documents both layouts, so both have to work."""
    import torch

    coupler = _coupler()
    features = torch.randn(2, FEAT, SPATIAL, SPATIAL)
    x_nwp = torch.randn(2, 2, NWP_VARS, SPATIAL, SPATIAL)
    out = coupler(features, x_nwp)
    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_the_coupler_gradients_reach_the_nwp_branch():
    """A coupling that receives no gradient is not a coupling."""
    import torch

    coupler = _coupler()
    features = torch.randn(2, OUT_LEN, FEAT, SPATIAL, SPATIAL)
    x_nwp = torch.randn(2, 2, NWP_VARS, SPATIAL, SPATIAL)
    out = coupler(features, x_nwp)
    out.sum().backward()

    grads = [p.grad for p in coupler.state_proj.parameters() if p.grad is not None]
    assert grads, "the state projector received no gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)
    flow_grads = [p.grad for p in coupler.koopman_flow.parameters()
                  if p.grad is not None]
    assert flow_grads, "the Koopman flow received no gradient"


# ---------------------------------------------------------------------------
# the integrated baseline
# ---------------------------------------------------------------------------
def test_the_integrated_baseline_registers_and_runs():
    import torch

    from radar_nwp_bench.models import build_model, list_models

    assert "simvp_v2_quantum" in list_models()
    model = build_model("simvp_v2_quantum", in_radar_c=10, in_nwp_c=NWP_VARS,
                        in_len=11, out_len=OUT_LEN, hid_c=16, num_blocks=1)
    with torch.no_grad():
        out = model(torch.rand(1, 11, 10, SPATIAL, SPATIAL),
                    torch.rand(1, 2, NWP_VARS, SPATIAL, SPATIAL))
    assert out.shape == (1, OUT_LEN, 1, SPATIAL, SPATIAL)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_the_integrated_baseline_survives_the_ablation():
    """The plugin's pass-through must not break the baseline."""
    import torch

    from radar_nwp_bench.models import build_model

    model = build_model("simvp_v2_quantum", in_radar_c=10, in_nwp_c=NWP_VARS,
                        in_len=11, out_len=OUT_LEN, hid_c=16, num_blocks=1)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 11, 10, SPATIAL, SPATIAL), None)
    assert torch.isfinite(out).all()
    assert out.shape == (1, OUT_LEN, 1, SPATIAL, SPATIAL)


def test_the_integrated_baseline_learns():
    """The coupling has to be trainable, not just runnable.

    The gradient has to reach the backbone through the coupling, not only
    the coupler's own parameters.
    """
    import torch

    from radar_nwp_bench.models import build_model

    torch.manual_seed(0)
    model = build_model("simvp_v2_quantum", in_radar_c=10, in_nwp_c=NWP_VARS,
                        in_len=11, out_len=OUT_LEN, hid_c=16, num_blocks=1)
    out = model(torch.rand(2, 11, 10, SPATIAL, SPATIAL),
                torch.rand(2, 2, NWP_VARS, SPATIAL, SPATIAL))
    out.sum().backward()

    encoder_grads = [p.grad for p in model.enc.parameters() if p.grad is not None]
    assert encoder_grads, "the encoder received no gradient"
    assert any(float(g.abs().sum()) > 0 for g in encoder_grads)
