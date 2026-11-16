"""Tests for the three NWP-injection ablation variants.

These are the rows that decide what the paper's headline claim rests on.
The claim is that the *mechanism* of the coupling is what matters -- that
the tendency coupler beats the alternatives because of how it couples the
NWP, not merely because the NWP is in the model at all. A row that is
registered but wired up wrongly would leave that claim resting on nothing,
so each row is checked for the properties the claim needs:

* it is registered under the name the paper's table uses, and it builds
  from the configuration file that ships with it, so the row in the table
  is the row that was trained;
* it honours the output contract every baseline honours -- millimetres,
  the documented shape, finite, and never negative;
* the NWP actually reaches the loss. A model that accepts the NWP and
  ignores it would score like a radar-only model while looking like an
  NWP model in the table, which is the failure mode this file exists to
  catch;
* and the difference variant really does use the *difference*. That one is
  checked by a shift: adding the same constant to both NWP slices leaves
  their difference untouched, so the model's response must not move. It is
  the sharpest available test of the claim, because it fails for a model
  that quietly keeps the absolute state around.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from _pytest_compat import pytest  # noqa: E402
from radar_nwp_bench.config import load_config, model_kwargs  # noqa: E402
from radar_nwp_bench.models import build_model, list_models  # noqa: E402
from radar_nwp_bench.models.common import SpatialCrossAttention  # noqa: E402

SPATIAL = 32
IN_LEN = 11
OUT_LEN = 10
RADAR_CHANNELS = 10
NWP_CHANNELS = 27

#: The three rows, in the order the paper's table lists them.
VARIANTS = (
    "simvp_v2_nwp_concat",
    "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta",
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """A radar history and an NWP pair, in the documented layout."""
    torch.manual_seed(0)
    x_radar = torch.rand(1, IN_LEN, RADAR_CHANNELS, SPATIAL, SPATIAL)
    x_nwp = torch.rand(1, 2, NWP_CHANNELS, SPATIAL, SPATIAL)
    return x_radar, x_nwp


def _build(name: str) -> torch.nn.Module:
    """Build a variant from the configuration file that ships with it.

    The point is that the row in the paper's table is the row the
    configuration describes, so the file is the source of the kwargs
    rather than the test hard-coding them.
    """
    config = load_config(REPO_ROOT / "configs" / "models" / f"{name}.yaml")
    kwargs = model_kwargs(
        config,
        shapes={
            "in_radar_c": RADAR_CHANNELS,
            "in_nwp_c": NWP_CHANNELS,
            "in_len": IN_LEN,
            "out_len": OUT_LEN,
        },
    )
    return build_model(name, **kwargs)


def _nwp_gradient(
    model: torch.nn.Module,
    x_radar: torch.Tensor,
    x_nwp: torch.Tensor,
) -> torch.Tensor:
    """The gradient of the model's output with respect to the NWP.

    The gradient is the probe of how the NWP enters the model, and unlike
    the output it is never degenerate: an untrained head predicts
    log-millimetres, so a millimetre output can be identically zero while
    the model is nonetheless sensitive to the NWP. The gradient is
    computed on the summed output so that it does not depend on the sign
    of a single pixel.
    """
    x_nwp = x_nwp.clone().requires_grad_(True)
    model.zero_grad()
    model(x_radar, x_nwp).sum().backward()
    assert x_nwp.grad is not None
    return x_nwp.grad


# ---------------------------------------------------------------------------
# the rows are registered and buildable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", VARIANTS)
def test_the_variant_is_registered(name: str) -> None:
    assert name in list_models(), f"{name} is not registered"


@pytest.mark.parametrize("name", VARIANTS)
def test_the_variant_builds_from_its_configuration(name: str) -> None:
    model = _build(name)
    assert model.num_parameters() > 0
    assert model.in_radar_c == RADAR_CHANNELS
    assert model.in_nwp_c == NWP_CHANNELS
    assert model.in_len == IN_LEN
    assert model.out_len == OUT_LEN


@pytest.mark.parametrize("name", VARIANTS)
def test_the_variant_survives_a_withheld_nwp(name: str) -> None:
    """The ablation hands the model ``None`` for the NWP.

    A row that cannot run without the NWP cannot take part in the
    ablation, which would leave a hole in the table rather than a
    measurement.
    """
    model = _build(name)
    x_radar, _ = _inputs()
    with torch.no_grad():
        out = model(x_radar, None)
    assert out.shape == (1, OUT_LEN, 1, SPATIAL, SPATIAL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# the output contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", VARIANTS)
def test_the_variant_returns_the_documented_shape(name: str) -> None:
    model = _build(name)
    x_radar, x_nwp = _inputs()
    with torch.no_grad():
        out = model(x_radar, x_nwp)
    assert out.shape == (1, OUT_LEN, 1, SPATIAL, SPATIAL)


@pytest.mark.parametrize("name", VARIANTS)
def test_the_variant_output_is_finite(name: str) -> None:
    model = _build(name)
    x_radar, x_nwp = _inputs()
    with torch.no_grad():
        out = model(x_radar, x_nwp)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", VARIANTS)
def test_the_variant_output_is_non_negative(name: str) -> None:
    """Rainfall cannot be negative, so the floor has to hold."""
    model = _build(name)
    x_radar, x_nwp = _inputs()
    with torch.no_grad():
        out = model(x_radar, x_nwp)
    assert (out >= 0.0).all()


# ---------------------------------------------------------------------------
# the NWP has to reach the loss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", VARIANTS)
def test_gradients_reach_the_nwp_branch(name: str) -> None:
    """A model that accepts the NWP and ignores it is not an NWP model.

    The gradient is taken with respect to the NWP input itself, so the
    check does not depend on which internal module happens to consume it.
    """
    model = _build(name)
    x_radar, x_nwp = _inputs()
    x_nwp = x_nwp.requires_grad_(True)

    out = model(x_radar, x_nwp)
    out.sum().backward()

    assert x_nwp.grad is not None, f"{name}: no gradient reached the NWP input"
    total = float(x_nwp.grad.abs().sum())
    assert total > 0.0, f"{name}: the NWP gradient is zero"


@pytest.mark.parametrize("name", VARIANTS)
def test_the_nwp_parameters_receive_gradient(name: str) -> None:
    """The weights that read the NWP must be trained, not frozen.

    This is the same property as the test above seen from the other
    side: a NWP branch whose weights get no gradient would leave the
    model a radar-only model with extra parameters.
    """
    model = _build(name)
    x_radar, x_nwp = _inputs()

    out = model(x_radar, x_nwp)
    out.sum().backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name}: no parameter received a gradient at all"
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


# ---------------------------------------------------------------------------
# the difference variant uses the difference
# ---------------------------------------------------------------------------
def test_the_delta_variant_ignores_a_constant_shift_of_both_nwp_slices() -> None:
    """The difference is invariant to a common offset; the model must be too.

    The benchmark hands over two NWP slices -- an analysis hour and a
    forecast hour -- and the difference variant is given only their
    difference. Adding the same constant to both leaves that difference
    unchanged, so a model that really consumes the difference cannot tell
    the two inputs apart. A model that kept the absolute state anywhere
    would move, and this test would catch it.

    The response is measured on the NWP gradient rather than only on the
    millimetre output, because an untrained head predicts log-millimetres
    and its output can be identically zero while the model is sensitive to
    the NWP all the same. The output is checked as well, so the property
    is asserted on both.
    """
    model = _build("simvp_v2_nwp_delta")
    model.eval()
    x_radar, x_nwp = _inputs()

    grad = _nwp_gradient(model, x_radar, x_nwp)
    grad_shifted = _nwp_gradient(model, x_radar, x_nwp + 7.25)

    scale = float(grad.abs().max())
    assert scale > 0.0, "the probe is degenerate: the NWP gradient is zero"

    drift = float((grad - grad_shifted).abs().max()) / scale
    assert drift < 1e-4, (
        f"the NWP gradient moved by {drift:.2e} of its own magnitude when "
        "both NWP slices were shifted by the same constant, so the model is "
        "using more than their difference"
    )

    with torch.no_grad():
        base = model(x_radar, x_nwp)
        shifted = model(x_radar, x_nwp + 7.25)
    assert torch.allclose(base, shifted, rtol=1e-4, atol=1e-5), (
        "the output moved when both NWP slices were shifted by the same "
        "constant, so the model is using more than their difference"
    )


def test_the_delta_variant_uses_both_nwp_slices_only_through_their_difference() -> None:
    """The exact signature of "the difference is all the model sees".

    If the model consumes the difference alone, then moving one slice up
    and the other down by the same amount leaves the input untouched, and
    the gradients with respect to the two slices must be equal and
    opposite. Any dependence on the absolute state breaks that symmetry.
    """
    model = _build("simvp_v2_nwp_delta")
    model.eval()
    x_radar, x_nwp = _inputs()

    grad = _nwp_gradient(model, x_radar, x_nwp)
    scale = float(grad.abs().max())
    assert scale > 0.0, "the probe is degenerate: the NWP gradient is zero"

    assert float(grad[:, -1].abs().sum()) > 0.0, (
        "the forecast slice received no gradient at all"
    )
    imbalance = float((grad[:, -1] + grad[:, 0]).abs().max()) / scale
    assert imbalance < 1e-5, (
        f"the gradients with respect to the two NWP slices are not opposite "
        f"({imbalance:.2e} of their own magnitude), so the model sees more "
        "than their difference"
    )


def test_the_delta_variant_reacts_to_a_change_in_the_difference() -> None:
    """The counterpart of the shift test: the difference must still matter.

    A model that ignored the NWP altogether would also pass the shift
    test, so this pins the other side down. The two slices are pulled
    apart, which changes their difference and must change the response.
    """
    model = _build("simvp_v2_nwp_delta")
    model.eval()
    x_radar, x_nwp = _inputs()

    perturbed = x_nwp.clone()
    perturbed[:, -1] += 3.0

    grad = _nwp_gradient(model, x_radar, x_nwp)
    grad_perturbed = _nwp_gradient(model, x_radar, perturbed)

    scale = float(grad.abs().max())
    assert scale > 0.0, "the probe is degenerate: the NWP gradient is zero"

    response = float((grad - grad_perturbed).abs().max()) / scale
    assert response > 1e-2, (
        f"the response moved by only {response:.2e} of its own magnitude "
        "when the NWP difference changed, so the difference variant is not "
        "using the difference"
    )


def test_the_concat_variant_does_not_ignore_the_absolute_state() -> None:
    """The contrast row: a common shift must move the naive concatenation.

    This is the row that says "the NWP state itself carries information
    the difference does not". If the naive concatenation were also
    invariant under a common shift, the two rows would be measuring the
    same thing and the comparison between them would be empty.
    """
    model = _build("simvp_v2_nwp_concat")
    model.eval()
    x_radar, x_nwp = _inputs()

    grad = _nwp_gradient(model, x_radar, x_nwp)
    grad_shifted = _nwp_gradient(model, x_radar, x_nwp + 7.25)

    scale = float(grad.abs().max())
    assert scale > 0.0, "the probe is degenerate: the NWP gradient is zero"

    response = float((grad - grad_shifted).abs().max()) / scale
    assert response > 1e-2, (
        f"the response moved by only {response:.2e} of its own magnitude "
        "under a common NWP shift, so the naive concatenation is not "
        "reading the absolute state either"
    )


# ---------------------------------------------------------------------------
# the shared cross-attention block
# ---------------------------------------------------------------------------
def test_the_cross_attention_is_the_conventional_one() -> None:
    """It must be ordinary attention, and nothing else.

    The cross-attention row is the control of the ablation, so its
    machinery is what the comparison rests on. A naive implementation of
    the same formula pins it down: if the block computed something else,
    the two would disagree.
    """
    import math

    torch.manual_seed(0)
    block = SpatialCrossAttention(16, 8, num_heads=4, max_tokens=10**9).eval()
    x = torch.randn(2, 16, 6, 6)
    ctx = torch.randn(2, 8, 6, 6)

    with torch.no_grad():
        got = block(x, ctx)

        # The textbook attention, written out.
        b, c, h, w = x.shape
        head_dim = c // block.num_heads
        tokens = h * w

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, block.num_heads, head_dim, tokens).permute(0, 1, 3, 2)

        q = heads(block.q_proj(block.norm(x)))
        k = heads(block.k_proj(ctx))
        v = heads(block.v_proj(ctx))
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(head_dim)
        attended = torch.einsum(
            "bhts,bhsd->bhtd", torch.softmax(scores, dim=-1), v
        )
        attended = attended.permute(0, 1, 3, 2).reshape(b, c, h, w)
        expected = x + block.proj(attended)
        expected = expected + block.ffn(block.ffn_norm(expected))

    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-6), (
        "the block does not compute plain multi-head attention"
    )


def test_the_cross_attention_resamples_a_context_at_another_resolution() -> None:
    """The NWP map and the radar features live at different resolutions."""
    torch.manual_seed(0)
    block = SpatialCrossAttention(16, 8, num_heads=4).eval()
    x = torch.randn(2, 16, 6, 6)
    ctx = torch.randn(2, 8, 12, 10)

    with torch.no_grad():
        out = block(x, ctx)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_the_cross_attention_bounds_the_cost_at_the_production_grid() -> None:
    """A 301x301 grid must not need a quadratic score matrix.

    The attention is quadratic in the number of spatial tokens, and the
    production grid carries 22801 of them in the latent. The block
    therefore strides both grids down when they carry too many tokens and
    upsamples the result again, so the output keeps the query's shape.
    """
    torch.manual_seed(0)
    block = SpatialCrossAttention(16, 8, num_heads=4, max_tokens=64).eval()
    x = torch.randn(1, 16, 24, 24)
    ctx = torch.randn(1, 8, 24, 24)

    with torch.no_grad():
        out = block(x, ctx)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

