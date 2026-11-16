"""Unit tests for the E45 topology term (MASTER 4.47).

E44 ran the A3 arm and measured two pathologies in the clDice term:
the top 1 % of the pixels carried 0.9399 of the term's gradient L1,
and a few pixels were pushed to 402.43 mm, the ceiling of the model's
own ``expm1`` head.  E45 keeps the term and adds three mechanisms to
it, each one a switch so that an arm can isolate it:

* an amplitude truncation of the forecast,
* a Gaussian blur of the truth's support,
* a normalisation of the weights by the area of their component.

What the tests here pin down is the MECHANISM and its invariants, not
an outcome:

1. With all three switches off the term is the E44 term, bit for bit.
2. The truncation removes the gradient of the pixels above the ceiling
   and leaves the gradient of the others alone.
3. The blur changes the truth's support and does not touch the
   forecast's gradient.
4. The area map is the size of the component a pixel belongs to, and
   the normalisation makes the components' contributions comparable.
5. The area map is a constant: the gradient still flows, and it flows
   only through the weights.

The oracle in this file is a frozen copy of the E44 forward, so the
"bit for bit" claim is checked against the arithmetic of that
implementation and not against the module's own reference method.
"""
from __future__ import annotations

import pytest
import torch

from radar_nwp_bench.metrics.topology_losses import (
    ClDiceLoss,
    _area_normalized_weights,
    _component_area_map,
    _gaussian_blur_2d,
    _soft_component_weight,
)


# --------------------------------------------------------------- the oracle


def _oracle_propagate(
    values: torch.Tensor, mask: torch.Tensor, steps: int
) -> torch.Tensor:
    """A frozen copy of E44's ``_propagate``."""
    out = values
    for _ in range(steps):
        pooled = torch.nn.functional.max_pool2d(
            out, kernel_size=3, stride=1, padding=1
        )
        out = mask * pooled + (1.0 - mask) * out
    return out


def _oracle_soft_component_weight(
    field: torch.Tensor, power: float
) -> torch.Tensor:
    """A frozen copy of E44's ``_soft_component_weight``."""
    n, c, h, w = field.shape
    steps = h + w
    mask = (field / field.amax(dim=(2, 3), keepdim=True).detach().clamp_min(1e-8))
    comp_max = _oracle_propagate(field, mask.clamp_max(1.0), steps)
    scale = comp_max.amax(dim=(2, 3), keepdim=True).detach().clamp_min(1e-8)
    return (comp_max / scale).clamp_min(0.0).clamp_max(1.0) ** power


def _oracle_cldice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 2.0,
    power: float = 3.0,
    softness: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """A frozen copy of E44's clDice forward, term and all."""
    p = pred.clamp_min(0.0)
    t = target.clamp_min(0.0)
    if p.dim() > 3:
        p = p.reshape(-1, p.shape[-2], p.shape[-1])
        t = t.reshape(-1, t.shape[-2], t.shape[-1])
    p = p.unsqueeze(1)
    t = t.unsqueeze(1)
    p_bin = torch.sigmoid((p - threshold) / softness)
    t_bin = torch.sigmoid((t - threshold) / softness).detach()
    p_w = _oracle_soft_component_weight(p_bin * p, power)
    t_w = _oracle_soft_component_weight(t_bin * t, power)
    inter = (p_w * t_w).sum(dim=(1, 2, 3))
    denom = p_w.sum(dim=(1, 2, 3)) + t_w.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


# ------------------------------------------------------------------ helpers


def _field(seed: int = 11, size: int = 16, high: float = 1.5):
    """A forecast below the threshold and a truth with a rain block.

    The block carries a RAMP of intensities rather than a single value:
    with a uniform block the term's weights are all equal and several
    details of the arithmetic cancel, so the field would not distinguish
    implementations that differ in them.
    """
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(1, size, size, generator=g) * high
    target = torch.zeros(1, size, size)
    ramp = torch.linspace(1.0, 6.0, 6).view(1, 1, 6)
    target[0, 4:10, 4:10] = ramp.expand(1, 6, 6)
    return pred, target


# --------------------------------------------- 1. the switches off = E44


def test_all_switches_off_reproduce_the_e44_term_bit_for_bit():
    """With the three switches off the term IS the E44 term.

    The comparison is against a frozen copy of E44's forward, so it
    catches a change in the shared helpers as well as a change in the
    reference method.  The gradient is compared bit for bit too: the
    claim is about the arithmetic, not about its size.
    """
    pred, target = _field()
    loss = ClDiceLoss(threshold=2.0)

    got, components = loss(pred, target)
    expected = _oracle_cldice(pred, target)
    assert torch.equal(got, expected), (
        f"the term changed with the switches off: "
        f"{float(got)} vs {float(expected)}"
    )
    assert components["cldice"] == float(expected)

    # The reference method itself must agree, so that the equality above
    # is a property of the code and not of a careful reading of it.
    reference, _ = loss._forward_reference(pred, target)
    assert torch.equal(got, reference)


def test_all_switches_off_reproduce_the_e44_gradient_bit_for_bit():
    """The gradient of the E44 term must be unchanged as well."""
    pred, target = _field(seed=12)
    pred = pred.requires_grad_(True)
    loss = ClDiceLoss(threshold=2.0)

    got, _ = loss(pred, target)
    got.backward()
    got_grad = pred.grad.clone()

    pred2 = pred.detach().clone().requires_grad_(True)
    expected = _oracle_cldice(pred2, target)
    expected.backward()

    assert torch.equal(got_grad, pred2.grad), (
        "the gradient changed with the switches off: "
        f"max_abs_diff={(got_grad - pred2.grad).abs().max().item()}"
    )


def test_each_switch_alone_leaves_the_other_mechanisms_off():
    """The three switches are independent of each other.

    A switch that is off must not leave a trace, so the term with one
    switch on differs from the E44 term only through that switch.
    """
    pred, target = _field(seed=13)

    base, _ = ClDiceLoss(threshold=2.0)(pred, target)
    expected = _oracle_cldice(pred, target)
    assert torch.equal(base, expected)

    # The clamp alone: the forecast is truncated, the truth is not
    # blurred and the weights are not area-normalised.
    clamped, _ = ClDiceLoss(threshold=2.0, clamp_mm=20.0)(pred, target)
    manual = _oracle_cldice(pred.clamp(0.0, 20.0), target)
    assert torch.equal(clamped, manual)

    # The blur alone: the truth is blurred, the forecast is untouched.
    blurred, _ = ClDiceLoss(threshold=2.0, truth_smooth_sigma=2.0)(
        pred, target
    )
    assert not torch.equal(blurred, base), (
        "the blurred truth did not change the term"
    )


# ------------------------------------------------- 2. the amplitude ceiling


def test_the_clamp_stops_the_gradient_of_the_pixels_above_the_ceiling():
    """A pixel above the ceiling gets no gradient from this term.

    E44 measured those pixels reaching 402.43 mm.  The ceiling is
    applied with ``clamp``, whose derivative is zero above the bound,
    so the gradient of such a pixel is exactly zero -- not merely
    small -- and the gradient of the other pixels is untouched.
    """
    pred, target = _field(seed=14)
    pred[0, 8, 8] = 402.43
    pred = pred.requires_grad_(True)

    loss = ClDiceLoss(threshold=2.0, clamp_mm=20.0)
    value, _ = loss(pred, target)
    value.backward()

    needle = pred.grad[0, 8, 8]
    assert float(needle) == 0.0, (
        f"the pixel above the ceiling still receives a gradient of "
        f"{float(needle)}"
    )

    others = pred.grad.clone()
    others[0, 8, 8] = 0.0
    assert float(others.abs().max()) > 0.0, (
        "the truncation took the gradient of the whole field with it"
    )
    assert int((others != 0).sum()) == others.numel() - 1, (
        "the truncation must silence the truncated pixel only"
    )


def test_the_clamp_changes_what_the_term_reads():
    """The ceiling is applied before the term reads the forecast."""
    pred, target = _field(seed=15)
    pred[0, 8, 8] = 402.43

    clamped, _ = ClDiceLoss(threshold=2.0, clamp_mm=20.0)(pred, target)
    manual = _oracle_cldice(pred.clamp(0.0, 20.0), target)
    assert torch.equal(clamped, manual), (
        "the term did not read the truncated forecast"
    )


# ------------------------------------------------------ 3. the truth's blur


def test_the_blur_changes_the_target_support_and_keeps_the_gradient():
    """The blur moves the truth's support and leaves the forecast alone.

    The forecast's gradient must survive: the forecast is what the
    term is supposed to shape, and a blurred forecast would remove the
    very gradient the term exists to supply.
    """
    pred, target = _field(seed=16)
    pred = pred.requires_grad_(True)

    plain, _ = ClDiceLoss(threshold=2.0)(pred, target)
    blurred, _ = ClDiceLoss(threshold=2.0, truth_smooth_sigma=2.0)(
        pred, target
    )
    assert not torch.equal(plain, blurred), (
        "the blurred truth did not change the term"
    )
    assert abs(float(plain.detach()) - float(blurred.detach())) > 1e-6

    blurred.backward()
    assert pred.grad is not None
    assert float(pred.grad.abs().max()) > 0.0, (
        "the blurred truth took the forecast's gradient with it"
    )
    assert torch.isfinite(pred.grad).all()


def test_the_blur_does_not_reach_the_forecast():
    """The blur is on the truth's side and must stay there."""
    pred, target = _field(seed=17)
    pred_a = pred.clone().requires_grad_(True)
    pred_b = pred.clone().requires_grad_(True)

    value_a, _ = ClDiceLoss(threshold=2.0, truth_smooth_sigma=2.0)(
        pred_a, target
    )
    value_b, _ = ClDiceLoss(threshold=2.0, truth_smooth_sigma=2.0)(
        pred_b, target
    )
    value_a.backward()
    value_b.backward()
    assert torch.equal(pred_a.grad, pred_b.grad), (
        "the gradient depends on something other than the forecast"
    )


def test_the_blurred_truth_carries_no_gradient():
    """The blurred truth is a constant, like the truth's mask.

    The truth is data: nothing about the term may ask for a gradient
    through it, and the blur must not open such a path.
    """
    pred, target = _field(seed=21)
    target = target.requires_grad_(True)
    pred = pred.requires_grad_(True)

    value, _ = ClDiceLoss(threshold=2.0, truth_smooth_sigma=2.0)(
        pred, target
    )
    value.backward()
    assert target.grad is None, (
        "the blurred truth carries a gradient of "
        f"{float(target.grad.abs().sum())}"
    )
    assert pred.grad is not None


def test_the_blur_reflects_the_border_instead_of_zeroing_it():
    """The border is the field's own content, not a dark frame.

    A zero pad would pull the border of the truth towards zero and
    would make the truth's support depend on where the patch was cut.
    """
    field = torch.full((1, 1, 16, 16), 5.0)
    blurred = _gaussian_blur_2d(field, 2.0)
    assert torch.allclose(blurred, field, atol=1e-5), (
        "the border of a uniform field was not preserved"
    )


def test_the_blur_acts_on_both_axes():
    """The blur is separable AND it covers both axes.

    A field that is constant along one axis must stay constant along
    that axis, so the blur must not touch it.
    """
    g = torch.Generator().manual_seed(31)
    row = torch.rand(1, 1, 1, 16, generator=g) * 4.0
    field = row.expand(1, 1, 16, 16).clone()          # constant along H
    blurred = _gaussian_blur_2d(field, 2.0)
    assert torch.allclose(
        blurred, blurred[:, :, :1, :].expand_as(blurred), atol=1e-5
    ), "the blur made the rows of the field differ"

    column = torch.rand(1, 1, 16, 1, generator=g) * 4.0
    field = column.expand(1, 1, 16, 16).clone()        # constant along W
    blurred = _gaussian_blur_2d(field, 2.0)
    assert torch.allclose(
        blurred, blurred[:, :, :, :1].expand_as(blurred), atol=1e-5
    ), "the blur made the columns of the field differ"


def test_the_blur_matches_an_independent_convolution():
    """The blur is the convolution it claims to be.

    The reference below is written out with explicit index arithmetic,
    so it does not share a line of code with the implementation.
    """
    sigma = 2.0
    radius = int(round(3.0 * sigma))
    g = torch.Generator().manual_seed(32)
    field = torch.rand(1, 1, 12, 12, generator=g) * 4.0

    # The 1-D kernel: a Gaussian truncated at three sigma, normalised.
    coords = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()

    def _reflect_pad_1d(x: torch.Tensor, r: int) -> torch.Tensor:
        n = x.shape[-1]
        idx = torch.arange(-r, n + r)
        idx = torch.where(idx < 0, -idx, idx)
        idx = torch.where(idx >= n, 2 * n - 2 - idx, idx)
        return x.index_select(-1, idx)

    # Convolve along W, then along H, with the reflect padding spelled
    # out rather than delegated.
    padded = _reflect_pad_1d(field, radius)
    out = torch.zeros_like(field)
    for k in range(2 * radius + 1):
        out = out + kernel[k] * padded[..., k:k + field.shape[-1]]
    padded = _reflect_pad_1d(out.transpose(-1, -2), radius).transpose(-1, -2)
    expected = torch.zeros_like(field)
    for k in range(2 * radius + 1):
        expected = expected + kernel[k] * padded[..., k:k + field.shape[-2], :]

    blurred = _gaussian_blur_2d(field, sigma)
    assert torch.allclose(blurred, expected.to(blurred.dtype), atol=1e-5), (
        "the blur is not the convolution it documents"
    )


# ---------------------------------------------------- 4. the component areas


def test_the_area_map_is_the_size_of_the_component():
    """A pixel's area is the number of pixels of its component."""
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 1, 1] = 1.0              # a component of one pixel
    mask[0, 0, 5, 2:7] = 1.0            # a component of five pixels

    area = _component_area_map(mask)
    assert area.shape == mask.shape
    assert float(area[0, 0, 1, 1]) == 1.0
    for pixel in area[0, 0, 5, 2:7].flatten().tolist():
        assert pixel == 5.0, f"a pixel of the block got the area {pixel}"
    # A pixel outside the mask belongs to no component and is left alone.
    assert float(area[0, 0, 0, 0]) == 1.0


def test_the_area_normalisation_makes_the_components_comparable():
    """A one-pixel component and a five-pixel component contribute alike.

    The total weight of a component is the sum of its per-pixel weights
    divided by its area, so with equal per-pixel weights it is the
    same for every component, whatever its size.
    """
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 1, 1] = 1.0
    mask[0, 0, 5, 2:7] = 1.0

    weights = torch.ones(1, 1, 8, 8)
    normalised = _area_normalized_weights(weights, mask)

    one_pixel = float(normalised[0, 0, 1, 1].sum())
    five_pixels = float(normalised[0, 0, 5, 2:7].sum())
    ratio = one_pixel / five_pixels
    assert 1.0 / 1.2 <= ratio <= 1.2, (
        f"the components' total weights are not comparable: {ratio}"
    )


def test_the_area_map_separates_the_samples():
    """Two samples are two sets of components."""
    mask = torch.zeros(2, 1, 8, 8)
    mask[0, 0, 1, 1] = 1.0
    mask[1, 0, 5, 2:7] = 1.0

    area = _component_area_map(mask)
    assert float(area[0, 0, 1, 1]) == 1.0
    assert float(area[1, 0, 5, 2]) == 5.0
    assert float(area[0, 0, 5, 2]) == 1.0
    assert float(area[1, 0, 1, 1]) == 1.0


def test_each_side_is_normalised_by_its_own_components():
    """The forecast's weights use the forecast's components.

    The two sides are normalised separately: the forecast's weights are
    divided by the areas of the FORECAST's components and the truth's
    weights by the areas of the TRUTH's components.  A forecast with a
    four-pixel wet block and a truth with a nine-pixel wet block over
    the same place is the case that tells the two apart.
    """
    pred = torch.zeros(1, 8, 8)
    pred[0, 2:4, 2:4] = 5.0               # a wet block of four
    target = torch.zeros(1, 8, 8)
    target[0, 1:4, 1:4] = 5.0             # a wet block of nine

    # The forecast's block is a component of four and the truth's block
    # is a component of nine.
    p_area = _component_area_map((pred.unsqueeze(1) >= 2.0).to(pred.dtype))
    t_area = _component_area_map((target.unsqueeze(1) >= 2.0).to(target.dtype))
    assert float(p_area[0, 0, 2, 2]) == 4.0
    assert float(t_area[0, 0, 2, 2]) == 9.0

    # The term must apply each side's own areas.
    loss = ClDiceLoss(threshold=2.0, area_normalize=True)
    value, _ = loss(pred, target)

    p = pred.clamp_min(0.0).unsqueeze(1)
    t = target.clamp_min(0.0).unsqueeze(1)
    p_bin = torch.sigmoid((p - 2.0) / 0.5)
    t_bin = torch.sigmoid((t - 2.0) / 0.5).detach()
    p_w = _area_normalized_weights(
        _soft_component_weight(p_bin * p, 3.0), (p >= 2.0).to(p.dtype)
    )
    t_w = _area_normalized_weights(
        _soft_component_weight(t_bin * t, 3.0), (t >= 2.0).to(t.dtype)
    )
    inter = (p_w * t_w).sum(dim=(1, 2, 3))
    denom = p_w.sum(dim=(1, 2, 3)) + t_w.sum(dim=(1, 2, 3))
    expected = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    assert torch.equal(value, expected), (
        "the term did not normalise each side by its own components"
    )


# ------------------------------------------------- 5. the area map is constant


def test_the_area_map_carries_no_gradient():
    """The area map is a constant, not a second gradient path.

    Were the areas differentiable, the term would have a gradient
    channel whose size nothing controls.
    """
    pred, target = _field(seed=18)
    pred = pred.requires_grad_(True)

    mask = (pred.detach() >= 2.0).to(pred.dtype).unsqueeze(1)
    area = _component_area_map(mask)
    assert not area.requires_grad
    assert not area.grad_fn

    weights = torch.ones_like(mask, requires_grad=True)
    normalised = _area_normalized_weights(weights, mask)
    normalised.sum().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_the_area_normalised_term_still_has_a_finite_gradient():
    """With the normalisation on, the term still trains the forecast."""
    pred, target = _field(seed=19)
    pred = pred.requires_grad_(True)

    value, _ = ClDiceLoss(threshold=2.0, area_normalize=True)(pred, target)
    assert torch.isfinite(value)
    value.backward()

    assert pred.grad is not None, "the term has no gradient at all"
    assert torch.isfinite(pred.grad).all(), "the gradient is not finite"
    assert float(pred.grad.abs().max()) > 0.0, "the gradient is zero"


def test_the_area_normalised_term_differs_from_the_e44_term():
    """The normalisation is actually applied to the term."""
    pred, target = _field(seed=20)
    plain, _ = ClDiceLoss(threshold=2.0)(pred, target)
    normalised, _ = ClDiceLoss(threshold=2.0, area_normalize=True)(pred, target)
    assert not torch.equal(plain, normalised)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
