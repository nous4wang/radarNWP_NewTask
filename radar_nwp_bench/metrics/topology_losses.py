"""E44 / E45 -- MASTER 4.46 / 4.47 -- the clDice connectivity loss.

A3's mechanism is a single extra term in the training objective, so this
module has to do exactly one thing: measure how well the connected
components of the forecast agree with the connected components of the
truth at the strong-rain threshold.

The soft clDice of Sit et al. (MICCAI 2021) is used.  Its definition is

    clDice = 1 - (2 * |S_p & S_t| + eps) / (|S_p| + |S_t| + eps)

where ``S_p`` and ``S_t`` are the connected components of the forecast
and of the truth.  The intersection of two component sets is not
differentiable, so the usual approximation is used: a component
survives with a probability that is the field value raised to a power
``p``, and the components are then weighted by those probabilities.

The threshold is the strong-rain one (2.0 mm).  The label's
connectivity is what distinguishes it from the radar's -- C0 measured
the label's connected kernel to be 3.37 times the radar's -- and at
0.5 mm almost every pixel is wet, so the connectivity would be satisfied
by anything that reports rain everywhere.

Gradients
---------
The first implementation of this file labelled the components with a
CPU union-find and summed over them with numpy.  That breaks the
autograd graph: the term came out with no gradient at all, so the arm
would have trained exactly like the control and the mechanism would
never have been tested.  A gradient check caught it before any compute
was spent.

The implementation below is therefore fully differentiable and stays
on the device.  The component structure is captured by a soft
max-pooling dilation: a value is propagated to its neighbours while the
field is above the threshold, so the reachable set of a pixel is its
component, and the propagated value is that component's survival
probability.  The number of propagation steps is the diameter of the
patch, so the reach is exact for the sizes used here.

E45's three mechanisms
----------------------
E44 ran the arm and measured two pathologies, both of them properties of
this term rather than of the model:

1. **Gradient concentration.**  On the 2 mm tier the top 1 % of the
   pixels carried **0.9399** of the term's gradient L1.  The term was
   therefore shaped by a handful of pixels, not by the field.
2. **Needles at the ceiling.**  Those pixels were pushed to **402.43 mm**,
   the clamp of the model's ``expm1`` head.  The E44 diagnostics had
   already established that the topology term PUNISHES a needle (cutting
   it LOWERED the term by 0.377 in 24 of 25 cases), so the needle is a
   side effect of where the gradient sits, not of what the term wants.

E45 therefore changes three things, all of them inside this term and all
of them independently switchable so that each one can be ablated on its
own:

* ``clamp_mm`` -- the forecast is truncated to a physical ceiling
  (default 20 mm) BEFORE the term reads it.  A pixel above the ceiling
  gets no gradient from this term at all, which cuts the path that
  leads to the ``expm1`` ceiling.
* ``truth_smooth_sigma`` -- the TRUTH side is read through a separable
  Gaussian blur, so the support the term compares against is a region
  rather than a few isolated pixels.  The prediction side is left
  untouched: it is the forecast that is being shaped, and blurring it
  would remove the very gradient the term is supposed to supply.
* ``area_normalize`` -- the per-pixel weight of a pixel is divided by
  the area of the connected component it belongs to, so that a
  one-pixel component and a hundred-pixel component contribute
  comparably to the term.  The area map is a constant: it multiplies
  the weight, it does not carry a gradient path of its own.

The boundary is unchanged from E44: the model, its backbone, its
decoder and its output contract are untouched, and the load-bearing
``balanced`` objective is used exactly as it was.  Everything here lives
in the loss.

Nothing here trains or evaluates a model.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _propagate(values: torch.Tensor, mask: torch.Tensor, steps: int) -> torch.Tensor:
    """Propagate a value within the connected components of a mask.

    A pixel's value is spread to its four neighbours while the mask is
    on, so after enough steps every pixel of a component carries the
    component's maximum value.

    The mask is used SOFTLY: the propagated value is the weighted
    average of the neighbour's value and the pooled value, with the
    weight being the mask value.  A hard mask would cut the gradient at
    every pixel below the threshold, and since the model under-reports
    the rain, most pixels are below it -- the term would then act on a
    handful of pixels and could not shape the field.

    Args:
        values: ``(N, C, H, W)`` the values to spread.
        mask: ``(N, 1, H, W)`` the soft mask that defines the components.
        steps: the number of propagation steps; the diameter of the
            patch is enough for the reach to be exact.

    Returns:
        The same shape as ``values``.
    """
    out = values
    for _ in range(steps):
        pooled = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
        # A soft blend: where the mask is on, the neighbour's value wins;
        # where it is off, the pixel keeps its own.
        out = mask * pooled + (1.0 - mask) * out
    return out


def _soft_component_weight(field: torch.Tensor, power: float) -> torch.Tensor:
    """A differentiable weight for the connected components of a field.

    The weight of a pixel is the survival probability of its component:
    the component's maximum field value, raised to ``power`` and
    normalised by the maximum over the whole frame so the weights stay
    in ``[0, 1]``.

    The maximum over a component is computed by the max-pooling
    propagation of ``_propagate``, which is differentiable and stays on
    the device.  The normalisation is by a DETACHED constant: dividing
    by a value that carries its own gradient would cancel the gradient
    of the numerator, which is the whole term.

    Args:
        field: ``(N, C, H, W)`` non-negative field.
        power: the exponent applied to the field.

    Returns:
        ``(N, C, H, W)`` weights, one per pixel.
    """
    n, c, h, w = field.shape
    steps = h + w                       # the patch diameter: an exact reach
    # The soft mask: a pixel belongs to a component with a probability
    # that grows with its own value.
    mask = (field / field.amax(dim=(2, 3), keepdim=True).detach().clamp_min(1e-8))
    comp_max = _propagate(field, mask.clamp_max(1.0), steps)
    # The survival probability of the component, as a fraction of the
    # strongest component in the frame.
    scale = comp_max.amax(dim=(2, 3), keepdim=True).detach().clamp_min(1e-8)
    return (comp_max / scale).clamp_min(0.0).clamp_max(1.0) ** power


def _gaussian_kernel_1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    """The 1-D Gaussian kernel of a given half-width, normalised.

    The kernel is truncated at three sigma, which is the usual choice and
    keeps the operation local: the truth's support is widened by a few
    pixels, not by a fraction of the patch.

    Args:
        sigma: the standard deviation in pixels.
        radius: the half-width of the kernel in pixels.
        device: the device the kernel is built on.
        dtype: the dtype of the kernel.

    Returns:
        A ``(2 * radius + 1,)`` tensor summing to one.
    """
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur_2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a field separably along the two spatial axes.

    The blur is the mechanism that turns the truth's support from a set
    of isolated pixels into a region: the E44 diagnostics measured the
    topology term's gradient concentrating on the top 1 % of the pixels
    (0.9399 of the gradient L1), and a hard truth mask is what puts the
    term's attention on exactly those pixels.

    The padding is ``reflect``, so a blurred pixel near the border is a
    reflection of the field's own content and not a zero, which keeps
    the border from becoming a dark frame of its own.

    Args:
        field: ``(N, C, H, W)`` the field to blur.
        sigma: the standard deviation in pixels.

    Returns:
        The same shape as ``field``.
    """
    n, c, h, w = field.shape
    # ``reflect`` padding cannot exceed the extent of the axis, so the
    # kernel is truncated to what the patch can carry.
    radius = min(int(round(3.0 * sigma)), h - 1, w - 1)
    if radius < 1:
        return field
    kernel = _gaussian_kernel_1d(sigma, radius, field.device, field.dtype)
    out = F.pad(field, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    out = F.conv2d(out, kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return out


def _component_area_map(mask: torch.Tensor) -> torch.Tensor:
    """The area of the connected component each pixel belongs to.

    The components are found with a MINIMUM-LABEL PROPAGATION on the
    device: every pixel of the mask starts with its own flattened index
    as a label, and the labels are then relaxed to the minimum over the
    3x3 neighbourhood until they stop changing.  Two pixels of the same
    component therefore end up carrying the same label, and counting
    the pixels per label gives the component's area.

    The relaxation must not let the labels leave the mask: a pixel
    outside the mask keeps a large sentinel value and is written back
    after every round, so a background pixel can never lower the label
    of a foreground one and merge two components that are not touching.
    The sentinel is larger than every real label, so it also never
    lowers the label of a foreground pixel from the other side.

    The relaxation runs until the labels stop changing or ``H + W``
    rounds have passed, which is the bound the pre-registration fixes.

    The whole computation stays on the input's device and is built from
    ``torch`` alone: no numpy, no scipy and no CPU union-find.  The
    E44 implementation labelled the components on the CPU with numpy,
    which broke the autograd graph and left the term with no gradient
    at all.

    Args:
        mask: ``(N, 1, H, W)`` binary mask; 1 marks a pixel that belongs
            to a component.

    Returns:
        ``(N, 1, H, W)`` float tensor: the number of pixels of the
        component the pixel belongs to.  A pixel outside the mask
        belongs to no component and is given an area of one, so that
        dividing by the area leaves it untouched.
    """
    n, _, h, w = mask.shape
    device = mask.device
    inside = mask > 0.5
    # One unique label per pixel: the flattened index.  The sentinel for
    # the pixels outside the mask is larger than any real label, so the
    # minimum over the neighbourhood never prefers it.
    sentinel = float(h * w)
    labels = torch.arange(h * w, device=device, dtype=torch.float32).view(1, 1, h, w)
    labels = labels.expand(n, 1, h, w).clone()
    labels = torch.where(inside, labels, torch.full_like(labels, sentinel))
    for _ in range(h + w):
        # The minimum over the 3x3 neighbourhood is the negated maximum
        # pool of the negated labels.  ``max_pool2d`` pads with -inf,
        # which is harmless here: the out-of-frame neighbours drop out
        # of the maximum instead of winning it.
        neighbour = -F.max_pool2d(-labels, 3, stride=1, padding=1)
        relaxed = torch.where(
            inside,
            torch.minimum(labels, neighbour),
            torch.full_like(labels, sentinel),
        )
        if torch.equal(relaxed, labels):
            break
        labels = relaxed
    # Count the pixels per label.  The labels of one sample are unique
    # within the sample, so the count of a label is the area of its
    # component.  The sentinel's count is the number of background
    # pixels and is discarded below.
    index = labels.reshape(n, -1).to(torch.long)
    counts = torch.zeros(n, h * w + 1, device=device, dtype=torch.float32)
    counts.scatter_add_(
        1, index, torch.ones(n, h * w, device=device, dtype=torch.float32)
    )
    area = counts.gather(1, index).view(n, 1, h, w)
    # A pixel outside the mask belongs to no component: an area of one
    # leaves its weight alone.
    return torch.where(inside, area, torch.ones_like(area)).detach()


def _area_normalized_weights(
    weights: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Divide every pixel's weight by the area of its component.

    A component of ``A`` pixels whose weights are all ``w`` then carries
    a total weight of ``A * w / A = w``, i.e. roughly its average
    per-pixel weight.  A one-pixel component and a hundred-pixel
    component therefore contribute comparably to the term, whatever
    their sizes.

    The area map is a CONSTANT: it is detached inside
    ``_component_area_map`` and it is not part of the autograd graph.
    The gradient of the term still flows only through the weights
    themselves; were the areas differentiable, they would add a second
    gradient path whose size nothing controls.

    Args:
        weights: ``(N, 1, H, W)`` the per-pixel weights.
        mask: ``(N, 1, H, W)`` binary mask of the components.

    Returns:
        The normalised weights, same shape as ``weights``.
    """
    return weights / _component_area_map(mask)


class ClDiceLoss(nn.Module):
    """The soft clDice connectivity term.

    Args:
        threshold: the rain rate in mm per 6-minute accumulation at which
            the field is binarised.
        power: the exponent of the component survival probability.
        softness: the width of the soft threshold, in mm.  A hard
            threshold would leave the term without a gradient while the
            forecast is still below it, which is the state a model
            starts in.
        eps: the numerical floor in the Dice ratio.
        clamp_mm: the physical ceiling in mm the forecast is truncated
            to before the term reads it, or ``None`` for no truncation.
            E44 measured the term's gradient concentrating on the top
            1 % of the pixels (0.9399 of the gradient L1), which pushed
            a few pixels to the model's ``expm1`` ceiling of 402.43 mm.
            The truncation cuts that path: a pixel above the ceiling
            gets no gradient from this term at all.
        truth_smooth_sigma: the width in pixels of the separable Gaussian
            blur applied to the truth before the term reads it, or 0 for
            no blur.  The truth's support becomes a region instead of a
            handful of isolated pixels, so the term compares structures
            rather than spikes.  The prediction side is NOT blurred:
            the forecast is what the term is supposed to shape.
        area_normalize: whether the per-pixel weights are divided by the
            area of the connected component they belong to.  A one-pixel
            component and a hundred-pixel component then contribute
            comparably to the term.  The area map is a constant, so the
            gradient still flows only through the original weights.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        power: float = 3.0,
        softness: float = 0.5,
        eps: float = 1e-6,
        clamp_mm: float | None = None,
        truth_smooth_sigma: float = 0.0,
        area_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.power = float(power)
        self.softness = float(softness)
        self.eps = float(eps)
        self.clamp_mm = None if clamp_mm is None else float(clamp_mm)
        self.truth_smooth_sigma = float(truth_smooth_sigma)
        self.area_normalize = bool(area_normalize)

    def _switches_off(self) -> bool:
        """Whether every E45 mechanism is disabled.

        When they all are, the term is the E44 term and must stay
        bit-for-bit identical to it: an E45 run that is supposed to
        isolate one mechanism would otherwise measure the difference
        between two implementations instead of the difference between
        two mechanisms.
        """
        return (
            self.clamp_mm is None
            and self.truth_smooth_sigma <= 0.0
            and not self.area_normalize
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the term.

        Args:
            pred: the forecast field in mm.  The training loop passes
                ``(B, T, 1, H, W)``, so the leading dimensions are folded
                into the batch: the connectivity is measured on every
                (window, lead) frame separately.
            target: the truth field, same shape.

        Returns:
            ``(value, components)`` where ``value`` is a scalar tensor
            that carries the gradient and ``components`` holds the
            diagnostic numbers.
        """
        if self._switches_off():
            return self._forward_reference(pred, target)
        return self._forward_augmented(pred, target)

    def _forward_reference(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """The E44 term, unchanged.

        This is the reference implementation the E45 mechanisms are
        measured against, and it is what the term evaluates to when all
        three switches are off.  It is kept as its own method -- and not
        as a branch inside the general path -- so that the equality of
        the two is a property of the code rather than of a careful
        reading of it.

        Args:
            pred: the forecast field in mm.
            target: the truth field.

        Returns:
            ``(value, components)``.
        """
        p = pred.clamp_min(0.0)
        t = target.clamp_min(0.0)
        # Fold the leading dimensions into the batch and drop the channel
        # axis: the connectivity is a property of a single field.
        if p.dim() > 3:
            p = p.reshape(-1, p.shape[-2], p.shape[-1])
            t = t.reshape(-1, t.shape[-2], t.shape[-1])
        p = p.unsqueeze(1)                             # (N, 1, H, W)
        t = t.unsqueeze(1)

        # A HARD threshold would leave the term with no gradient at all
        # whenever the forecast has no pixel above the threshold, which
        # is exactly the state a model starts in: it under-reports the
        # rain, so the binarised forecast is empty and the topology term
        # can never wake it up.  The threshold is therefore applied
        # SOFTLY -- a sigmoid centred on it -- so that a pixel below the
        # threshold still carries a gradient, while the connectivity is
        # still measured at the strong-rain scale.
        p_bin = torch.sigmoid((p - self.threshold) / self.softness)
        t_bin = torch.sigmoid((t - self.threshold) / self.softness).detach()

        # The component survival probabilities, computed on the soft
        # mask so the connectivity is what is being compared.
        p_w = _soft_component_weight(p_bin * p, self.power)
        t_w = _soft_component_weight(t_bin * t, self.power)

        # The overlap of the two component sets, approximated by the
        # soft Dice of the two weighted fields.
        inter = (p_w * t_w).sum(dim=(1, 2, 3))
        denom = p_w.sum(dim=(1, 2, 3)) + t_w.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        value = 1.0 - dice.mean()

        components = {
            "cldice": float(value.detach()),
            "pred_area": float((p >= self.threshold).sum().detach()),
            "truth_area": float((t >= self.threshold).sum().detach()),
            "pred_soft_area": float(p_bin.sum().detach()),
        }
        return value, components

    def _forward_augmented(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """The E45 term: the E44 term plus the three mechanisms.

        The three are independent and each one is a switch, so that an
        arm can isolate any one of them.  They act on different things:
        the truncation acts on the forecast's amplitude, the smoothing
        acts on the truth's support, and the area normalisation acts on
        the weights.  The model's output, the backbone and the
        load-bearing ``balanced`` objective are not touched here.

        Args:
            pred: the forecast field in mm.
            target: the truth field.

        Returns:
            ``(value, components)``.
        """
        # (3) Amplitude truncation.  The forecast is cut to a physical
        # ceiling BEFORE the threshold and the weights are computed, so
        # that a pixel above the ceiling cannot be pushed further up by
        # this term: the ceiling removes the path that reached the
        # model's own 402.43 mm clamp.  ``clamp`` is used and not a
        # hand-written mask, so the gradient of a truncated pixel is
        # exactly zero rather than merely small.
        if self.clamp_mm is not None:
            p = pred.clamp(0.0, self.clamp_mm)
        else:
            p = pred.clamp_min(0.0)
        t = target.clamp_min(0.0)

        # Fold the leading dimensions into the batch and drop the channel
        # axis: the connectivity is a property of a single field.
        if p.dim() > 3:
            p = p.reshape(-1, p.shape[-2], p.shape[-1])
            t = t.reshape(-1, t.shape[-2], t.shape[-1])
        p = p.unsqueeze(1)                             # (N, 1, H, W)
        t = t.unsqueeze(1)

        # (1) The truth's support is read through a separable Gaussian
        # blur, so that it is a region rather than a handful of isolated
        # pixels: the term then compares structures instead of spikes.
        # The blurred truth is detached, like the truth's mask below it:
        # the truth is data, the term must not ask for a gradient
        # through it, and the blur must not reach the forecast.
        if self.truth_smooth_sigma > 0.0:
            t = _gaussian_blur_2d(t, self.truth_smooth_sigma).detach()

        # A HARD threshold would leave the term with no gradient at all
        # whenever the forecast has no pixel above the threshold, which
        # is exactly the state a model starts in: it under-reports the
        # rain, so the binarised forecast is empty and the topology term
        # can never wake it up.  The threshold is therefore applied
        # SOFTLY -- a sigmoid centred on it -- so that a pixel below the
        # threshold still carries a gradient, while the connectivity is
        # still measured at the strong-rain scale.
        p_bin = torch.sigmoid((p - self.threshold) / self.softness)
        t_bin = torch.sigmoid((t - self.threshold) / self.softness).detach()

        # The component survival probabilities, computed on the soft
        # mask so the connectivity is what is being compared.
        p_w = _soft_component_weight(p_bin * p, self.power)
        t_w = _soft_component_weight(t_bin * t, self.power)

        # (2) Area normalisation.  Each pixel's weight is divided by the
        # area of the connected component it belongs to, so that a
        # one-pixel component and a hundred-pixel component contribute
        # comparably to the term: the total weight of a component becomes
        # roughly its average per-pixel weight.  The area map is a
        # constant -- it multiplies the weight and is not part of the
        # autograd graph -- so the gradient still flows only through the
        # original weights, and the areas cannot turn into a second,
        # uncontrolled gradient path.
        if self.area_normalize:
            p_w = _area_normalized_weights(
                p_w, (p >= self.threshold).to(p.dtype)
            )
            t_w = _area_normalized_weights(
                t_w, (t >= self.threshold).to(t.dtype)
            )

        # The overlap of the two component sets, approximated by the
        # soft Dice of the two weighted fields.
        inter = (p_w * t_w).sum(dim=(1, 2, 3))
        denom = p_w.sum(dim=(1, 2, 3)) + t_w.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        value = 1.0 - dice.mean()

        components = {
            "cldice": float(value.detach()),
            "pred_area": float((p >= self.threshold).sum().detach()),
            "truth_area": float((t >= self.threshold).sum().detach()),
            "pred_soft_area": float(p_bin.sum().detach()),
        }
        return value, components


class TopoCriterion(nn.Module):
    """The balanced objective plus the topology term.

    The main term is the load-bearing ``balanced`` objective, used
    unchanged; the topology term is added on top with a single weight.
    There is no area penalty and no second sparsity term, because a
    second term whose optimum points the same way as the regression
    drives the field to zero -- that was E42's lesson.

    The E45 mechanisms are passed to the topology term and change
    nothing else: the balanced objective is still the one that carries
    the model, and the weight of the topology term is still the single
    number that was fixed in advance.

    Args:
        base: the main criterion.
        topo_weight: the weight of the topology term.
        threshold: the rain rate at which the field is binarised.
        clamp_mm: the physical ceiling in mm the forecast is truncated
            to inside the topology term, or ``None`` for no truncation.
        truth_smooth_sigma: the width in pixels of the separable
            Gaussian blur applied to the truth inside the topology
            term, or 0 for no blur.
        area_normalize: whether the topology term's per-pixel weights
            are divided by the area of their connected component.
    """

    def __init__(
        self,
        base: nn.Module,
        topo_weight: float = 0.1,
        threshold: float = 2.0,
        clamp_mm: float | None = None,
        truth_smooth_sigma: float = 0.0,
        area_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.topo_weight = float(topo_weight)
        self.cldice = ClDiceLoss(
            threshold=threshold,
            clamp_mm=clamp_mm,
            truth_smooth_sigma=truth_smooth_sigma,
            area_normalize=area_normalize,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss.

        Args:
            pred: the forecast field.
            target: the truth field.
            *args: forwarded to the base criterion.
            **kwargs: forwarded to the base criterion.

        Returns:
            ``(total, components)``.
        """
        base_value, components = self.base(pred, target, *args, **kwargs)
        topo_value, topo_components = self.cldice(pred, target)
        total = base_value + self.topo_weight * topo_value
        components = dict(components or {})
        components.update({f"topo_{k}": v for k, v in topo_components.items()})
        components["topo_loss"] = float(topo_value.detach())
        components["total"] = float(total.detach())
        return total, components
