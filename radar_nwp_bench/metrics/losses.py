"""Training and evaluation losses.

The regression target is ``log1p(mm)`` (see :func:`..data.transforms.normalize_rain`),
so the base loss is a masked MSE in log space. The optional weighted BCE
operates directly in millimetres, where it discriminates rain from no-rain --
something an L2 loss in log space cannot do well, because ``log1p(0) == 0``
makes the dry pixels dominate the gradient.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _floor_at_zero(x: torch.Tensor) -> torch.Tensor:
    """Clamp a tensor at zero while keeping the gradient path alive.

    ``torch.clamp_min(x, 0)`` has a zero derivative below the threshold, so
    every negative prediction would be cut off from the loss. A model whose
    output drifts below zero would then receive no learning signal and could
    never recover.

    The straight-through form returns the same *value* as the clamp, but
    routes the gradient of the unclamped tensor, so the model is still told
    which way to move.

    Args:
        x: the tensor to floor.

    Returns:
        ``max(x, 0)`` in value, with the gradient of ``x``.
    """
    return x + (x.clamp_min(0.0) - x).detach()


def inverse_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    rain_mask: torch.Tensor | None = None,
    weight_rain: float = 3.0,
) -> torch.Tensor:
    """Weighted MSE in log space, biased towards rainy pixels.

    Args:
        pred: ``(B, T, 1, H, W)`` prediction in log space.
        target: ``(B, T, 1, H, W)`` target in log space.
        rain_mask: optional ``(B, T, 1, H, W)`` mask of rainy pixels.
        weight_rain: relative weight of rainy pixels.
    """
    se = (pred - target) ** 2
    if rain_mask is None:
        return se.mean()
    w = 1.0 + (weight_rain - 1.0) * rain_mask
    return (se * w).sum() / w.sum().clamp_min(1.0)


class HierarchicalPrecipLoss(nn.Module):
    """A loss that supervises the exceedance probabilities, if the model has them.

    The regression in log space places the rain well but says nothing about
    how much of it there is: the measured frequency bias of the models
    trained under it is 0.08 to 0.33, so the heavy-rain scores collapse.
    A model that also predicts a nested set of exceedance probabilities can
    be told directly what it is getting wrong at each threshold, and the
    thresholds can be weighted differently -- the light rain can be protected
    against false alarms while the heavy rain is pushed for recall.

    The loss does not know which models those are. It looks for the
    probabilities on the model, and if they are not there it degrades to the
    plain regression. That is a capability check rather than a name check:
    the name of a model is not evidence of what it can do, and a switch on
    the name breaks the moment a model is renamed or aliased.

    Args:
        thresholds: the rain rates in mm per 6-minute accumulation, in
            increasing order.
        alpha_prob: the weight of the probability term against the
            regression.
        use_focal: whether to modulate the probability term by
            ``(1 - p_t)^2``. The modulation damps the gradient where the
            model is already confident, which on a sparse heavy-rain field
            can blunt exactly the recall the term is there to encourage.
        tier_weights: the ``(miss, false alarm)`` weights per threshold.
            The default protects the light rain against false alarms and
            pushes the heavy rain towards recall.
    """

    #: The default per-threshold weights: ``(miss, false alarm)``, keyed
    #: by the threshold they belong to.
    #:
    #: The weights are looked up by the threshold rather than taken by
    #: position. A run that scores a different set of thresholds -- the
    #: business set is ``(0.1, 0.5, 2.0, 4.0)`` -- must not silently
    #: inherit the weights of a different tier: with positional indexing
    #: the 2.0 mm threshold was given the weights meant for 1.0 mm, and the
    #: heaviest tier's weights were never used at all.
    DEFAULT_TIER_WEIGHTS_BY_THRESHOLD: dict[float, tuple[float, float]] = {
        0.1: (1.0, 3.0),   # punish the false alarm hard
        0.5: (1.5, 2.0),   # balanced
        1.0: (2.5, 1.0),   # encourage the convective cell
        2.0: (4.0, 0.5),   # push for the heavy rain
        4.0: (6.0, 0.2),   # the extreme
    }

    #: The default per-threshold weights in the canonical order.
    DEFAULT_TIER_WEIGHTS: tuple[tuple[float, float], ...] = (
        (1.0, 3.0),
        (1.5, 2.0),
        (2.5, 1.0),
        (4.0, 0.5),
        (6.0, 0.2),
    )

    @classmethod
    def weights_for(
        cls, thresholds: tuple[float, ...]
    ) -> tuple[tuple[float, float], ...]:
        """The default weights for a set of thresholds.

        Args:
            thresholds: the rain rates the loss is scored at.

        Returns:
            One ``(miss, false alarm)`` pair per threshold, in the order
            the thresholds were given.
        """
        weights = []
        for threshold in thresholds:
            if threshold in cls.DEFAULT_TIER_WEIGHTS_BY_THRESHOLD:
                weights.append(cls.DEFAULT_TIER_WEIGHTS_BY_THRESHOLD[threshold])
            else:
                # A threshold the table does not know gets the median
                # weight, so an unusual threshold is neither privileged
                # nor starved.
                weights.append((2.5, 1.0))
        return tuple(weights)

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0),
        alpha_prob: float = 0.3,
        use_focal: bool = False,
        tier_weights: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        super().__init__()
        self.thresholds = tuple(thresholds)
        self.alpha_prob = alpha_prob
        self.use_focal = use_focal
        self.tier_weights = tuple(
            tier_weights if tier_weights is not None
            else self.weights_for(self.thresholds)
        )
        if len(self.tier_weights) < len(self.thresholds):
            raise ValueError(
                f"{len(self.thresholds)} thresholds need "
                f"{len(self.thresholds)} tier weights, got "
                f"{len(self.tier_weights)}"
            )

    @staticmethod
    def _find_cum_probs(model: nn.Module | None) -> torch.Tensor | None:
        """Look for the exceedance probabilities on the model.

        The head keeps them from the forward pass, so the loss reads them
        from there rather than running the model a second time. The search
        is a capability check: the model exposes them, or one of its
        submodules does.

        Args:
            model: the model, or ``None``.

        Returns:
            The probabilities, or ``None`` when the model does not have
            them.
        """
        if model is None:
            return None

        probs = getattr(model, "last_cum_probs", None)
        if isinstance(probs, torch.Tensor):
            return probs

        # A model that keeps them on the head rather than on itself.
        for name in ("quantum_povm_head", "q_head", "povm_head"):
            head = getattr(model, name, None)
            if head is None:
                continue
            probs = getattr(head, "last_cum_probs", None)
            if isinstance(probs, torch.Tensor):
                return probs
        return None

    @staticmethod
    def _find_head_thresholds(model: nn.Module | None) -> tuple[float, ...]:
        """The thresholds the model's probability channels answer.

        The exceedance probabilities come out of the head one per
        threshold, in the head's own order, which is not necessarily the
        order the loss asks about them in. A loss that scores the
        thresholds (0.1, 0.5, 2.0, 4.0) must therefore be told which
        channel answers which threshold, and the answer has to be looked
        up by value.

        Matching them by position is not a small mistake. The head's
        channels answer (0.1, 0.5, 1.0, 2.0, 4.0), so a loss that
        asked about (0.1, 0.5, 2.0, 4.0) and took them in order was
        supervising P(R >= 1.0) against the truth's "at or above 2 mm"
        indicator and P(R >= 2.0) against the truth's "at or above 4 mm"
        one. The consequence was measured: the model was trained to make
        P(R >= 2 mm) vanish almost everywhere, and the rate it
        reconstructed saturated at the bottom of the heavy bin.

        Args:
            model: the model, or ``None``.

        Returns:
            The thresholds the channels answer, or an empty tuple when
            the model has no head to ask.
        """
        if model is None:
            return ()
        for name in ("quantum_povm_head", "q_head", "povm_head"):
            head = getattr(model, name, None)
            if head is None:
                continue
            # The head publishes the thresholds its channels actually
            # answer, which is not the same as the ones it was built
            # with when the bin count leaves no room for the dry bin.
            thresholds = getattr(head, "channel_thresholds", None)
            if thresholds is None:
                thresholds = getattr(head, "thresholds", None)
            if thresholds:
                return tuple(float(t) for t in thresholds)
        return ()

    @classmethod
    def _channel_for(
        cls, head_thresholds: tuple[float, ...], threshold: float
    ) -> int | None:
        """The channel that answers a threshold.

        Args:
            head_thresholds: the thresholds the model's channels answer.
            threshold: the threshold to look for.

        Returns:
            The channel index, or ``None`` when the model does not carry
            that threshold at all.
        """
        for index, value in enumerate(head_thresholds):
            if abs(value - threshold) < 1e-6:
                return index
        return None

    def forward(
        self,
        pred_rain: torch.Tensor,
        target_rain: torch.Tensor,
        model: nn.Module | None = None,
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss.

        Args:
            pred_rain: ``(B, T, 1, H, W)`` predicted rainfall in millimetres.
            target_rain: matching observed rainfall in millimetres.
            model: the model, looked at for the exceedance probabilities.
            **kwargs: ignored; the training loop passes more than this
                loss needs.

        Returns:
            The scalar loss and its components.
        """
        pred_log = torch.log1p(pred_rain.clamp_min(0.0))
        target_log = torch.log1p(target_rain.clamp_min(0.0))
        reg_loss = F.smooth_l1_loss(pred_log, target_log)

        cum_probs = self._find_cum_probs(model)
        if cum_probs is None or self.alpha_prob <= 0.0:
            total = reg_loss
            return total, {
                "reg_loss": float(reg_loss.detach()),
                "total": float(total.detach()),
            }

        # The probabilities come out of the head as (B*T, K, H, W) and the
        # target as (B, T, 1, H, W); both are flattened over the batch and
        # the lead time so the two line up.
        if target_rain.dim() == 5:
            b, t, _, h, w = target_rain.shape
            target_flat = target_rain.reshape(b * t, 1, h, w)
        else:
            target_flat = target_rain
        if cum_probs.dim() == 5:
            b, t, k, h, w = cum_probs.shape
            cum_probs = cum_probs.reshape(b * t, k, h, w)

        prob_loss = pred_rain.new_zeros(())
        # The probabilities come out of the head one per threshold, in the
        # head's own order, and the channel that answers a threshold is
        # found by value rather than by position: the loss asks about
        # the thresholds it was configured with, and the head may carry a
        # different set.
        head_thresholds = self._find_head_thresholds(model)
        n_tiers = 0
        for k, threshold in enumerate(self.thresholds):
            w_miss, w_fa = self.tier_weights[k]

            channel = (
                self._channel_for(head_thresholds, threshold)
                if head_thresholds
                else k
            )
            if channel is None or channel >= cum_probs.shape[1]:
                # The model does not carry this threshold, so there is no
                # probability to supervise it with.
                continue
            n_tiers += 1

            p_k = cum_probs[:, channel : channel + 1].clamp(1e-6, 1.0 - 1e-6)
            y_k = (target_flat >= threshold).to(pred_rain.dtype)

            # The asymmetric binary cross-entropy: a miss and a false alarm
            # cost different amounts, and the ratio is set per threshold.
            loss_miss = -w_miss * y_k * torch.log(p_k)
            loss_fa = -w_fa * (1.0 - y_k) * torch.log(1.0 - p_k)

            if self.use_focal:
                # The modulator damps the gradient where the model is
                # already confident. On a field where the heavy rain is
                # rare, that is exactly where the recall the term is
                # meant to encourage lives, so it is a switch and not a
                # default.
                pt = torch.where(y_k == 1.0, p_k, 1.0 - p_k)
                focal_weight = (1.0 - pt).square()
                tier_loss = torch.mean(
                    focal_weight * (loss_miss + loss_fa)
                )
            else:
                tier_loss = torch.mean(loss_miss + loss_fa)
            prob_loss = prob_loss + tier_loss

        prob_loss = prob_loss / max(n_tiers, 1)
        total = reg_loss + self.alpha_prob * prob_loss
        return total, {
            "reg_loss": float(reg_loss.detach()),
            "prob_loss": float(prob_loss.detach()),
            "total": float(total.detach()),
        }


class HierarchicalFieldCSILoss(nn.Module):
    """A loss that supervises the exceedance probabilities over the whole field.

    Two things have been measured on this benchmark, and this loss is built
    from both of them.

    The first: the models trained with a plain regression predict between
    two and a third of the rain that actually fell. The frequency bias of
    the family sits at 0.08 to 0.33, and the extreme-rain scores collapse
    as a result. Something has to push the model to predict more.

    The second: the per-pixel version of that push does not work. Measured
    over the archive, the false-alarm term of a per-pixel asymmetric
    cross-entropy outweighs the miss term by seventeen to forty-two times
    at every threshold, and the dry pixels are ninety-seven per cent of the
    field, so the loss is overwhelmingly a statement about the dry pixels.
    The nesting of the exceedance probabilities makes it worse: the
    light-rain probability is what the dry bin leaves behind, so raising
    the heavy rain at a pixel necessarily raises the light-rain
    probability there, which is exactly the false alarm the light tier is
    meant to punish.

    The soft-CSI loss avoids both problems by counting over the whole
    field and taking the ratio, which is the score itself. This loss keeps
    that structure and adds the tier information: the soft mask of a tier
    is the model's own exceedance probability rather than a step function
    of the prediction, and the counts are formed per tier.

    The soft mask is what makes the trade continuous. A pixel where the
    model says "70 % chance of exceeding 2 mm" contributes 0.7 to the
    hits and 0.7 to the false alarms at the same time, so the model can
    commit to the heavy rain where it is confident without being charged
    the full false alarm for it. With a binary mask the model has only two
    options, and the conservative one is always cheaper.

    Args:
        thresholds: the rain rates in mm per 6-minute accumulation.
        alpha_prob: the weight of the tier term against the regression.
        tier_weights: the ``(miss, false alarm)`` weights per threshold.
        field_ratio: whether the tier term is formed from the field-wide
            counts (the default) or per pixel. The field-wide form is what
            makes it work; the per-pixel form is kept for the comparison.
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.1, 0.5, 2.0, 4.0),
        alpha_prob: float = 0.3,
        tier_weights: tuple[tuple[float, float], ...] | None = None,
        field_ratio: bool = True,
    ) -> None:
        super().__init__()
        self.thresholds = tuple(thresholds)
        self.alpha_prob = alpha_prob
        self.field_ratio = field_ratio
        self.tier_weights = tuple(
            tier_weights if tier_weights is not None
            else HierarchicalPrecipLoss.weights_for(self.thresholds)
        )

    @staticmethod
    def _find_cum_probs(model: nn.Module | None) -> torch.Tensor | None:
        """Look for the exceedance probabilities on the model."""
        return HierarchicalPrecipLoss._find_cum_probs(model)

    @staticmethod
    def _find_head_thresholds(model: nn.Module | None) -> tuple[float, ...]:
        """The thresholds the model's probability channels answer."""
        return HierarchicalPrecipLoss._find_head_thresholds(model)

    @staticmethod
    def _channel_for(
        head_thresholds: tuple[float, ...], threshold: float
    ) -> int | None:
        """The channel that answers a threshold."""
        return HierarchicalPrecipLoss._channel_for(head_thresholds, threshold)

    def forward(
        self,
        pred_rain: torch.Tensor,
        target_rain: torch.Tensor,
        model: nn.Module | None = None,
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss.

        Args:
            pred_rain: ``(B, T, 1, H, W)`` predicted rainfall in millimetres.
            target_rain: matching observed rainfall in millimetres.
            model: the model, looked at for the exceedance probabilities.

        Returns:
            The scalar loss and its components.
        """
        pred_log = torch.log1p(pred_rain.clamp_min(0.0))
        target_log = torch.log1p(target_rain.clamp_min(0.0))
        reg_loss = F.smooth_l1_loss(pred_log, target_log)

        cum_probs = self._find_cum_probs(model)
        if cum_probs is None or self.alpha_prob <= 0.0:
            total = reg_loss
            return total, {
                "reg_loss": float(reg_loss.detach()),
                "total": float(total.detach()),
            }

        if target_rain.dim() == 5:
            b, t, _, h, w = target_rain.shape
            target_flat = target_rain.reshape(b * t, 1, h, w)
        else:
            target_flat = target_rain
        if cum_probs.dim() == 5:
            b, t, k, h, w = cum_probs.shape
            cum_probs = cum_probs.reshape(b * t, k, h, w)

        tier_loss = pred_rain.new_zeros(())
        # The channel that answers a threshold is found by value, not by
        # position. The head's channels answer its own thresholds, which
        # are not necessarily the ones this loss was configured with.
        head_thresholds = self._find_head_thresholds(model)
        n_tiers = 0
        for k, threshold in enumerate(self.thresholds):
            w_miss, w_fa = self.tier_weights[k]

            channel = (
                self._channel_for(head_thresholds, threshold)
                if head_thresholds
                else k
            )
            if channel is None or channel >= cum_probs.shape[1]:
                continue
            n_tiers += 1

            p_k = cum_probs[:, channel : channel + 1].clamp(0.0, 1.0)
            y_k = (target_flat >= threshold).to(pred_rain.dtype)

            if self.field_ratio:
                # The counts over the whole field, with the probability as
                # the soft mask. The ratio they form is the score the
                # verification computes, so the gradient it produces is
                # already pointed at the score.
                hits = torch.sum(p_k * y_k)
                misses = torch.sum((1.0 - p_k) * y_k)
                false_alarms = torch.sum(p_k * (1.0 - y_k))
                csi = (hits + 1e-5) / (hits + misses + false_alarms + 1e-5)

                # The asymmetric part: a miss and a false alarm are
                # charged differently, and the charge is on the recall
                # and on the precision rather than on the pixels.
                recall = (hits + 1e-5) / (hits + misses + 1e-5)
                precision = (hits + 1e-5) / (hits + false_alarms + 1e-5)
                tier_loss = tier_loss + w_miss * (1.0 - recall) + w_fa * (
                    1.0 - precision
                )
            else:
                # The per-pixel form, kept for the comparison: it is the
                # variant that does not work, and the reason it does not
                # is measured in tools/_why_hierarchical.py.
                loss_miss = -w_miss * y_k * torch.log(
                    p_k.clamp(1e-6, 1.0)
                )
                loss_fa = -w_fa * (1.0 - y_k) * torch.log(
                    (1.0 - p_k).clamp(1e-6, 1.0)
                )
                tier_loss = tier_loss + torch.mean(loss_miss + loss_fa)

        tier_loss = tier_loss / max(n_tiers, 1)
        total = reg_loss + self.alpha_prob * tier_loss
        return total, {
            "reg_loss": float(reg_loss.detach()),
            "tier_loss": float(tier_loss.detach()),
            "total": float(total.detach()),
        }


def weighted_bce(
    pred_mm: torch.Tensor,
    target_mm: torch.Tensor,
    threshold: float = 0.1,
    pos_weight: float = 4.0,
) -> torch.Tensor:
    """Binary cross-entropy on the rain/no-rain decision, in millimetres.

    The millimetre prediction is not a logit -- it is unbounded and mostly
    below one, and feeding it to the BCE as one makes the loss meaningless.
    It is mapped to a logit through the log accumulation, with the
    threshold as the decision boundary, which is the same mapping the
    benchmark's own loss uses.

    Args:
        pred_mm: predicted rainfall in millimetres.
        target_mm: observed rainfall in millimetres.
        threshold: the rain rate above which a pixel counts as rain.
        pos_weight: the weight of the rainy class.

    Returns:
        The scalar loss.
    """
    p = (pred_mm >= threshold).float()
    t = (target_mm >= threshold).float()
    weight = 1.0 + (pos_weight - 1.0) * t
    logit = torch.log1p(pred_mm.clamp_min(0.0)) - float(np.log1p(threshold))
    bce = F.binary_cross_entropy_with_logits(
        logit, t, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=pred_mm.device),
    )
    return (bce * weight).mean()


class PrecipitationLoss(nn.Module):
    """The benchmark's standard training objective.

    The regression is performed in ``log1p(mm)`` space, because rainfall is
    zero-inflated and heavy-tailed: an L2 loss on raw millimetres is dominated
    by the dry pixels and by a handful of extreme cells.

    Args:
        rain_threshold: mm above which a pixel counts as rain (for the BCE term).
        bce_weight: weight of the rain/no-rain term.
        physics_weight: weight of an optional model-specific physics penalty.
        target_is_log: whether ``target`` is already in ``log1p(mm)`` space.
            ``False`` (the default) means both ``pred`` and ``target`` are in
            millimetres and the transform is applied here.
    """

    def __init__(
        self,
        rain_threshold: float = 0.1,
        bce_weight: float = 0.1,
        physics_weight: float = 0.0,
        target_is_log: bool = False,
    ) -> None:
        super().__init__()
        self.rain_threshold = rain_threshold
        self.bce_weight = bce_weight
        self.physics_weight = physics_weight
        self.target_is_log = target_is_log

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module | None = None,
        **extra: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss.

        Args:
            pred: ``(B, T, 1, H, W)`` prediction. Millimetres when
                ``target_is_log`` is False, otherwise ``log1p(mm)``.
            target: matching target in the same space as ``pred``.
            model: optional model, used to pull in a physics regulariser when
                the adapter exposes one.

        Returns:
            ``(total_loss, components)``.
        """
        if self.target_is_log:
            pred_log = pred
            target_log = target
            pred_mm = torch.expm1(pred_log.clamp(max=12.0))
            target_mm = torch.expm1(target_log.clamp(max=12.0))
        else:
            # The floor must come *before* log1p: log1p(x) is NaN for x <= -1.
            # A plain clamp_min would also zero the gradient for every negative
            # prediction, so a model that drifts negative would stop learning
            # entirely. The straight-through form keeps the value physical
            # while letting the gradient through the floor.
            pred_mm = _floor_at_zero(pred)
            target_mm = target.clamp_min(0.0)
            pred_log = torch.log1p(pred_mm)
            target_log = torch.log1p(target_mm)

        rain_mask = (target_mm >= self.rain_threshold).float()

        mse = inverse_mse(pred_log, target_log, rain_mask)
        # BCE needs a logit, and the predicted log accumulation is not one.
        #
        # log1p(mm) is never negative, so feeding it to the BCE as a logit
        # means sigmoid(log1p(mm)) is never below 0.5: a dry pixel cannot be
        # predicted with a confidence below one half, and the term
        # saturates at -log(1/2) = 0.693 no matter how dry the pixel is.
        # The decision boundary has to sit at the rain threshold itself,
        # so the logit is the log accumulation *relative to* the threshold:
        # zero exactly when the prediction crosses the threshold, negative
        # below it and positive above it.
        #
        # The gradient still reaches every pixel, so the term sharpens the
        # rain/no-rain decision without introducing a second output head.
        logit = pred_log - float(np.log1p(self.rain_threshold))
        bce = F.binary_cross_entropy_with_logits(
            logit, rain_mask, reduction="mean",
            pos_weight=torch.tensor(4.0, device=pred.device),
        )
        total = mse + self.bce_weight * bce

        components = {"mse": float(mse.detach()), "bce": float(bce.detach())}

        if self.physics_weight > 0 and model is not None:
            penalty = self._physics_penalty(model, extra)
            if penalty is not None:
                total = total + self.physics_weight * penalty
                components["physics"] = float(penalty.detach())

        components["total"] = float(total.detach())
        return total, components

    @staticmethod
    def _physics_penalty(
        model: nn.Module, extra: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Pull a physics regulariser off the model if it exposes one.

        The regularisers differ in what they need: the FURECast self-consistency
        penalty works on the prediction alone, while the MMF-RNN consistency
        penalty needs the two inputs it is about to be consistent with. The
        signature is inspected so that each is called with the arguments it
        actually declares, rather than guessing and failing.
        """
        import inspect

        for name in ("self_consistency_loss", "multimodal_consistency_loss"):
            fn = getattr(model, name, None)
            if not callable(fn):
                continue
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):  # pragma: no cover
                params = {}
            kwargs: dict[str, torch.Tensor] = {}
            skip = False
            for pname, param in params.items():
                if pname in ("self", "pred", "prediction"):
                    continue
                if pname in extra:
                    kwargs[pname] = extra[pname]
                elif param.default is inspect.Parameter.empty:
                    # a required argument we cannot supply: skip this method
                    skip = True
                    break
            if skip:
                continue
            try:
                value = fn(**kwargs)
            except TypeError:
                try:
                    value = fn()
                except TypeError:
                    continue
            if isinstance(value, torch.Tensor):
                return value
        return None


class WeightedCRPSLoss(nn.Module):
    """Continuous Ranked Probability Score-style loss on the rain/no-rain decision.

    A smoother alternative to the hard-threshold BCE: penalises the continuous
    difference between the predicted accumulation and the target, normalised by
    the target magnitude.
    """

    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_mm: torch.Tensor, target_mm: torch.Tensor) -> torch.Tensor:
        denom = target_mm.mean().clamp_min(self.eps)
        return ((pred_mm - target_mm).abs() / denom).mean()


class SoftFSSPrecipLoss(nn.Module):
    """Any existing objective plus a neighbourhood tolerance term.

    The mechanism is a spatial one. A pixel-wise loss punishes a storm
    that is in the right place but a couple of kilometres off twice: once
    for the miss and once for the false alarm. The cheapest way out for
    the network is to flatten the peak, and the flattened peak is what
    the heavy-rain scores collapse on.

    The fractions skill score compares the *fraction* of the
    neighbourhood that is rain in the forecast with the fraction in the
    observation, so a displaced storm scores well. This loss adds that
    comparison to whatever objective is already in place.

    Two details matter and both were got wrong in the proposal this
    replaces.

    The first is the boundary. The fraction of the neighbourhood that is
    rain has to be computed with the edge handled consistently, or the
    edges of every tile are systematically wrong and the network learns
    an edge artefact. The pooling here replicates the edge rather than
    padding with zeros, because a zero is a confident statement that
    there is no rain there.

    The second is the direction of the optimisation. The fractions skill
    score is ``1 - numerator/denominator``, so the quantity to minimise
    is the ratio itself. Clamping the ratio to one before minimising it
    would cut the gradient in the region where the forecast is worse
    than climatology, which is exactly the region that needs the most
    pressure. The ratio is therefore left unclamped above one.

    The term is added to the score-aware objective rather than replacing
    it. That objective has been measured to lift the heavy-rain score on
    every one of the eighteen models it was tried on, and the baselines
    this experiment is measured against were trained under it; throwing
    it away to test a spatial term would confound the two mechanisms and
    break the comparison with them.

    Args:
        base: the objective to build on. It is called as
            ``base(pred, target, model=model, **extra)`` and has to
            return the loss and its components.
        thresholds: the rain rates in mm per 6-minute accumulation.
        scales: the neighbourhood sizes in pixels, as window sides. A
            side of three is a three by three window.
        alpha_fss: the weight of the fractions term against the base.
        sharpness: the slope of the soft threshold. Larger values make
            the soft indicator closer to a hard one.
    """

    #: The weight of each threshold in the fractions term.
    #:
    #: Looked up by value, like the tier weights above: a run that scores
    #: a different set of thresholds must not inherit another tier's
    #: weight by position.
    DEFAULT_THRESHOLD_WEIGHTS: dict[float, float] = {
        0.1: 1.0,
        0.5: 1.5,
        1.0: 2.5,
        2.0: 4.0,
        4.0: 6.0,
    }

    def __init__(
        self,
        base,
        thresholds: tuple[float, ...] = (0.1, 0.5, 2.0, 4.0),
        scales: tuple[int, ...] = (3, 5),
        alpha_fss: float = 0.4,
        sharpness: float = 4.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.thresholds = tuple(thresholds)
        self.scales = tuple(int(s) for s in scales)
        self.alpha_fss = float(alpha_fss)
        self.sharpness = float(sharpness)
        self.threshold_weights = {
            thr: self.DEFAULT_THRESHOLD_WEIGHTS.get(thr, 2.5)
            for thr in self.thresholds
        }

    def _box_mean(self, x: torch.Tensor, size: int) -> torch.Tensor:
        """The mean over a square window, with the edge replicated.

        ``avg_pool2d`` pads with zeros, and a zero in the fraction field
        says "no rain here" with the same confidence as a measured dry
        pixel. Replicating the edge keeps the fraction meaning what it
        means in the interior.
        """
        pad = size // 2
        xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        kernel = x.new_ones(1, 1, size, size) / float(size * size)
        return F.conv2d(xp, kernel)

    def _fss_ratio(self, p_bin: torch.Tensor, t_bin: torch.Tensor,
                   size: int) -> torch.Tensor:
        """One minus the fractions skill score, per sample.

        Args:
            p_bin: ``(N, 1, H, W)`` the soft rain indicator of the forecast.
            t_bin: ``(N, 1, H, W)`` the hard rain indicator of the truth.
            size: the neighbourhood window side in pixels.

        Returns:
            ``(N,)`` the ratio to minimise, per sample.
        """
        p_frac = self._box_mean(p_bin, size)
        t_frac = self._box_mean(t_bin, size)
        numer = ((p_frac - t_frac) ** 2).flatten(1).sum(dim=1)
        denom = (p_frac**2 + t_frac**2).flatten(1).sum(dim=1)
        # The ratio is one minus the skill score. It is not clamped: a
        # forecast worse than climatology gives a ratio above one and
        # needs more pressure, not less.
        return numer / denom.clamp_min(1e-6)

    def forward(
        self,
        pred_rain: torch.Tensor,
        target_rain: torch.Tensor,
        model: nn.Module | None = None,
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss.

        Args:
            pred_rain: ``(B, T, 1, H, W)`` predicted rainfall in millimetres.
            target_rain: matching observed rainfall in millimetres.
            model: the model, looked at for the exceedance probabilities.
            **kwargs: ignored; the training loop passes more than this
                loss needs.

        Returns:
            The scalar loss and its components.
        """
        base_total, components = self.base(
            pred_rain, target_rain, model=model, **kwargs
        )
        components = dict(components)

        if self.alpha_fss <= 0.0:
            components["total"] = float(base_total.detach())
            return base_total, components

        if pred_rain.dim() == 5:
            b, t, _, h, w = pred_rain.shape
            pred_flat = pred_rain.reshape(b * t, 1, h, w)
            target_flat = target_rain.reshape(b * t, 1, h, w)
        else:
            pred_flat = pred_rain
            target_flat = target_rain

        weighted_sum = pred_rain.new_zeros(())
        weight_total = 0.0
        per_threshold: dict[str, float] = {}
        for thr in self.thresholds:
            w = self.threshold_weights[thr]
            # Both fields go through the same soft indicator. Softening
            # only the forecast and leaving the truth hard would put a
            # constant offset between the two fraction fields -- a
            # perfect forecast would score about 0.52 instead of 1.0 --
            # and that offset would swamp the base objective. With both
            # softened the comparison is fair, and the softening is
            # confined to the pixels that sit within a fraction of a
            # millimetre of the threshold, which is exactly where the
            # displacement tolerance is wanted.
            p_bin = torch.sigmoid((pred_flat - thr) * self.sharpness)
            t_bin = torch.sigmoid((target_flat - thr) * self.sharpness)

            thr_sum = pred_rain.new_zeros(())
            for size in self.scales:
                ratio = self._fss_ratio(p_bin, t_bin, size)
                thr_sum = thr_sum + ratio.mean()
                per_threshold[f"fss_{thr:g}mm_{size}px"] = float(
                    (1.0 - ratio.mean()).detach()
                )
            weighted_sum = weighted_sum + w * thr_sum / len(self.scales)
            weight_total += w

        fss_term = weighted_sum / max(weight_total, 1e-6)
        total = base_total + self.alpha_fss * fss_term

        components["fss_loss"] = float(fss_term.detach())
        components["fss_skill"] = float((1.0 - fss_term).detach())
        components.update(per_threshold)
        components["total"] = float(total.detach())
        return total, components
