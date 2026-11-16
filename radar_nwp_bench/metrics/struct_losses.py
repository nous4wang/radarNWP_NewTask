"""E41 / MASTER 4.43 -- structure-aware training objectives.

The 12 reproduced baselines and 6 in-house variants all share one output
parameterisation: a single-channel ``expm1`` head regressed with a log-space
MSE (``PrecipitationLoss``).  The local read-only probe in
``tools/_evals/struct_probe_case.json`` shows the target is instead a
zero-inflated (91.2 % zeros), amplitude-heavy-tailed (max/p99 = 4.0),
spatially coherent (connected core 108.8 px) sparse positive measure whose
support is displaced from radar (r = +0.17).

This module supplies TWO independent, single-variable interventions:

* :class:`HurdleCriterion` -- change the OUTPUT PARAMETERISATION only:
  occurrence logit + conditional amount, trained with a hurdle NLL.  The
  model still emits millimetres, so the evaluator is untouched.
* :class:`UOTCriterion` -- change the LOSS GEOMETRY only: keep the standard
  regression and add an unbalanced, entropy-regularised optimal-transport
  divergence on a coarse grid.  No model change at all.

Neither claims to add information that is not in the inputs.  They separate
the zero mass, the amplitude tail and the support from one another so that a
dense Gaussian regression no longer has to express all three in one scalar
per pixel.  Whether that helps is exactly what E41 S1/S3 measures; this
module does NOT assert that it does.

Pure PyTorch, no new third-party dependency (no geomloss / POT required).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import PrecipitationLoss


def _floor_at_zero(x: torch.Tensor) -> torch.Tensor:
    """Non-negative in value, straight-through gradient below zero."""
    return x + (x.clamp_min(0.0) - x).detach()


def _coarse_mass(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Pool a mm field to a coarse mass histogram.

    Args:
        x: ``(B, T, 1, H, W)`` millimetres.
        grid: side length of the coarse grid.

    Returns:
        ``(B * T, grid * grid)`` non-negative mass values.  Adaptive average
        pooling preserves total mass up to the area ratio, so under-reporting
        (BIAS << 1) remains visible to the unbalanced transport term.
    """
    b, t, _, h, w = x.shape
    y = x.reshape(b * t, 1, h, w).float()
    y = F.adaptive_avg_pool2d(y, (grid, grid))
    return y.reshape(b * t, grid * grid)


class HurdleNLL(nn.Module):
    """Two-part (hurdle) negative log-likelihood for zero-inflated rainfall.

    ``P(rain)`` is supervised everywhere with a Bernoulli term; the
    conditional amount ``E[mm | rain]`` is supervised **only on wet target
    pixels** in log space.  Dry pixels therefore no longer dominate the
    regression, which is the mechanism E41 S1 tests.

    Args:
        rain_threshold: mm defining a wet pixel (matches ``PrecipitationLoss``).
        occ_weight: weight of the Bernoulli term.
        amount_weight: weight of the conditional log-amount term.
    """

    def __init__(
        self,
        rain_threshold: float = 0.1,
        occ_weight: float = 1.0,
        amount_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.rain_threshold = float(rain_threshold)
        self.occ_weight = float(occ_weight)
        self.amount_weight = float(amount_weight)

    def forward(
        self,
        occ_logit: torch.Tensor,
        amount_mm: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target_mm = target.clamp_min(0.0)
        wet = (target_mm >= self.rain_threshold).float()

        occ_loss = F.binary_cross_entropy_with_logits(occ_logit, wet)

        pred_log = torch.log1p(_floor_at_zero(amount_mm))
        target_log = torch.log1p(target_mm)
        per_pixel = F.smooth_l1_loss(pred_log, target_log, reduction="none")
        n_wet = wet.sum().clamp_min(1.0)
        amount_loss = (per_pixel * wet).sum() / n_wet

        total = self.occ_weight * occ_loss + self.amount_weight * amount_loss
        components = {
            "occ_bce": float(occ_loss.detach()),
            "amount_nll": float(amount_loss.detach()),
            "wet_frac": float(wet.mean().detach()),
            "total": float(total.detach()),
        }
        return total, components


class HurdleCriterion:
    """Adapter from the training loop to :class:`HurdleNLL`.

    ``scripts/train.py`` passes ``model=`` into every criterion, so the
    occurrence logit and conditional amount are read from the model's cached
    forward pass.  A model without the hurdle head fails loudly rather than
    silently degrading to a plain regression.
    """

    def __init__(
        self,
        rain_threshold: float = 0.1,
        occ_weight: float = 1.0,
        amount_weight: float = 1.0,
    ) -> None:
        self.loss = HurdleNLL(rain_threshold, occ_weight, amount_weight)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module | None = None,
        **_extra: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        occ = getattr(model, "last_occ_logit", None) if model is not None else None
        amount = getattr(model, "last_amount", None) if model is not None else None
        if occ is None or amount is None:
            raise RuntimeError(
                "--loss hurdle requires a model with head_mode='hurdle' "
                "(it must publish last_occ_logit and last_amount)"
            )
        return self.loss(occ, amount, target)


class UnbalancedSinkhornDivergence(nn.Module):
    """Entropy-regularised unbalanced OT between coarse predicted/truth mass.

    A KL relaxation on both marginals (``rho = tau / (tau + eps)``) lets the
    transport plan ignore mass that has no counterpart.  That is the right
    geometry when the forecast under-reports area (BIAS 0.035-0.435) and the
    support is displaced (r = +0.17): a balanced OT would be forced to move
    mass that simply is not there.

    Everything is done on a ``grid x grid`` coarse histogram (default 16x16),
    so cost is O(T * G^4) with G = 16 instead of 301x301.  No external
    dependency: log-domain Sinkhorn scaling, pure PyTorch.

    Args:
        grid: coarse grid side length.
        eps: entropic regularisation on a cost matrix normalised to [0, 1].
        tau: marginal-relaxation strength (larger = closer to balanced OT).
        n_iters: Sinkhorn scaling iterations.
        mass_weight: weight of the explicit total-mass gap term.
    """

    def __init__(
        self,
        grid: int = 16,
        eps: float = 0.05,
        tau: float = 1.0,
        n_iters: int = 30,
        mass_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.eps = float(eps)
        self.tau = float(tau)
        self.n_iters = int(n_iters)
        self.mass_weight = float(mass_weight)

        g = self.grid
        axis = torch.linspace(0.0, 1.0, g)
        try:
            ys, xs = torch.meshgrid(axis, axis, indexing="ij")
        except TypeError:  # torch < 1.10 has no indexing argument
            ys, xs = torch.meshgrid(axis, axis)
        coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)
        cost = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        cost = cost / cost.max().clamp_min(1e-8)
        self.register_buffer("cost", cost)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        p = _coarse_mass(_floor_at_zero(pred), self.grid)
        q = _coarse_mass(target.clamp_min(0.0), self.grid)

        # E42 (2026-11-09): how the masses are normalised.
        #
        # The original code divided by EACH SAMPLE's own truth total:
        #
        #     scale = q.sum(dim=1, keepdim=True).detach().clamp_min(1e-6)
        #     a = (p + 1e-6) / scale
        #     b = (q + 1e-6) / scale
        #
        # and measured the mass mismatch as ``|a.sum() - b.sum()|``.  That
        # put an ABSOLUTE amount of rain into a ratio: the difference
        # became ``|predicted mass - truth mass| / truth total``, so a
        # nearly dry window multiplied the term by 1/total.  The A2 arm
        # of E41's first batch measured the consequence exactly: with the
        # truth total at 1e-5 the normalised masses reached 5e5, the
        # transport cost reached 1e8, and 23 % of the training steps
        # carried a loss above 100.  The divergence is in the SCALE, not
        # in the mechanism: an under-reporting forecast at a constant
        # ratio was stable at 0.103 across that whole range.
        #
        # The fix keeps the geometry untouched and normalises the two
        # quantities separately:
        #
        #   * a and b are each normalised to a SHAPE (total mass one), so
        #     the transport term compares WHERE the rain is and is
        #     completely decoupled from how much fell;
        #   * the mass mismatch is carried by the LOG RATIO of the two
        #     totals, which is dimensionless, bounded, and says exactly
        #     how many times the forecast under-reports.
        #
        # Nothing else changes: grid, eps, tau, n_iters, the cost matrix,
        # rho = tau / (tau + eps), the logsumexp iteration, mass_weight
        # and uot_weight are all as they were.  No new information enters
        # the model (only the current batch's prediction and truth, as
        # before) and no new dependency is needed.  This is a fix to the
        # implementation, not a change of mechanism.
        p_tot = p.sum(dim=1, keepdim=True)
        q_tot = q.sum(dim=1, keepdim=True)
        a = p / p_tot.clamp_min(1e-12)
        b = q / q_tot.clamp_min(1e-12)

        log_a = a.clamp_min(1e-12).log()
        log_b = b.clamp_min(1e-12).log()
        # ``cost`` is a registered buffer built on CPU.  scripts/train.py
        # never moves the criterion to the GPU, while ``pred`` lives on
        # cuda:0, so the buffer has to follow the prediction's device here;
        # otherwise the logsumexp below mixes cuda:0 and cpu tensors.
        cost_mat = self.cost.to(device=p.device, dtype=p.dtype)
        log_k = -cost_mat / self.eps                       # (M, M)
        rho = self.tau / (self.tau + self.eps)

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.n_iters):
            # (K v)[n, i] = sum_j K[i, j] v[n, j]
            kv = torch.logsumexp(log_k.unsqueeze(0) + log_v.unsqueeze(1), dim=2)
            log_u = rho * (log_a - kv)
            # (K^T u)[n, j] = sum_i K[i, j] u[n, i]
            ktu = torch.logsumexp(log_k.unsqueeze(0) + log_u.unsqueeze(2), dim=1)
            log_v = rho * (log_b - ktu)

        log_plan = log_u.unsqueeze(2) + log_v.unsqueeze(1) + log_k.unsqueeze(0)
        cost = (log_plan.exp() * cost_mat.unsqueeze(0)).sum(
            dim=(1, 2)
        ).mean()
        # The mass mismatch, as a log ratio of the two totals.  With a and
        # b normalised to shapes, their sums are one, so the absolute
        # difference of the sums carries no information; the ratio of the
        # totals is what says how many times the forecast under-reports,
        # and on a log scale it is bounded and dimensionless.
        mass_gap = (
            ((p_tot + 1e-12) / (q_tot + 1e-12)).squeeze(1).log().abs().mean()
        )
        value = cost + self.mass_weight * mass_gap
        return value, {
            "sinkhorn_cost": float(cost.detach()),
            "mass_gap": float(mass_gap.detach()),
        }


class UOTCriterion:
    """Standard regression + one extra term: the unbalanced OT divergence.

    The only difference from the ``standard`` arm is the added loss geometry.
    Architecture, output head, data and evaluator are identical, which is
    what makes E41 A2 a single-variable ablation.
    """

    def __init__(
        self,
        uot_weight: float = 1.0,
        grid: int = 16,
        eps: float = 0.05,
        tau: float = 1.0,
        n_iters: int = 30,
        mass_weight: float = 0.1,
        physics_weight: float = 0.0,
    ) -> None:
        self.base = PrecipitationLoss(physics_weight=physics_weight)
        self.uot = UnbalancedSinkhornDivergence(
            grid=grid, eps=eps, tau=tau, n_iters=n_iters,
            mass_weight=mass_weight,
        )
        self.uot_weight = float(uot_weight)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module | None = None,
        **extra: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        base, components = self.base(pred, target, model=model, **extra)
        uot_term, uot_parts = self.uot(pred, target)
        value = base + self.uot_weight * uot_term
        out = dict(components)
        out["uot_loss"] = float(uot_term.detach())
        out["uot_cost"] = uot_parts["sinkhorn_cost"]
        out["mass_gap"] = uot_parts["mass_gap"]
        out["total"] = float(value.detach())
        return value, out
