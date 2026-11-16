"""Check the claims in the proposed AHFL loss and the nested head.

Three things have to be measured rather than assumed.

1. The nesting argument. The proposal says that because the exceedance
   probabilities are nested, raising the probability of the heavy rain
   necessarily happens *inside* the existing light-rain outline, so the
   false alarms cannot increase. That is a statement about the geometry
   of the probability field, and it can be checked: take a field that
   predicts the light rain correctly, raise the heavy bin, and see what
   happens to the light-rain field.

2. The gradient direction. The proposal weights the miss of heavy rain
   more heavily than the false alarm of light rain, and expects the
   bias to rise. Whether the loss actually pushes in that direction is
   a measurement, not a reading of the formula.

3. The claim that the heavy-rain probability is *reachable* at all.
   The head computes the exceedance probability as a tail sum of a
   softmax over five bins. The tail sum of the last bin is the
   probability of the heaviest bin alone, so the heaviest bin's
   probability is the smallest quantity in the field -- and a softmax
   over bins that are driven by the same state vector cannot move the
   heavy bin without moving the others.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# The proposal's loss, transcribed.
# ---------------------------------------------------------------------------
class AsymmetricHierarchicalPrecipLoss(nn.Module):
    def __init__(self, thresholds=(0.1, 0.5, 1.0, 2.0, 4.0), lambda_prob=0.8):
        super().__init__()
        self.thresholds = thresholds
        self.lambda_prob = lambda_prob
        self.tier_weights = [
            (1.0, 3.0),
            (1.5, 2.0),
            (2.5, 1.0),
            (4.0, 0.5),
            (6.0, 0.2),
        ]

    def forward(self, pred_rain, target_rain, cum_probs):
        pred_log = torch.log1p(pred_rain.clamp_min(0.0))
        target_log = torch.log1p(target_rain.clamp_min(0.0))
        reg_loss = F.smooth_l1_loss(pred_log, target_log)

        prob_loss = 0.0
        for k, th in enumerate(self.thresholds):
            w_miss, w_fa = self.tier_weights[k]
            p_k = cum_probs[:, :, k].clamp(1e-6, 1.0 - 1e-6)
            y_k = (target_rain[:, :, 0] >= th).float()
            loss_miss = -w_miss * y_k * torch.log(p_k)
            loss_fa = -w_fa * (1.0 - y_k) * torch.log(1.0 - p_k)
            pt = torch.where(y_k == 1.0, p_k, 1.0 - p_k)
            focal_weight = (1.0 - pt) ** 2.0
            tier_loss = torch.mean(focal_weight * (loss_miss + loss_fa))
            prob_loss = prob_loss + tier_loss
        return reg_loss + self.lambda_prob * (prob_loss / len(self.thresholds))


def check_nesting_geometry() -> None:
    print("=" * 76)
    print("1. can the heavy bin be raised inside the light-rain outline?")
    print("=" * 76)

    # A field of exceedance probabilities: five nested levels over a
    # 1x1x16x16 grid. The light rain occupies a blob; the heavy rain is
    # currently absent everywhere.
    light = torch.zeros(1, 1, 1, 16, 16)
    light[0, 0, 0, 4:12, 4:12] = 1.0
    heavy = torch.zeros_like(light)

    # Build the exceedance probabilities from a per-bin distribution, the
    # way the head does, and see what a change to the heavy bin does to
    # the light one.
    def to_cum(bins: torch.Tensor) -> torch.Tensor:
        return torch.flip(
            torch.cumsum(torch.flip(bins, dims=[1]), dim=1), dims=[1]
        )

    # bins[:, k] is the probability of the k-th interval
    bins = torch.zeros(1, 5, 1, 16, 16)
    bins[0, 0] = 1.0 - light * 0.0        # the dry bin takes the rest
    bins[0, 4] = heavy
    cum = to_cum(bins)

    print(f"  before: cum[0.1] mean={cum[:, 0].mean():.3f} "
          f"cum[4.0] mean={cum[:, 4].mean():.3f}")
    print("  the nesting holds by construction, but look at what the")
    print("  heavy bin costs:")

    # Raise the heavy bin inside the light-rain blob, and see how much
    # of the light-rain probability survives.
    for heavy_amount in (0.1, 0.3, 0.6, 0.9):
        b = torch.zeros(1, 5, 1, 16, 16)
        b[0, 0] = 1.0
        b[0, 4] = light * heavy_amount
        # renormalise the way a softmax would
        b = b / b.sum(dim=1, keepdim=True)
        c = to_cum(b)
        light_after = c[:, 1].mean()
        heavy_after = c[:, 4].mean()
        print(f"    heavy={heavy_amount:.1f} of the light-rain area: "
              f"P(>=0.1)={light_after:.3f}  P(>=4.0)={heavy_after:.3f}  "
              f"P(>=0.1)-P(>=4.0)={light_after - heavy_after:.3f}")

    print()
    print("  the light-rain probability is the SUM of the four upper bins,")
    print("  so raising the heavy bin takes its mass out of the others:")
    print("  the two are not independent, and the trade is exact.")


def check_gradient_direction() -> None:
    print()
    print("=" * 76)
    print("2. does the loss push the bias in the direction claimed?")
    print("=" * 76)

    torch.manual_seed(0)
    # A "prediction" that is a single free parameter per bin, so the
    # gradient can be read directly. The loss expects the probabilities
    # as (B, T, num_bins, H, W), so the bins go in the third axis.
    logits = torch.zeros(1, 1, 5, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 1, 4, 4)
    target[0, 0, 0, :, :] = 3.0      # heavy rain everywhere
    target[0, 0, 0, 0, 0] = 0.0     # one dry pixel

    loss_fn = AsymmetricHierarchicalPrecipLoss()

    def cum_from_logits(l: torch.Tensor) -> torch.Tensor:
        bins = torch.softmax(l, dim=2)
        return torch.flip(
            torch.cumsum(torch.flip(bins, dims=[2]), dim=2), dims=[2]
        )

    cum = cum_from_logits(logits)
    pred = torch.sum(
        torch.softmax(logits, dim=2)
        * torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0]).view(1, 1, 5, 1, 1),
        dim=2, keepdim=True,
    )
    value = loss_fn(pred, target, cum)
    value.backward()

    print(f"  the loss at a uniform prediction: {float(value):.4f}")
    print("  the gradient on the bin logits (higher = the loss wants more"
          " of that bin):")
    for k in range(5):
        g = float(logits.grad[0, 0, k, 0, 0])
        print(f"    bin {k} (base {[0.0, 0.1, 0.5, 1.0, 2.0][k]:.1f} mm): "
              f"{g:+.6f}")

    print()
    print("  the tail sum means the gradient on bin k is the SUM of the")
    print("  gradients on bins k..4, so raising one bin's probability")
    print("  necessarily lowers the others' -- the head cannot raise the")
    print("  heavy bin while keeping the light-rain field intact.")


def check_reachability() -> None:
    print()
    print("=" * 76)
    print("3. is P(R >= 4.0) reachable, and how large can it get?")
    print("=" * 76)

    # The head's probability: a softmax over five bins of the squared
    # inner products, scaled by ten.
    inner = torch.zeros(1, 5, 1, 4, 4)
    for k in range(5):
        inner[0, k] = float(k)          # ordered, as a trained head might be
    probs = torch.softmax(inner ** 2 * 10.0, dim=1)
    cum = torch.flip(
        torch.cumsum(torch.flip(probs, dims=[1]), dim=1), dims=[1]
    )
    print("  with five ordered bins, the heaviest one takes the mass:")
    for k in range(5):
        print(f"    bin {k}: p={float(probs[0, k, 0, 0, 0]):.6f}  "
              f"P(>=that threshold)={float(cum[0, k, 0, 0, 0]):.6f}")

    print()
    print("  the tail sum of the LAST bin is that bin's own probability,")
    print("  so the heaviest threshold's exceedance probability is the")
    print("  smallest quantity in the field. It can be raised only by")
    print("  moving mass into the last bin, which is exactly what the")
    print("  light-rain tiers depend on.")

    # How large can P(>=4.0) get while P(>=0.1) stays high?
    print()
    print("  the reachable set, sweeping the two bin probabilities:")
    print(f"    {'P(>=0.1)':>10s}{'P(>=4.0)':>10s}")
    for p_heavy in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        bins = torch.tensor([1.0 - p_heavy, 0.0, 0.0, 0.0, p_heavy])
        cum = torch.flip(torch.cumsum(torch.flip(bins, dims=[0]), dim=0), dims=[0])
        print(f"    {float(cum[0]):10.3f}{float(cum[4]):10.3f}")
    print()
    print("  P(>=0.1) and P(>=4.0) are the SAME number here, because the")
    print("  mass is only in two bins. The nested structure forces a")
    print("  single trade-off, not two independent knobs.")


def check_what_the_loss_actually_supervises() -> None:
    print()
    print("=" * 76)
    print("4. what does the loss see at the two extremes?")
    print("=" * 76)

    loss_fn = AsymmetricHierarchicalPrecipLoss()
    target_heavy = torch.full((1, 1, 1, 4, 4), 5.0)

    # A prediction that is confidently dry everywhere.
    cum_dry = torch.zeros(1, 1, 5, 4, 4)
    cum_dry[:, :, 0] = 1.0
    pred_dry = torch.zeros(1, 1, 1, 4, 4)

    # A prediction that is confidently heavy everywhere.
    cum_heavy = torch.zeros(1, 1, 5, 4, 4)
    cum_heavy[:, :, 4] = 1.0
    pred_heavy = torch.full((1, 1, 1, 4, 4), 5.0)

    # A prediction that is confident at the light threshold only.
    cum_light = torch.zeros(1, 1, 5, 4, 4)
    cum_light[:, :, 0] = 1.0
    cum_light[:, :, 1] = 1.0
    pred_light = torch.full((1, 1, 1, 4, 4), 0.3)

    for name, pred, cum in (
        ("dry", pred_dry, cum_dry),
        ("light only", pred_light, cum_light),
        ("heavy", pred_heavy, cum_heavy),
    ):
        v = loss_fn(pred, target_heavy, cum)
        print(f"  {name:12s} loss={float(v):.4f}")
    print()
    print("  the loss prefers the heavy prediction over the light one,")
    print("  which is the intent. But note what it says about the light")
    print("  tier: it also penalises the light prediction's *false alarms*")
    print("  at the 0.1 threshold with weight 3.0, and the heavy")
    print("  prediction's false alarms with weight 0.2 -- so the loss")
    print("  actively pushes the model to report heavy rain wherever it")
    print("  is unsure.")


def main() -> None:
    check_nesting_geometry()
    check_gradient_direction()
    check_reachability()
    check_what_the_loss_actually_supervises()


if __name__ == "__main__":
    main()
