"""Why did the hierarchical loss not do what the soft-CSI loss did?

The scan's verdict is negative: the hierarchical loss raises the
frequency bias a little over the backbone (0.145 against 0.124) but it
loses on the score (CSI@0.5 0.061 against 0.066, and CSI@0.1 0.296
against 0.500). The soft-CSI loss, on the same architecture, reaches a
bias of 0.211 and a CSI@0.5 of 0.078.

The question is why, and it is worth answering with numbers rather than
speculation. Three candidates:

  1. the probability term is too weak relative to the regression, so it
     barely moves the model;
  2. it pushes in the wrong direction -- the nesting means that raising
     the heavy probability raises the light probability too, so the
     light threshold gets most of the false alarms;
  3. the loss's own gradient is dominated by the dry pixels, because
     they are the overwhelming majority and every one of them
     contributes a false-alarm term.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss


def main() -> None:
    print("=" * 78)
    print("1. the loss's own decomposition, at the two operating points")
    print("=" * 78)

    # The real class balance, measured from the archive's label.
    # At 0.5 mm per 6-minute frame, 3.04 % of the pixels are wet and
    # 0.54 % reach 2 mm.
    prevalence = {0.1: 0.1012, 0.5: 0.0304, 1.0: 0.0131, 2.0: 0.0054,
                  4.0: 0.0019}

    loss_fn = HierarchicalPrecipLoss(alpha_prob=0.3)
    print(f"  {'threshold':>10s}{'prevalence':>12s}{'w_miss':>8s}"
          f"{'w_fa':>7s}{'miss term':>12s}{'FA term':>12s}{'ratio':>9s}")
    print("  " + "-" * 68)
    for k, th in enumerate(loss_fn.thresholds):
        p = prevalence[th]
        w_miss, w_fa = loss_fn.tier_weights[k]
        # The two terms per pixel, in units of the binary cross-entropy.
        miss_term = w_miss * p * (-np.log(0.5))
        fa_term = w_fa * (1 - p) * (-np.log(0.5))
        ratio = miss_term / fa_term if fa_term > 0 else float("inf")
        print(f"  {th:9.1f}mm{p:12.4f}{w_miss:8.1f}{w_fa:7.1f}"
              f"{miss_term:12.5f}{fa_term:12.5f}{ratio:9.4f}")

    print()
    print("  The false-alarm term dominates at every threshold, because the")
    print("  dry pixels are the overwhelming majority. At 0.5 mm the")
    print("  false-alarm weight is 2.0 against a prevalence of 3 %, so the")
    print("  term is roughly 65 times larger than the miss term.")
    print()
    print("  That is the intended protection of the light rain -- but it")
    print("  means the loss is overwhelmingly a statement about the dry")
    print("  pixels, and the heavy rain is a rounding error in it.")

    print()
    print("=" * 78)
    print("2. where the gradient actually goes")
    print("=" * 78)

    # A pixel that is dry, and a pixel that is heavy rain.
    for name, target in (("dry pixel", 0.0), ("heavy pixel", 5.0)):
        total = 0.0
        parts = []
        for k, th in enumerate(loss_fn.thresholds):
            w_miss, w_fa = loss_fn.tier_weights[k]
            y = 1.0 if target >= th else 0.0
            # A model that is unsure: p = 0.5 everywhere.
            p = 0.5
            loss_miss = -w_miss * y * np.log(p)
            loss_fa = -w_fa * (1 - y) * np.log(1 - p)
            parts.append((th, loss_miss + loss_fa))
            total += loss_miss + loss_fa
        print(f"  a {name}:")
        for th, v in parts:
            if abs(v) > 1e-9:
                print(f"    threshold {th:4.1f} mm contributes {v:8.4f}")
        print(f"    the sum over the thresholds: {total:8.4f}")
        print()

    print("  A dry pixel and a heavy pixel cost the same order of")
    print("  magnitude, and the dry pixel is 97 % of the field. So the")
    print("  loss spends almost all of its gradient on telling the model")
    print("  that it should not predict rain, which is the opposite of")
    print("  what the heavy-rain tiers are asking for.")

    print()
    print("=" * 78)
    print("3. what the nesting does to the false alarms")
    print("=" * 78)

    # The exceedance probabilities are nested, so P(>=0.1) = 1 - probs[0]
    # and moving mass out of the dry bin into the heavy bin raises the
    # light probability at the same pixel.
    bins = torch.tensor([0.98, 0.01, 0.005, 0.003, 0.002, 0.0])

    def cum(b):
        return torch.flip(torch.cumsum(torch.flip(b, dims=[0]), dim=0), dims=[0])

    before = cum(bins)
    moved = bins.clone()
    moved[5] = 0.30
    moved[0] = 0.68
    after = cum(moved)

    print(f"  before: P(>=0.1) = {float(before[1]):.4f}, "
          f"P(>=4.0) = {float(before[5]):.4f}")
    print(f"  after:  P(>=0.1) = {float(after[1]):.4f}, "
          f"P(>=4.0) = {float(after[5]):.4f}")
    print(f"  the light-rain probability rose by {float(after[1] - before[1]):+.4f}"
          f" while the heavy one rose by {float(after[5] - before[5]):+.4f}")
    print()
    print("  The heavy probability can only be raised by moving mass out of")
    print("  the dry bin, and the light probability is what the dry bin")
    print("  leaves behind. So raising the heavy rain necessarily raises")
    print("  the light-rain probability at the same pixel -- which is")
    print("  exactly the false alarm the 0.1 mm tier is meant to punish.")
    print()
    print("  The nesting that makes the probabilities self-consistent is")
    print("  the same property that couples the tiers against each other.")

    print()
    print("=" * 78)
    print("4. why the soft-CSI loss did better")
    print("=" * 78)
    print("  The soft-CSI term computes the counts over the whole field and")
    print("  takes their ratio, so it is a statement about the field as a")
    print("  whole rather than about each pixel. It does not have to")
    print("  choose between the tiers: a pixel that is predicted wet and")
    print("  is wet raises the hit count, and a pixel that is predicted")
    print("  wet and is dry raises the false-alarm count, and the ratio")
    print("  they form is exactly the score. The gradient it produces is")
    print("  therefore already aligned with the quantity being scored.")
    print()
    print("  The hierarchical loss makes the same trade but per pixel, and")
    print("  a per-pixel trade on a field that is 97 % dry is a trade")
    print("  about the dry pixels.")


if __name__ == "__main__":
    main()
