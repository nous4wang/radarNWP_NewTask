"""Three checks on the proposal that decide whether it can work as argued.

1. The indexing of the exceedance probabilities. The head computes them
   as the tail sums of a softmax over five bins, and the loss then
   supervises ``cum_probs[k]`` against ``target >= thresholds[k]``. Whether
   those two line up is a matter of arithmetic, and it decides what the
   loss actually supervises.

2. The geometric argument. The proposal says that because the
   exceedance probabilities are nested, raising the heavy rain must
   happen inside the existing light-rain outline, so it cannot create
   false alarms at the light threshold. That is a claim about what
   happens to the light-rain probability when the heavy bin is raised,
   and it can be measured.

3. The expected numbers. The proposal expects the false alarms to be
   compressed from 4.3 million to under 2.0 million *and* the frequency
   bias to settle between 0.5 and 0.8. Those two are arithmetic claims
   about the same contingency table, and they may not be compatible.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F

from radar_nwp_bench.models.common import QuantumHierarchicalRainHead


def check_indexing() -> None:
    print("=" * 78)
    print("1. what the loss actually supervises")
    print("=" * 78)

    head = QuantumHierarchicalRainHead(in_channels=4, hidden_dim=16)
    feat = torch.randn(2, 4, 8, 8)
    _, cum = head(feat)

    print("  the head's bin floors are [0.0] + thresholds[:-1], so the")
    print("  five bins are the intervals:")
    thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)
    floors = [0.0] + list(thresholds[:-1])
    for k in range(5):
        upper = thresholds[k] if k < len(thresholds) else "inf"
        print(f"    bin {k}: [{floors[k]}, {upper}) mm")

    print()
    print("  the tail sum cum_probs[k] = sum(probs[j] for j >= k) is")
    print("  therefore the probability of the *bin floor*, and:")
    for k in range(5):
        print(f"    cum_probs[{k}] = P(R >= {floors[k]:.1f} mm)"
              f"   -- the loss supervises it as P(R >= {thresholds[k]:.1f} mm)")

    print()
    print("  two consequences:")
    print(f"    cum_probs[0] = P(R >= 0.0) = 1 identically"
          f"  (measured: {float(cum[:, 0].min()):.6f})")
    print("    so the 0.1 mm tier supervises a quantity that cannot move:")
    print("    its clamp kills the gradient, and the tier contributes a")
    print("    constant to the loss instead of a gradient.")

    print()
    print("    and P(R >= 4.0 mm) has no representation at all: the top")
    print("    bin's floor is 2.0 mm, so cum_probs[4] = P(R >= 2.0 mm).")
    print("    supervising it against (target >= 4.0) asks the head to")
    print("    mean '2 mm or more' by '4 mm or more'.")

    # Measure the constant the dead tier contributes.
    print()
    print("  the magnitude of the dead tier, measured:")
    p0 = cum[:, 0].clamp(1e-6, 1.0 - 1e-6)
    dry = torch.zeros_like(p0)
    wet = torch.ones_like(p0)
    loss_fa_dry = -3.0 * (1.0 - dry) * torch.log(1.0 - p0)
    loss_miss_wet = -1.0 * wet * torch.log(p0)
    print(f"    on a dry pixel: {float(loss_fa_dry.mean()):.4f}"
          f"   (a constant, gradient is zero)")
    print(f"    on a wet pixel: {float(loss_miss_wet.mean()):.6f}")


def check_geometry() -> None:
    print()
    print("=" * 78)
    print("2. does raising the heavy bin create light-rain false alarms?")
    print("=" * 78)

    # Two pixels: one the model is sure is dry, one where it is unsure.
    # Raise the heavy bin at the dry pixel and watch the light-rain
    # probability there.
    bins = torch.tensor([0.98, 0.01, 0.005, 0.003, 0.002])
    cum = torch.flip(torch.cumsum(torch.flip(bins, dims=[0]), dim=0), dims=[0])
    print(f"  before: the pixel's P(>=0.1) = {float(cum[1]):.4f}, "
          f"P(>=4.0) = {float(cum[4]):.4f}")

    # Move 30 % of the mass from the dry bin into the heavy bin.
    moved = bins.clone()
    moved[4] += 0.30 * moved[0]
    moved[0] -= 0.30 * moved[0]
    cum2 = torch.flip(torch.cumsum(torch.flip(moved, dims=[0]), dim=0), dims=[0])
    print(f"  after:  the pixel's P(>=0.1) = {float(cum2[1]):.4f}, "
          f"P(>=4.0) = {float(cum2[4]):.4f}")
    print(f"          the light-rain probability ROSE by "
          f"{float(cum2[1] - cum[1]):+.4f}")

    print()
    print("  this is the opposite of the proposal's argument. The nested")
    print("  structure means P(>=0.1) = 1 - probs[0]: the light-rain")
    print("  probability is what is LEFT after the dry bin. So moving")
    print("  mass out of the dry bin and into the heavy bin raises the")
    print("  light-rain probability at that pixel as well.")
    print()
    print("  the nesting says the heavy rain cannot exceed the light")
    print("  rain. It does not say the light rain stays put.")

    # Measure it over a whole field.
    print()
    print("  over a field of 1000 dry pixels, raising the heavy bin at")
    print("  each of them:")
    n = 1000
    field = torch.zeros(5, n)
    field[0] = 1.0
    cum_before = torch.flip(
        torch.cumsum(torch.flip(field, dims=[0]), dim=0), dims=[0]
    )
    field2 = field.clone()
    field2[4] = 0.3
    field2[0] = 0.7
    cum_after = torch.flip(
        torch.cumsum(torch.flip(field2, dims=[0]), dim=0), dims=[0]
    )
    added = float((cum_after[1] - cum_before[1]).sum())
    print(f"    pixels that now exceed 0.1 mm: +{added:.0f}")
    print(f"    of which they exceed 4.0 mm:   +{float((cum_after[4] - cum_before[4]).sum()):.0f}")
    print(f"    so {added:.0f} light-threshold false alarms were created"
          f" to buy the heavy rain")


def check_expectations() -> None:
    print()
    print("=" * 78)
    print("3. are the expected numbers compatible?")
    print("=" * 78)

    # The measured contingency table of the best deep model in the
    # matrix and of the balanced-loss run.
    cases = {
        "simvp_v2 (backbone)": (2_429_711, 1_905_736, 32_633_831),
        "hqma_balanced": (3_072_709, 4_332_760, 31_990_833),
    }
    for name, (hits, fa, misses) in cases.items():
        bias = (hits + fa) / (hits + misses)
        csi = hits / (hits + fa + misses)
        print(f"  {name}: hits={hits:,} FA={fa:,} misses={misses:,}")
        print(f"    BIAS@0.5 = {bias:.3f}   CSI@0.5 = {csi:.3f}")

    print()
    print("  the proposal expects the false alarms to fall from 4.3 M to")
    print("  under 2.0 M AND the bias to settle between 0.5 and 0.8.")
    print("  Those are two constraints on one contingency table:")
    print()
    hits, fa, misses = cases["hqma_balanced"]
    print(f"  BIAS = (hits + FA) / (hits + misses)")

    print()
    print("  a) holding the hits and misses, what FA gives BIAS = 0.5?")
    target_bias = 0.5
    needed_fa = target_bias * (hits + misses) - hits
    print(f"     FA = {needed_fa:,.0f}  ({needed_fa / fa:.1f}x the current one)")
    print(f"     i.e. the false alarms have to go UP, not down.")

    print()
    print("  b) holding FA at 2.0 M, what hits give BIAS = 0.5?")
    target_fa = 2_000_000
    needed_hits = (0.5 * (hits + misses) - target_fa) / 0.5
    print(f"     hits = {needed_hits:,.0f}  ({needed_hits / hits:.1f}x the current one)")

    print()
    print("  c) what does FA = 2.0 M actually buy, at the current hits?")
    csi_new = hits / (hits + target_fa + misses)
    bias_new = (hits + target_fa) / (hits + misses)
    print(f"     CSI@0.5 = {csi_new:.3f} (now {csi_new - 0.078:+.3f})")
    print(f"     BIAS@0.5 = {bias_new:.3f} (now {bias_new - 0.211:+.3f})")
    print()
    print("  so the two expectations cannot hold at once unless the hit")
    print("  count rises by about nine times, which no change to the loss")
    print("  weighting will do on its own -- it would need the model to")
    print("  find nine times more heavy rain correctly.")

    print()
    print("  for reference, the physical baseline's numbers:")
    print("    hits=11,916,868  FA=20,291,040  misses=23,146,674")
    h, f, m = 11_916_868, 20_291_040, 23_146_674
    print(f"    BIAS={((h + f) / (h + m)):.3f}  CSI={h / (h + f + m):.3f}")
    print("    it reaches BIAS 0.90 WITH a high false-alarm rate, so the")
    print("    trade is real and the baseline makes it in the other")
    print("    direction: more rain predicted, more of it wrong, but the")
    print("    heavy rain is found.")


def main() -> None:
    check_indexing()
    check_geometry()
    check_expectations()


if __name__ == "__main__":
    main()
