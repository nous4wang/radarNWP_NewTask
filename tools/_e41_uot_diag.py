"""E41 diagnostic: why does the A2 (uot) arm's training loss explode?

READ-ONLY diagnosis.  No training, no evaluation, no checkpoint written.

The A2 arm's per-step loss reaches 3.4e8 while the other three arms stay
around 0.1-0.2, and the A2 checkpoints' largest weight magnitude is 1624
against 19.6 for B0.  The hypothesis under test here is that the
unbalanced Sinkhorn term divides by the TRUTH total mass of a coarse
cell, so a window whose truth is nearly dry produces a huge normalised
mass and a huge transport cost.

The script prints, for a range of truth totals, the value of the UOT
term and the normalised masses that feed it.  It is a diagnostic only:
it writes no file and changes no code.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.metrics.struct_losses import (  # noqa: E402
    UnbalancedSinkhornDivergence,
    _coarse_mass,
)


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uot = UnbalancedSinkhornDivergence(grid=16, eps=0.05, tau=1.0, n_iters=30)
    torch.manual_seed(0)
    print(f"device={dev}")
    print("truth_total | uot_value | sinkhorn_cost | mass_gap | a.sum | b.sum")
    for target_total in [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5]:
        pred = torch.rand(1, 10, 1, 64, 64, device=dev) * 0.5
        tgt = torch.rand(1, 10, 1, 64, 64, device=dev) * target_total
        val, parts = uot(pred, tgt)
        p = _coarse_mass(torch.clamp(pred, min=0.0), 16)
        q = _coarse_mass(tgt, 16)
        scale = q.sum(dim=1, keepdim=True).clamp_min(1e-6)
        a = (p + 1e-6) / scale
        b = (q + 1e-6) / scale
        print(
            f"{target_total:11.0e} | {float(val):10.4g} | "
            f"{parts['sinkhorn_cost']:13.4g} | {parts['mass_gap']:9.4g} | "
            f"{float(a.sum()):5.4g} | {float(b.sum()):5.4g}"
        )

    # The same question with a realistic under-reporting forecast: the
    # forecast has less mass than the truth, which is what the model
    # actually does (BIAS 0.035-0.435).
    print("\nunder-reporting forecast (0.2x the truth mass):")
    for target_total in [1.0, 0.01, 1e-3, 1e-4]:
        tgt = torch.rand(1, 10, 1, 64, 64, device=dev) * target_total
        pred = tgt * 0.2
        val, parts = uot(pred, tgt)
        p = _coarse_mass(torch.clamp(pred, min=0.0), 16)
        q = _coarse_mass(tgt, 16)
        scale = q.sum(dim=1, keepdim=True).clamp_min(1e-6)
        a = (p + 1e-6) / scale
        b = (q + 1e-6) / scale
        print(
            f"{target_total:11.0e} | {float(val):10.4g} | "
            f"{parts['sinkhorn_cost']:13.4g} | {parts['mass_gap']:9.4g} | "
            f"{float(a.sum()):5.4g} | {float(b.sum()):5.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
