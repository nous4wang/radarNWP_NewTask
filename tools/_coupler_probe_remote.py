"""Measure what the quantum-inspired coupler actually does to the features.

A coupling that runs, keeps the shape and receives gradients can still be
doing nothing useful. These are the measurements that say whether it is
doing something:

* how much does it change the features, relative to their own scale;
* does the change depend on the NWP, or is it the same for every input;
* is the modulation bounded (a FiLM with an unbounded shift can run away);
* does it stay finite when the NWP is extreme.
"""
import sys

import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.models.common import QuantumInspiredTendencyCoupler  # noqa: E402

FEAT = 32
SPATIAL = 16
NWP_VARS = 27
OUT_LEN = 10

torch.manual_seed(0)
coupler = QuantumInspiredTendencyCoupler(
    in_nwp_c=NWP_VARS, radar_feat_dim=FEAT, out_len=OUT_LEN
)
coupler.eval()

features = torch.randn(4, OUT_LEN, FEAT, SPATIAL, SPATIAL)
nwp_a = torch.randn(4, 2, NWP_VARS, SPATIAL, SPATIAL)
nwp_b = torch.randn(4, 2, NWP_VARS, SPATIAL, SPATIAL)

with torch.no_grad():
    out_a = coupler(features, nwp_a)
    out_b = coupler(features, nwp_b)

scale = features.abs().mean()
change = (out_a - features).abs().mean()
print(f"feature scale (mean |x|)      : {scale:.4f}")
print(f"mean |modulation|             : {change:.4f}")
print(f"relative modulation           : {float(change / scale):.4f}")

# Does the modulation depend on the NWP, or is it a fixed offset?
difference = (out_a - out_b).abs().mean()
print(f"mean |out(a) - out(b)|        : {float(difference):.4f}")
print("  (if this were ~0, the coupler would ignore the NWP)")

# Is the modulation bounded?
print(f"max |out|                    : {float(out_a.abs().max()):.4f}")
print(f"out is finite                 : {bool(torch.isfinite(out_a).all())}")

# Extreme NWP: the coupling must not produce NaN or an explosion.
for factor in (10.0, 100.0):
    with torch.no_grad():
        extreme = coupler(features, nwp_a * factor)
    ratio = float(extreme.abs().max() / max(float(features.abs().max()), 1e-6))
    print(f"NWP x{factor:<6g}: max|out| = {float(extreme.abs().max()):.4f} "
          f"(ratio to the features: {ratio:.3f}), finite="
          f"{bool(torch.isfinite(extreme).all())}")

# The zero-NWP pass-through must be exact.
with torch.no_grad():
    blank = coupler(features, torch.zeros_like(nwp_a))
print(f"blanked NWP is an exact pass-through: "
      f"{bool(torch.equal(blank, features))}")
