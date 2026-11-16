"""Does the coupling extract *physics* from the NWP, or just noise?

The interesting question about a physical coupling is not whether it
changes the forecast -- any modulation would do that -- but whether the
change it makes is a function of the NWP's *structure*.

The test: feed the coupling the same radar features and two different
NWP fields, and see how much of the output difference is explained by
the NWP difference. A coupling that has learned to read the dynamics
should produce a systematically different forecast for a different
atmospheric state; one that has learned a constant offset would not.

Three NWP fields are compared:
  * the real field from the archive;
  * the same field with its spatial structure scrambled (the values are
    kept, the arrangement is destroyed), which breaks any spatial
    relation between the variables and the field;
  * random noise with the same per-variable statistics.

If the coupling reads the physics, the real field should separate from
the scrambled one.
"""
import sys

import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.data import RadarNWPDataset  # noqa: E402
from radar_nwp_bench.models.common import QuantumInspiredTendencyCoupler  # noqa: E402

DATA = "/home/wf/data/radarNWP/BJ_mmap"

dataset = RadarNWPDataset(
    DATA, packed=True, in_len=11, out_len=10, patch_size=64,
    split="test", max_samples=8, random_crop=False,
)
item = dataset[0]
x_nwp = item["x_nwp"].unsqueeze(0)        # (1, 2, 27, 64, 64)

torch.manual_seed(0)
coupler = QuantumInspiredTendencyCoupler(in_nwp_c=27, radar_feat_dim=64, out_len=10)
coupler.eval()

features = torch.randn(1, 10, 64, 64, 64)

# The same field with the pixels shuffled: the values are unchanged, the
# spatial arrangement is not.
generator = torch.Generator().manual_seed(1)
shuffled = x_nwp.clone()
flat = shuffled[0, 0].reshape(27, -1)
order = torch.randperm(flat.shape[-1], generator=generator)
shuffled[0, 0] = flat[:, order].reshape(27, 64, 64)

# Random noise with the same per-variable mean and standard deviation.
noise = torch.randn_like(x_nwp)
for c in range(x_nwp.shape[2]):
    mean = x_nwp[0, :, c].mean()
    std = x_nwp[0, :, c].std()
    noise[0, :, c] = (noise[0, :, c] * std + mean)

with torch.no_grad():
    out_real = coupler(features, x_nwp)
    out_scrambled = coupler(features, shuffled)
    out_noise = coupler(features, noise)

scale = float((out_real - features).abs().mean())
print(f"调制的平均幅度 (真实 NWP)      : {scale:.5f}")
print()
print("两两之间的输出差异（相对于调制幅度）:")
real_vs_scrambled = float((out_real - out_scrambled).abs().mean())
real_vs_noise = float((out_real - out_noise).abs().mean())
scrambled_vs_noise = float((out_scrambled - out_noise).abs().mean())
print(f"  真实 vs 打乱空间结构 : {real_vs_scrambled / max(scale, 1e-9):.3f}")
print(f"  真实 vs 随机噪声     : {real_vs_noise / max(scale, 1e-9):.3f}")
print(f"  打乱 vs 随机噪声     : {scrambled_vs_noise / max(scale, 1e-9):.3f}")
print()
print("  (若耦合器只学到一个常数偏置，三个数会接近 0；")
print("   若它读的是空间结构，'真实 vs 打乱' 应明显大于 0)")
