"""Does the projector keep the spatial structure of a real NWP field?

The random-input probe could only say that the coupling does not blow up.
What matters is whether a real field -- a smooth, physically structured
one -- comes out with its structure intact, because that structure is
what the tendency is supposed to describe.

The measurement: the correlation between neighbouring pixels of the
projected state. A real NWP field is smooth, so the projection should not
destroy that smoothness. The reference is the correlation of the input
field itself.
"""
import sys

import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.data import RadarNWPDataset  # noqa: E402
from radar_nwp_bench.models.common import QuantumStateProjector  # noqa: E402

DATA = "/home/wf/data/radarNWP/BJ_mmap"

dataset = RadarNWPDataset(
    DATA, packed=True, in_len=11, out_len=10, patch_size=64,
    split="test", max_samples=8, random_crop=False,
)
item = dataset[0]
x_nwp = item["x_nwp"]            # (2, 27, 64, 64), normalised
print(f"NWP 输入: shape={tuple(x_nwp.shape)} std={float(x_nwp.std()):.4f}")


def neighbour_correlation(field: torch.Tensor) -> float:
    """Mean correlation between horizontally adjacent pixels."""
    a = field[..., :, :-1].reshape(-1)
    b = field[..., :, 1:].reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)


print(f"输入场的相邻像素相关: {neighbour_correlation(x_nwp):.4f}")
print("  (NWP 是物理量，平滑场应该有极高的相关)")

torch.manual_seed(0)
projector = QuantumStateProjector(27)
projector.eval()
with torch.no_grad():
    psi = projector(x_nwp[0])
print(f"InstanceNorm2d (当前): 投影后相邻像素相关 = "
      f"{neighbour_correlation(psi):.4f}")

# The GroupNorm the projector started with, for comparison. It normalises
# over the channel axis at each pixel, which is the suspect.
projector.norm = torch.nn.GroupNorm(1, psi.shape[1], affine=True)
with torch.no_grad():
    psi_g = projector(x_nwp[0])
print(f"GroupNorm(1) (原版):   投影后相邻像素相关 = "
      f"{neighbour_correlation(psi_g):.4f}")

# No norm at all: the reference for what the field looks like coming in.
projector.norm = torch.nn.Identity()
with torch.no_grad():
    psi_n = projector(x_nwp[0])
print(f"无归一化 (参照):        投影后相邻像素相关 = "
      f"{neighbour_correlation(psi_n):.4f}")
