"""A neural QPE inverter with no temporal extrapolation whatsoever.

The Oracle ceiling (MASTER §4.15.2) says that if the ground rainfall field
were known perfectly at the start of the forecast, simply copying it for
ten frames scores CSI@2.0 = 0.1653 -- 1.79 times the persistence floor of
0.0924.  Every route that tries to *move* that field has come back negative
(§4.15.3, §4.16.2).  The remaining question is therefore narrow: can a
learned inversion of the t0 radar and NWP fields approach the Oracle field
better than the analytical Z-R relation does?

This model answers exactly that and nothing else.  It consumes ONE radar
frame (the last observed one, ten channels) and ONE NWP slice (27 channels),
produces ONE ground-rainfall field at t0, and copies that field across the
ten forecast frames.  There is no recurrent decoder, no flow, no motion
term and no supervision on any future frame.  The only learning signal is
the t0 ground rainfall, under a score-aware objective.

The copy is deliberate, not a shortcut: the benchmark seam is fixed at
``model(x_radar, x_nwp) -> (B, T_out, 1, H, W)`` (§16), so the plain
``scripts/evaluate.py`` can score this model with no special case, on the
same full-grid Hann stitching and the same verifier as every other arm.

Design notes pinned down by earlier measurements:
  * GroupNorm, not InstanceNorm2d: the latter destroys spatial structure
    (§13: adjacent-pixel correlation 0.8662 vs 0.9907 for GroupNorm(1)).
  * ``floor_at_zero`` straight-through floor at the output, never a hard
    ``nn.ReLU()`` (§12).
  * Plain channel concat for the NWP: the simplest injection measured best
    (§31).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import floor_at_zero


class _ResidualBlock(nn.Module):
    """A small conv block that keeps the grid resolution unchanged."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.body(x))


@register_model("physical_qpe")
class PhysicalQPEModel(BasePrecipitationModel):
    """Invert the t0 radar and NWP fields into a t0 ground-rainfall field.

    Args:
        in_radar_c: radar channels in one frame.
        in_nwp_c: NWP channels in one slice.
        in_len: length of the radar history (only its last frame is used).
        out_len: number of forecast frames to emit by copying.
        hid_c: latent width.
        num_blocks: number of residual blocks.
        nwp_slice: ``"analysis"`` for the t0 NWP slice (``x_nwp[:, 0]``)
            or ``"forecast"`` for the T+60 slice (``x_nwp[:, -1]``).
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        nwp_slice: str = "analysis",
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.hid_c = hid_c
        self.nwp_slice = nwp_slice
        if nwp_slice not in ("analysis", "forecast"):
            raise ValueError("nwp_slice must be 'analysis' or 'forecast'")

        total_in = in_radar_c + in_nwp_c
        groups = 8 if hid_c % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(total_in, hid_c, 3, padding=1),
            nn.GroupNorm(groups, hid_c),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hid_c) for _ in range(num_blocks)]
        )
        self.head = nn.Conv2d(hid_c, 1, 1)

    def forward(
        self, x_radar: torch.Tensor, x_nwp: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the t0 rainfall copied across ``out_len`` frames.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``. Only ``[:, -1]`` is read.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)`` or ``None`` (the seam
                resolves ``None`` to zeros before this call; the guard here
                keeps the model callable in isolation too).

        Returns:
            ``(B, T_out, 1, H, W)`` non-negative rainfall in millimetres.
        """
        radar_t0 = x_radar[:, -1]  # (B, C_radar, H, W)
        if x_nwp is None:
            nwp_t0 = radar_t0.new_zeros(
                (radar_t0.shape[0], self.in_nwp_c, *radar_t0.shape[-2:])
            )
        else:
            slice_index = 0 if self.nwp_slice == "analysis" else -1
            nwp_t0 = x_nwp[:, slice_index]

        x = torch.cat([radar_t0, nwp_t0], dim=1)
        h = self.blocks(self.stem(x))
        log_mm = self.head(h)
        # The head predicts log1p(mm); expm1 returns millimetres. The
        # straight-through floor keeps the dry pixels in the gradient path.
        r0 = floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        # (B, 1, H, W) -> (B, T_out, 1, H, W), by copying with no motion.
        return r0.unsqueeze(1).expand(-1, self.out_len, -1, -1, -1)
