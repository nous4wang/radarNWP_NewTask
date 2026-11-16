"""E44 / MASTER 4.46 -- A3: the topology branch (a clDice connectivity term).

This module is deliberately NOT a new architecture.  Its backbone and its
decoder are ``simvp_v2``'s, parameter for parameter, so that the S0
gate checks a bit equality against ``simvp_v2`` and the arm differs from
the control in exactly one thing: the training objective carries one
extra term.

The term is the soft clDice of Sit et al. (MICCAI 2021), which scores the
agreement between the connected components of the forecast and of the
truth at a threshold, with the soft Dice used to approximate the
intersection.  It is written for the strong-rain threshold (2.0 mm),
because the label's connectivity is what distinguishes it from the
radar's: C0 measured the connected kernel of the label to be 3.37 times
the radar's, and the section 2 probe measured it at 80.8 pixels against
the forecast's 72.1.

What is NOT changed: the backbone, the decoder, the output contract, the
weights of the main regression term.  There is exactly one added term
with one weight, and the weight is fixed in advance.

Why this is not inside a family that was already closed: E42 closed the
optimal-transport class of loss geometry (UOT, Sinkhorn variants,
Wasserstein variants, contrastive and frequency losses).  A topology term
is not a metric-geometry term.  E34 closed the "topology constraint
MODULE" derived from E33, whose premise was that a number measured with
ground-truth substitution could be extrapolated into a trainable module;
this term is computed on the forecast field itself, and its premise is
C0's positive kernel difference.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero


@register_model("topo_simvp")
class TopologySimVPModel(BasePrecipitationModel):
    """``simvp_v2`` unchanged; the mechanism lives in the loss.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: channel width.
        num_blocks: translator depth.
        clamp_mm: optional upper bound (mm) on the OUTPUT rain field.
            ``0.0`` (default) leaves the output bit-identical to
            ``simvp_v2``; a positive value clamps the millimetre field
            *after* the expm1/floor transform.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        clamp_mm: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.clamp_mm = float(clamp_mm)

        total_in = in_radar_c + in_nwp_c * 2

        # --- identical to simvp_v2, module for module ---
        self.enc = nn.Sequential(
            nn.Conv2d(total_in, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hid_c * 2),
            nn.GELU(),
        )

        self.in_time_proj = nn.Conv2d(in_len * hid_c * 2, hid_c * 4, kernel_size=1)
        self.translator = nn.Sequential(
            *[ConvNeXtBlock(hid_c * 4) for _ in range(num_blocks)]
        )
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, kernel_size=1)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, kernel_size=1),
        )

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, _, H, W = x_radar.shape
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        feats = self.enc(x_concat.view(B * T_in, -1, H, W))
        _, C_f, H_f, W_f = feats.shape

        feats_time = feats.view(B, T_in * C_f, H_f, W_f)
        h = self.in_time_proj(feats_time)
        h = self.translator(h)
        future_feats = self.out_time_proj(h).view(B * self.out_len, C_f, H_f, W_f)

        out = self.dec(future_feats)
        log_mm = out.view(B, self.out_len, 1, H, W)
        mm = floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
        if self.clamp_mm > 0.0:
            mm = mm.clamp(max=self.clamp_mm)
        return mm
