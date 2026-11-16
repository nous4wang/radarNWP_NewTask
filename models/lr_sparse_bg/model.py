"""E44 / MASTER 4.46 -- A1: the low-rank term constrains the BACKGROUND only.

E43's low-rank plus sparse arm found the conflict this module is built
to resolve.  Projecting the decoder's output map onto the eight
lowest-frequency DCT modes flattened the background exactly as intended
-- the total variation fell by 34 %, the Laplacian by 45 % and the
number of local peaks halved -- but it also shaved the amplitude tail
(tail_max_over_p99 3.2245 -> 2.9483, against a label value of 4.5475).
The label is smooth in the background AND heavy in the tail, and a single
low-rank projection cannot hold both.

This module keeps the projection and changes only WHAT IT IS APPLIED TO.
E43 composed the output as::

    out = low + mask * amp          # low = P_low(amp), one projection

Here the rain region leaves the projection entirely::

    out = low + mask * (amp - low)  # the rain keeps the residual

Mathematically that is ``P_low(amp) + mask * (I - P_low)(amp)``: the
background takes the low-rank component and the rain region takes what
the projection removed.  The background is still confined to the
eight-dimensional subspace, but the amplitude tail is no longer at the
projection's mercy.

What is NOT changed: the projection basis and rank, the gate expression
and its learned threshold, the backbone, the loss, the output contract.
The S0 gate checks that with both switches off this module is bit-equal
to ``simvp_v2``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero

from models.lr_sparse.model import _lowest_dct_modes


@register_model("lr_sparse_bg")
class LowRankBackgroundModel(BasePrecipitationModel):
    """SimVP-v2 whose low-rank projection constrains the background only.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: channel width.
        num_blocks: translator depth.
        r_low: dimension of the spatial subspace the background is
            confined to.
        gate_temp: temperature of the support gate.
        low_rank: when False the background is left unconstrained.
        sparse_gate: when False the rain branch carries no support mask.
            Both switches off reproduce ``simvp_v2`` exactly.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        r_low: int = 8,
        gate_temp: float = 1.0,
        low_rank: bool = True,
        sparse_gate: bool = True,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.r_low = int(r_low)
        self.gate_temp = float(gate_temp)
        self.use_low_rank = bool(low_rank)
        self.use_sparse_gate = bool(sparse_gate)

        total_in = in_radar_c + in_nwp_c * 2

        # --- identical to simvp_v2 (module names matter for load_state_dict) ---
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

        self.sparse_gate = nn.Conv2d(hid_c, 1, kernel_size=1)
        self.gate_threshold = nn.Conv2d(hid_c, 1, kernel_size=1)

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

        body = self.dec[:-1](future_feats)
        amp = self.dec[-1](body)                      # (B*T, 1, H, W)

        if not self.use_low_rank and not self.use_sparse_gate:
            # The S0 path: exactly simvp_v2's decoder on exactly
            # simvp_v2's features.
            log_mm = amp.view(B, self.out_len, 1, H, W)
            return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        # --- the low-rank background ---
        if self.use_low_rank:
            basis = _lowest_dct_modes(H, W, self.r_low).to(
                device=amp.device, dtype=amp.dtype
            )
            flat = amp.reshape(amp.shape[0], -1)
            coeff = flat @ basis.reshape(self.r_low, -1).T
            low = (coeff @ basis.reshape(self.r_low, -1)).reshape(
                amp.shape[0], 1, H, W
            )
            # What the projection removed is what the rain region keeps.
            residual = amp - low
        else:
            low = torch.zeros_like(amp)
            residual = amp

        # --- the sparse rain region: the region bypasses the projection ---
        if self.use_sparse_gate:
            gate = self.sparse_gate(body)
            thr = self.gate_threshold(body).mean(dim=(2, 3))
            thr = thr.view(-1, 1, 1, 1)
            mask = torch.sigmoid((gate - thr) / self.gate_temp)
            rain = mask * residual
        else:
            rain = residual

        out = low + rain
        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
