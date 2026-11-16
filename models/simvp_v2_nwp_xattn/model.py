"""SimVP-v2 with the NWP fused by a conventional spatial cross-attention.

Ablation row M2 of the NWP-injection study. The NWP does not arrive on the
input channels here: it is encoded into a feature map and the radar
features *query* it, position by position, through an ordinary
cross-attention block (:class:`~radar_nwp_bench.models.common.SpatialCrossAttention`).

The block is deliberately the textbook one -- multi-head attention, a
residual connection and a channel MLP -- and contains no Koopman or
quantum-inspired machinery at all. That is what makes the row a control
rather than a straw man: it answers "would any competent fusion of the
NWP have done as well as the tendency coupler?", and the only way that
question has an answer is if the alternative is a good one.

The seam is the same one the plugin uses. The NWP is fused into the
temporal latent, before the translator, so the two rows differ in *how*
the NWP is coupled and not in where it enters the network.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    ConvNeXtBlock,
    NWPEncoder,
    SpatialCrossAttention,
    floor_at_zero,
)


@register_model("simvp_v2_nwp_xattn")
class SimVPv2NWPCrossAttention(BasePrecipitationModel):
    """SimVP-v2 with the NWP fused into the latent by cross-attention.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels in one slice.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: width of the latent feature maps.
        num_blocks: number of spatio-temporal translator blocks.
        nwp_hidden: internal width of the NWP encoder backbone.
        num_heads: number of attention heads in the cross-attention.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        nwp_hidden: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.in_len = in_len
        self.out_len = out_len

        # The radar encoder sees the radar only. The NWP reaches the network
        # through the cross-attention below and through nothing else, which
        # is the whole point of this row.
        self.enc = nn.Sequential(
            nn.Conv2d(in_radar_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hid_c * 2),
            nn.GELU(),
        )

        # The NWP cube becomes one feature map on the radar grid. The shared
        # encoder keeps the analysis hour and the forecast hour apart and
        # blends them, so the two slices are not confused with each other.
        self.nwp_encoder = NWPEncoder(in_nwp_c, hid_c * 2, hidden=nwp_hidden)

        self.in_time_proj = nn.Conv2d(in_len * hid_c * 2, hid_c * 4, kernel_size=1)

        # The fusion itself: the radar latent is the query, the NWP feature
        # map is the key and the value.
        self.nwp_cross = SpatialCrossAttention(
            q_channels=hid_c * 4,
            kv_channels=hid_c * 2,
            num_heads=num_heads,
        )

        self.translator = nn.Sequential(
            *[ConvNeXtBlock(hid_c * 4) for _ in range(num_blocks)]
        )
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, kernel_size=1)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                hid_c * 2, hid_c, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, kernel_size=1),
        )

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """Predict future accumulated rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``.

        Returns:
            ``(B, T_out, 1, H, W)`` non-negative rainfall in millimetres.
        """
        b, t_in = x_radar.shape[:2]
        h, w = x_radar.shape[-2:]

        feats = self.enc(x_radar.reshape(b * t_in, -1, h, w))
        c_f, h_f, w_f = feats.shape[1:]
        feats_time = feats.reshape(b, t_in * c_f, h_f, w_f)
        hidden = self.in_time_proj(feats_time)

        # The NWP feature map is resampled onto the latent grid by the
        # cross-attention itself, so the two live at different resolutions
        # until they meet here.
        nwp_feat = self.nwp_encoder(x_nwp)
        hidden = self.nwp_cross(hidden, nwp_feat)

        hidden = self.translator(hidden)
        future = self.out_time_proj(hidden).reshape(b, self.out_len, c_f, h_f, w_f)

        out = self.dec(future.reshape(b * self.out_len, c_f, h_f, w_f))
        log_mm = out.reshape(b, self.out_len, 1, h, w)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
