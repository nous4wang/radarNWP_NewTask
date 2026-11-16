"""SimVP-v2 with the NWP concatenated onto the radar input.

Ablation row M1 of the NWP-injection study: the plainest way of giving a
nowcasting model the NWP is to paste the fields onto the channel axis of
the radar history and let the first convolution work out what they mean.
The forecast slice of the NWP is repeated across the history steps
exactly the way ``models/simvp/model.py`` repeats the NWP, so the encoder
sees 37 channels where the radar alone would give 10.

Everything else is SimVP-v2's: the encoder, the ConvNeXt translator, the
decoder and the millimetre output are unchanged. The row exists to
separate two claims that are easy to confuse in a paper. "The NWP helps"
is a statement about the *information* the model is given; "the tendency
coupler helps" is a statement about the *mechanism* that carries it.
This row buys the first one as cheaply as possible, so that the second
one can be measured against it rather than against a radar-only model.

The forecast slice (``x_nwp[:, -1]``) is the field that carries the
mode's evolution across the target window. The analysis slice is
deliberately left out: the difference row differs from this row exactly
in that it replaces the forecast state by the *change* between the two
slices, and the two are directly comparable that way.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero


@register_model("simvp_v2_nwp_concat")
class SimVPv2NWPConcat(BasePrecipitationModel):
    """SimVP-v2 with the NWP forecast slice concatenated onto the radar.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels in one slice.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: width of the latent feature maps.
        num_blocks: number of spatio-temporal translator blocks.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.in_len = in_len
        self.out_len = out_len

        # The radar history and the NWP forecast slice, side by side. This
        # is the whole of the difference to SimVP-v2: the encoder sees the
        # NWP on its input channels instead of not seeing it at all.
        total_in = in_radar_c + in_nwp_c
        self.enc = nn.Sequential(
            nn.Conv2d(total_in, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hid_c * 2),
            nn.GELU(),
        )

        # The spatiotemporal translator, unchanged from SimVP-v2.
        self.in_time_proj = nn.Conv2d(in_len * hid_c * 2, hid_c * 4, kernel_size=1)
        self.translator = nn.Sequential(
            *[ConvNeXtBlock(hid_c * 4) for _ in range(num_blocks)]
        )
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, kernel_size=1)

        # The decoder back to the radar grid, unchanged from SimVP-v2.
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

        # The forecast slice, repeated over the history steps: the same
        # fields next to every radar frame, which is what "paste the NWP
        # on" means.
        nwp_flat = x_nwp[:, -1].unsqueeze(1).expand(-1, t_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        feats = self.enc(x_concat.reshape(b * t_in, -1, h, w))
        c_f, h_f, w_f = feats.shape[1:]
        feats_time = feats.reshape(b, t_in * c_f, h_f, w_f)
        hidden = self.translator(self.in_time_proj(feats_time))
        future = self.out_time_proj(hidden).reshape(b, self.out_len, c_f, h_f, w_f)

        out = self.dec(future.reshape(b * self.out_len, c_f, h_f, w_f))
        # The decoder predicts in log space and the exponentiation turns it
        # back into millimetres; the straight-through floor then keeps the
        # rainfall non-negative without freezing the pixels that start
        # below zero.
        log_mm = out.reshape(b, self.out_len, 1, h, w)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
