"""SimVP-v2 with the NWP replaced by its difference between the two slices.

Ablation row M3 of the NWP-injection study. The model is handed the
*tendency* of the NWP -- the field difference between its analysis hour and
its forecast hour -- and nothing else of the NWP: the absolute state is
never shown to the network.

The point is to separate the two things the NWP could be doing. A model
that receives the raw NWP can use the state of the atmosphere; a model
that receives only the difference can use how the atmosphere is *going to
change* over the target window. The plugin's coupler claims the second is
the useful one, and this row is the control that tests the claim without
any Koopman or quantum machinery: if the plain difference is as good as
the coupler, then the coupler's machinery is not what is doing the work.

The difference is a constant-offset invariant field: shifting both NWP
slices by the same constant leaves it unchanged. That is a property the
row can be *tested* for, and the test file does exactly that.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero


@register_model("simvp_v2_nwp_delta")
class SimVPv2NWPDelta(BasePrecipitationModel):
    """SimVP-v2 with the NWP difference field in place of the NWP.

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

        # The radar history and the NWP tendency, side by side: the same
        # input width as the plain concatenation row, so the two differ in
        # what the NWP channels carry and not in how much of the network
        # there is.
        total_in = in_radar_c + in_nwp_c
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

        # The tendency: forecast hour minus analysis hour. Only this field
        # reaches the network, so a constant offset added to both NWP
        # slices cannot change the prediction.
        tendency = x_nwp[:, -1] - x_nwp[:, 0]
        tend_flat = tendency.unsqueeze(1).expand(-1, t_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, tend_flat], dim=2)

        feats = self.enc(x_concat.reshape(b * t_in, -1, h, w))
        c_f, h_f, w_f = feats.shape[1:]
        feats_time = feats.reshape(b, t_in * c_f, h_f, w_f)
        hidden = self.translator(self.in_time_proj(feats_time))
        future = self.out_time_proj(hidden).reshape(b, self.out_len, c_f, h_f, w_f)

        out = self.dec(future.reshape(b * self.out_len, c_f, h_f, w_f))
        log_mm = out.reshape(b, self.out_len, 1, h, w)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
