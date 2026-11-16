"""SimVP-v2 with the quantum-inspired NWP tendency plugin attached.

This is the reference integration of
:class:`~radar_nwp_bench.models.common.QuantumInspiredTendencyCoupler`:
the plugin hangs off the baseline's feature map, before the rainfall head,
and modulates it with the physical tendency the NWP fields imply.

The point of the exercise is to show that the plugin is a *plug-in*: the
backbone, the head and the loss are the ones SimVP-v2 already had, and
the only line added is the call to the coupler. That is what makes it
usable on all fifteen baselines rather than on one.

The coupling is deliberately placed where the features still carry the
NWP's spatial resolution. SimVP-v2 downsamples twice, so its feature map
is at a quarter of the field size, and the tendency has to be resampled
onto it -- the plugin does that itself, because a baseline should not have
to know the spatial bookkeeping of the coupling it carries.
"""
import torch
import torch.nn as nn
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.common import (
    ConvNeXtBlock,
    QuantumInspiredTendencyCoupler,
    floor_at_zero,
)


@register_model("simvp_v2_quantum")
class SimVPv2QuantumNWP(BasePrecipitationModel):
    """SimVP-v2 with a quantum-inspired NWP tendency coupler.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: radar history frames.
        out_len: forecast frames.
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
    ):
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.in_len = in_len
        self.out_len = out_len

        total_in = in_radar_c + in_nwp_c * 2

        # 空间编码器 (下采样 2x)
        self.enc = nn.Sequential(
            nn.Conv2d(total_in, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hid_c * 2),
            nn.GELU(),
        )

        # 隐空间时空序列转换器 (gAST / ConvNeXt Translator)
        self.in_time_proj = nn.Conv2d(in_len * hid_c * 2, hid_c * 4, kernel_size=1)
        self.translator = nn.Sequential(
            *[ConvNeXtBlock(hid_c * 4) for _ in range(num_blocks)]
        )
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, kernel_size=1)

        # 空间解码器 (上采样回原分辨率)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, kernel_size=1),
        )

        # The plug-in. It modulates the (B, T_out, C, H, W) feature map with
        # the tendency the NWP implies, and passes the features through
        # untouched when the NWP is withheld.
        self.qnwp_coupler = QuantumInspiredTendencyCoupler(
            in_nwp_c=in_nwp_c,
            radar_feat_dim=hid_c * 2,
            out_len=out_len,
        )

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, _, H, W = x_radar.shape
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        # 1. 空间编码
        feats = self.enc(x_concat.view(B * T_in, -1, H, W))
        _, C_f, H_f, W_f = feats.shape

        # 2. 时空转换
        feats_time = feats.view(B, T_in * C_f, H_f, W_f)
        h = self.in_time_proj(feats_time)
        h = self.translator(h)
        future_feats = self.out_time_proj(h).view(B, self.out_len, C_f, H_f, W_f)

        # 3. 量子启发的物理倾向调制 (插件挂载点)
        future_feats = self.qnwp_coupler(future_feats, x_nwp)

        # 4. 解码到地面降水
        out = self.dec(future_feats.reshape(B * self.out_len, C_f, H_f, W_f))
        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
