"""SimVP-v2 Baseline: Pure CNN with Gated Spatiotemporal Attention Translator."""
import torch
import torch.nn as nn
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero


@register_model("simvp_v2")
class SimVPv2Model(BasePrecipitationModel):
    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
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
        self.translator = nn.Sequential(*[ConvNeXtBlock(hid_c * 4) for _ in range(num_blocks)])
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, kernel_size=1)

        # 空间解码器 (上采样回原分辨率)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, kernel_size=3, stride=2, padding=1, output_padding=1),
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

        # 1. 空间编码
        feats = self.enc(x_concat.view(B * T_in, -1, H, W))
        _, C_f, H_f, W_f = feats.shape

        # 2. 时空转换
        feats_time = feats.view(B, T_in * C_f, H_f, W_f)
        h = self.in_time_proj(feats_time)
        h = self.translator(h)
        future_feats = self.out_time_proj(h).view(B * self.out_len, C_f, H_f, W_f)

        # 3. 解码到地面降水
        out = self.dec(future_feats)
        # 解码器以普通卷积收尾，在 log 空间预测，再指数还原成毫米，
        # 最后施加直通零点。
        # 原来的 nn.ReLU() 会切断负值像素的梯度（目标量是 log1p(mm)，几乎
        # 所有像素初始都在阈值以下），模型整轮预测同一个常数场。
        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))