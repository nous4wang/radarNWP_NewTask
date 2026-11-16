import torch
import torch.nn as nn
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.common import floor_at_zero


class BasicConv2d(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimVP_Core(nn.Module):
    def __init__(self, in_c, hid_c, in_len, out_len):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.enc = nn.Sequential(
            BasicConv2d(in_c, hid_c),
            BasicConv2d(hid_c, hid_c * 2, stride=2),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(in_len * hid_c * 2, in_len * hid_c * 2, 3, 1, 1, groups=in_len),
            nn.GroupNorm(in_len, in_len * hid_c * 2),
            nn.GELU(),
            nn.Conv2d(in_len * hid_c * 2, out_len * hid_c * 2, 1),
            nn.GroupNorm(out_len, out_len * hid_c * 2),
            nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, 3, 1, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat = self.enc(x.view(B * T, C, H, W))
        _, C_f, H_f, W_f = feat.shape
        t_feat = self.mid(feat.view(B, T * C_f, H_f, W_f))
        out = self.dec(t_feat.view(B * self.out_len, C_f, H_f, W_f))
        # 解码器以普通卷积收尾，在 log 空间预测，再指数还原成毫米，
        # 最后施加直通零点。
        # 原来的 nn.ReLU() 会切断负值像素的梯度（目标量是 log1p(mm)，几乎
        # 所有像素初始都在阈值以下），模型整轮预测同一个常数场。
        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))


@register_model("simvp_radar_only")
class SimVPRadarOnly(BasePrecipitationModel):
    """消融基准 1：纯雷达 10 通道输入，完全不给 NWP。"""
    def __init__(self, in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=64):
        super().__init__()
        self.core = SimVP_Core(in_c=in_radar_c, hid_c=hid_c, in_len=in_len, out_len=out_len)

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        # 显式忽略 x_nwp
        return self.core(x_radar)


@register_model("simvp_early_concat")
class SimVPEarlyConcat(BasePrecipitationModel):
    """消融基准 2：将 NWP 简单沿通道广播拼接，测试朴素拼接能力。"""
    def __init__(self, in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=64):
        super().__init__()
        # x_nwp 形状为 (B, 2, C_nwp, H, W)，展平为 2*C_nwp = 54 通道后拼接到每帧
        total_in_c = in_radar_c + in_nwp_c * 2
        self.core = SimVP_Core(in_c=total_in_c, hid_c=hid_c, in_len=in_len, out_len=out_len)

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, C_rad, H, W = x_radar.shape
        # x_nwp 展平广播拼接: (B, 1, 54, H, W) -> (B, T_in, 54, H, W)
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)
        return self.core(x_concat)