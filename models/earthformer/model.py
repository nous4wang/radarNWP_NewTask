"""Earthformer Baseline with Cuboid Spatiotemporal Self-Attention."""
import math
import torch
import torch.nn as nn
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.common import floor_at_zero


class CuboidSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_blocks, N_tokens, Dim)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // self.num_heads))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class EarthformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CuboidSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register_model("earthformer")
class EarthformerModel(BasePrecipitationModel):
    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        embed_dim: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        total_in = in_radar_c + in_nwp_c * 2

        # 3D Patch 切块 (空间下采样 4x)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(total_in, embed_dim, kernel_size=4, stride=4),
            nn.GroupNorm(4, embed_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            EarthformerBlock(embed_dim, num_heads=4) for _ in range(num_blocks)
        ])

        # 时间维度投影 (从 11 步投影至 10 步)
        self.time_proj = nn.Linear(in_len, out_len)

        # 解码上采样回原分辨率
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, _, H, W = x_radar.shape
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        # 1. Patch 提取
        patches = self.patch_embed(x_concat.view(B * T_in, -1, H, W))  # (B*T, Dim, H/4, W/4)
        _, dim, H_p, W_p = patches.shape

        # 2. 转换为空间块序列并执行 Cuboid Attention
        # (B, T, Dim, H_p*W_p) -> (B * H_p*W_p, T, Dim)
        tokens = patches.view(B, T_in, dim, H_p * W_p).permute(0, 3, 1, 2).reshape(B * H_p * W_p, T_in, dim)
        for blk in self.blocks:
            tokens = blk(tokens)

        # 3. 时间预测转换
        # (B * H_p*W_p, Dim, T_in) -> (B * H_p*W_p, Dim, T_out)
        out_tokens = self.time_proj(tokens.transpose(1, 2)).transpose(1, 2)
        out_feats = out_tokens.view(B, H_p, W_p, self.out_len, dim).permute(0, 3, 4, 1, 2).reshape(B * self.out_len, dim, H_p, W_p)

        # 4. 上采样解码
        decoded = self.decoder(out_feats)
        # The decoder ends in a plain convolution, predicts in log space,
        # and is exponentiated back to millimetres; the floor is applied
        # outside, with a straight-through gradient. A hard nn.ReLU() here
        # would kill the gradient at every pixel that starts below zero --
        # which is nearly all of them, because the target is log1p(mm) -- and
        # the model would predict a constant field for the whole run.
        log_mm = decoded.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))