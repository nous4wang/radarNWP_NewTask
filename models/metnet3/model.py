"""MetNet-3 adapter: sparse-observation day forecasting with MaxViT blocks.

Paper: "Deep Learning for Day Forecasts from Sparse Observations" (MetNet-3,
arXiv:2306.06079).

The adapter follows the published recipe:

* **U-shaped convolutional backbone.**  The radar history is encoded by a
  convolutional encoder, downsampled through a U-Net and decoded back to the
  input grid.
* **MaxViT blocks.**  Every stage alternates *local* window attention with
  *global* grid attention, as in MaxViT: the local branch attends inside
  non-overlapping windows, the global branch attends on a downsampled map and
  is upsampled back.
* **Lead-time conditioning.**  A learnable embedding indexed by the forecast
  horizon is injected additively and multiplicatively (FiLM style) at every
  decoder stage, so the decoder knows which lead frame it is predicting.
* **NWP fusion.**  The NWP cube is encoded with :class:`..common.NWPEncoder`
  and fused into the bottleneck by cross-attention
  (:class:`..common.CrossAttention2d`).
* **One-pass multi-horizon output.**  The decoder is evaluated for all lead
  times in a single pass, the final layer predicts ``out_len`` channels which
  are then split into the forecast frames and mapped to millimetres with
  :class:`..common.RainHead`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    ChannelAttention,
    ConvBlock,
    CrossAttention2d,
    NWPEncoder,
    RainHead,
    SelfAttention2d,
    floor_at_zero,
)


class WindowAttention2d(nn.Module):
    """Local (windowed) self-attention, the local branch of a MaxViT block.

    The feature map is split into non-overlapping windows and attention is
    computed inside each window, which is the cheap local mixing that MaxViT
    pairs with the global grid attention.
    """

    def __init__(self, channels: int, window: int = 8, max_tokens: int = 1024) -> None:
        super().__init__()
        self.window = window
        self.attn = SelfAttention2d(channels, max_tokens=max_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        win = self.window
        pad_h = (win - h % win) % win
        pad_w = (win - w % win) % win
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[-2:]
        n_h, n_w = hp // win, wp // win
        # (B, C, H, W) -> one image per window.
        windows = (
            x.reshape(b, c, n_h, win, n_w, win)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b * n_h * n_w, c, win, win)
        )
        attended = self.attn(windows)
        attended = (
            attended.reshape(b, n_h, n_w, c, win, win)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, c, hp, wp)
        )
        return attended[:, :, :h, :w]


class MaxViTBlock(nn.Module):
    """MaxViT block: local window attention plus global grid attention.

    The two branches are applied in alternating order across the blocks of a
    stage, mirroring the block/grid attention alternation of MaxViT.
    """

    def __init__(
        self,
        channels: int,
        window: int = 8,
        global_stride: int = 2,
        use_global: bool = True,
    ) -> None:
        super().__init__()
        self.local = WindowAttention2d(channels, window=window)
        self.use_global = use_global
        if use_global:
            self.global_attn = SelfAttention2d(channels, max_tokens=1024)
            self.global_stride = global_stride
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x)
        if self.use_global:
            # Global grid attention runs on a downsampled map and is upsampled
            # back to the resolution of the local branch.
            small = F.avg_pool2d(x, self.global_stride, ceil_mode=True)
            glob = self.global_attn(small)
            glob = F.interpolate(
                glob, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + glob
        x = x + self.ffn(self.norm(x))
        return x



def init_rain_head(head: RainHead) -> None:
    """Start the rain head close to "no rain".

    The head predicts in log space and exponentiates at the end, so a small
    random final layer with a zero bias makes the initial prediction a small
    rainfall instead of an arbitrary value.
    """
    final = head.net[-1]
    if isinstance(final, nn.Conv2d):
        if final.bias is not None:
            nn.init.zeros_(final.bias)
        nn.init.normal_(final.weight, std=0.01)


class LeadTimeEmbedding(nn.Module):
    """FiLM-style conditioning on the forecast horizon.

    A learnable embedding indexed by the lead time produces a per-channel
    scale and shift that is injected into the decoder features, so the decoder
    knows which lead frame it is producing.
    """

    def __init__(self, lead_times: int, channels: int, dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(lead_times, dim)
        self.to_params = nn.Linear(dim, 2 * channels)
        self.channels = channels
        # The conditioning must start as the identity (scale 1, shift 0), but
        # only the *bias* may be zeroed. Zeroing the weight as well would
        # make the FiLM parameters identically zero for every lead time, and
        # the zero gradient would propagate back into the embedding: the
        # lead-time conditioning would never activate at all.
        nn.init.zeros_(self.to_params.bias)

    def forward(self, x: torch.Tensor, lead_index: torch.Tensor) -> torch.Tensor:
        """Inject the lead-time conditioning into ``x``.

        Args:
            x: ``(B, C, H, W)`` decoder features.
            lead_index: ``(B,)`` index of the forecast horizon.

        Returns:
            ``(B, C, H, W)`` conditioned features.
        """
        params = self.to_params(self.embed(lead_index))  # (B, 2C)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


@register_model("metnet3")
class MetNet3Adapter(BasePrecipitationModel):
    """MetNet-3 adapter for the radar + NWP precipitation benchmark.

    Args:
        in_radar_c: number of radar channels per history frame.
        in_nwp_c: number of NWP channels per NWP frame.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        width: base channel width of the U-Net.
        window: window size of the local MaxViT attention.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        width: int = 32,
        head_channels: int = 16,
        window: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.width = width
        self.head_channels = head_channels
        self.in_len = in_len
        self.out_len = out_len

        # Convolutional encoder over the radar stack.
        self.input_proj = ConvBlock(in_radar_c, width, 3)
        self.stem = nn.Sequential(
            ConvBlock(width, width, 3),
            ConvBlock(width, width, 3, residual=True),
        )
        self.down1 = nn.Sequential(
            ConvBlock(width, width * 2, 3, stride=2),
            MaxViTBlock(width * 2, window=window, use_global=False),
        )
        self.down2 = nn.Sequential(
            ConvBlock(width * 2, width * 4, 3, stride=2),
            MaxViTBlock(width * 4, window=window, use_global=True),
        )
        self.bottleneck = nn.Sequential(
            MaxViTBlock(width * 4, window=window, use_global=True),
            MaxViTBlock(width * 4, window=window, use_global=False),
        )

        # NWP cube -> bottleneck fusion.
        self.nwp_encoder = NWPEncoder(in_nwp_c, width * 4, hidden=width * 2)
        self.nwp_cross = CrossAttention2d(width * 4, width * 4)
        self.nwp_proj = nn.Conv2d(width * 4, width * 4, 1)

        # Decoder with lead-time conditioning at every stage.
        self.up2 = ConvBlock(width * 4 + width * 2, width * 2, 3)
        self.reduce = nn.Conv2d(width * 2, head_channels, 1)
        self.skip0 = nn.Conv2d(width, head_channels, 1)
        self.lead_embed_mid = LeadTimeEmbedding(out_len, width * 4, dim=32)
        self.lead_embed_deep = LeadTimeEmbedding(out_len, width * 2, dim=32)
        self.lead_embed_fine = LeadTimeEmbedding(out_len, head_channels, dim=32)
        self.channel_attn = ChannelAttention(head_channels)

        # Final layer: predicts all lead times at once, then they are split.
        self.head = nn.Conv2d(head_channels * out_len, out_len, 1)
        self._init_block_diagonal_head()
        self.rain_head = RainHead(1, hidden=width)
        init_rain_head(self.rain_head)

    def _init_block_diagonal_head(self) -> None:
        """Start the head close to a lead-aligned mapping.

        Output channel ``l`` initially reads mostly from the features that
        were produced for lead ``l``; the network is free to mix them later.
        """
        with torch.no_grad():
            weight = torch.zeros_like(self.head.weight)
            group = self.head_channels
            for lead in range(self.out_len):
                weight[lead, lead * group : (lead + 1) * group] = torch.randn(
                    group, 1, 1
                ) * (1.0 / group**0.5)
            self.head.weight.copy_(weight)
            self.head.bias.zero_()

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """Predict future accumulated rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``.

        Returns:
            ``(B, T_out, 1, H, W)`` non-negative rainfall in millimetres.
        """
        self._check_shapes(x_radar, x_nwp)
        b, t = x_radar.shape[:2]
        device = x_radar.device

        # Encode the radar history and build the U-Net skip connections.
        frames = self.input_proj(x_radar.reshape(b * t, *x_radar.shape[2:]))
        frames = frames.reshape(b, t, *frames.shape[1:])
        # Learned temporal pooling of the history into the encoder input.
        temporal_weight = torch.softmax(
            torch.linspace(0.0, 1.0, t, device=device, dtype=frames.dtype), dim=0
        )
        x0 = (frames * temporal_weight.reshape(1, t, 1, 1, 1)).sum(dim=1)
        x0 = self.stem(x0)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.bottleneck(x2)

        # NWP cube fused into the bottleneck by cross-attention.
        nwp_feat = self.nwp_encoder(x_nwp)
        if nwp_feat.shape[-2:] != x3.shape[-2:]:
            nwp_feat = F.interpolate(
                nwp_feat, size=x3.shape[-2:], mode="bilinear", align_corners=False
            )
        nwp_feat = self.nwp_proj(nwp_feat)
        x3 = self.nwp_cross(x3, nwp_feat)

        # Decoder: all lead times are decoded in a single pass, each with its
        # own lead-time conditioning at every stage.
        leads = torch.arange(self.out_len, device=device)
        lead_index = leads.repeat(b)  # (B * out_len,)
        d = x3.unsqueeze(1).expand(b, self.out_len, *x3.shape[1:])
        d = d.reshape(b * self.out_len, *x3.shape[1:])
        d = self.lead_embed_mid(d, lead_index)

        d = F.interpolate(d, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.up2(torch.cat([d, x1.repeat_interleave(self.out_len, dim=0)], dim=1))
        d = self.lead_embed_deep(d, lead_index)

        # Densify back to the input resolution and add the finest skip.
        d = self.reduce(d)
        d = F.interpolate(d, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d = d + self.skip0(x0).repeat_interleave(self.out_len, dim=0)
        d = self.lead_embed_fine(d, lead_index)
        d = self.channel_attn(d)

        # Final layer predicts the whole lead-time distribution at once.
        logits = self.head(d.reshape(b, self.out_len * self.head_channels, *d.shape[-2:]))
        logits = logits.reshape(b, self.out_len, 1, *logits.shape[-2:])

        flat = logits.reshape(b * self.out_len, 1, *logits.shape[-2:])
        rain = self.rain_head(flat)
        rain = rain.reshape(b, self.out_len, 1, *logits.shape[-2:])
        # A hard floor would cut the gradient wherever the rain is still
        # negative, which is everywhere at initialisation.
        return floor_at_zero(rain)
