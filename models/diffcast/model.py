"""DiffCast adapter for the radar + NWP precipitation nowcasting benchmark.

Paper
-----
"DiffCast: A Unified Framework via Residual Diffusion for Precipitation
Nowcasting", CVPR 2024.  Official implementation:
https://github.com/DeminYu98/DiffCast

What this adapter implements
-----------------------------
The paper decomposes the prediction into a deterministic mean forecast and a
stochastic residual:

1. ``SimVPBackbone`` -- a SimVP-style spatio-temporal predictor.  The radar
   history is projected into a latent token sequence, mixed by an MLP-Mixer
   over time and channel, and decoded into the mean forecast ``mu`` of shape
   ``(B, out_len, 1, H, W)``.  SimVP is the paper's default backbone.
2. ``ResidualUNet`` -- a 3D (video) U-Net with skip connections and a
   sinusoidal timestep embedding that denoises the residual ``r = y - mu``.
3. ``GlobalContextNetwork`` -- scans the concatenated input frames and the
   backbone output to produce a global context vector that conditions the
   denoiser, so the residual knows what the deterministic branch already
   explained.
4. A **DDIM sampling loop** with a configurable small number of steps that
   denoises the residual from an all-zero start (the "empty residual"
   hypothesis), so the diffusion only has to model the deviation from the
   mean forecast.

The final prediction is ``mu + r_hat``, clamped to be non-negative.

Substitutions and limitations
------------------------------
* The paper's DiffCast is radar-only.  Here the NWP fields are an addition:
  they are encoded with the shared ``NWPEncoder`` and injected into the
  denoiser as an extra conditioning channel alongside the global context.
* ``common.NWPEncoder`` concatenates a ``hidden``-wide temporal embedding to
  every NWP slice before its first convolution, so it has to be constructed
  with ``in_nwp_c + hidden`` input channels or the channel counts disagree.
* The published checkpoints are for SEVIR; everything here is trained from
  scratch inside the benchmark.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    ConvBlock,
    NWPEncoder,
    RainHead,
    floor_at_zero,
)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding (Ho et al., DDPM)."""

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: ``t`` -- ``(B,)`` timesteps.  Returns ``(B, dim)``."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class MLPMixerBlock(nn.Module):
    """SimVP's mixer block: token mixing over time, channel mixing over features.

    The token axis is the *time* axis of the sequence, so the block works for
    any spatial resolution -- the spatial extent is carried by the feature
    axis instead.
    """

    def __init__(self, seq_len: int, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, seq_len),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` -- ``(B, seq_len, dim)``."""
        h = self.norm1(x).transpose(1, 2)
        x = x + self.token(h).transpose(1, 2)
        return x + self.channel(self.norm2(x))


class SimVPBackbone(nn.Module):
    """SimVP-style deterministic predictor producing the mean forecast ``mu``.

    Each radar frame is encoded into a feature map, the maps are stacked over
    time into a token sequence of ``T`` tokens per pixel, and an MLP-Mixer
    mixes the tokens (over time) and the features (over channels).  The
    decoder then expands the mixed representation to ``out_len`` frames and
    back to the input resolution.

    The output lives in log1p(mm) space, matching the training target.
    """

    def __init__(
        self,
        in_c: int,
        in_len: int,
        out_len: int,
        dim: int = 64,
        hidden: int = 128,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.dim = dim
        self.encoder = nn.Sequential(
            ConvBlock(in_c, dim, 3, stride=2),
            ConvBlock(dim, dim, 3, residual=True),
        )
        self.blocks = nn.ModuleList(
            [MLPMixerBlock(in_len, dim, hidden) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.time_expand = nn.Linear(in_len, out_len)
        self.decoder = nn.Sequential(
            ConvBlock(dim, dim, 3),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(dim, dim, 3, residual=True),
        )

    def forward(self, x_radar: torch.Tensor) -> torch.Tensor:
        """Args: ``x_radar`` -- ``(B, T, C, H, W)``.

        Returns ``(B, out_len, dim, H, W)`` features.
        """
        b, t, c, h, w = x_radar.shape
        y = self.encoder(x_radar.reshape(b * t, c, h, w))
        hh, ww = y.shape[-2:]
        y = y.reshape(b, t, self.dim, hh, ww)
        # One token per time step, per pixel.
        tokens = y.permute(0, 3, 4, 1, 2).reshape(b * hh * ww, t, self.dim)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        tokens = self.time_expand(tokens.transpose(1, 2)).transpose(1, 2)
        out = tokens.reshape(b, hh, ww, self.out_len, self.dim)
        out = out.permute(0, 3, 4, 1, 2).reshape(b * self.out_len, self.dim, hh, ww)
        out = self.decoder(out)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out.reshape(b, self.out_len, self.dim, h, w)


class ResidualUNet(nn.Module):
    """3D U-Net denoiser for the residual field.

    Operates on ``(B, T, C, H, W)`` tensors: the temporal axis is folded into
    the batch for the 2D convolutions, the timestep embedding is injected into
    every block, and skip connections carry encoder features to the decoder.
    """

    def __init__(
        self,
        in_ch: int,
        cond_ch: int,
        base: int = 32,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbedding(cond_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, base),
            nn.SiLU(),
            nn.Linear(base, base),
        )
        self.cond_proj = nn.Conv2d(cond_ch, base, 1)
        self.stem = nn.Sequential(
            ConvBlock(in_ch, base, 3),
            ConvBlock(base, base, 3, residual=True),
        )
        self.down1 = nn.Sequential(
            ConvBlock(base, base * 2, 3, stride=2),
            ConvBlock(base * 2, base * 2, 3, residual=True),
        )
        self.down2 = nn.Sequential(
            ConvBlock(base * 2, base * 4, 3, stride=2),
            ConvBlock(base * 4, base * 4, 3, residual=True),
        )
        self.mid = ConvBlock(base * 4, base * 4, 3)
        # skip connections: mid(4b) + s1(2b) -> 2b ; u2(2b) + s0(b) -> b
        self.up2 = ConvBlock(base * 6, base * 2, 3)
        self.up1 = ConvBlock(base * 3, base, 3)
        self.out = nn.Sequential(
            ConvBlock(base, base, 3),
            nn.Conv2d(base, 1, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Args: ``x`` -- ``(B, T, C, H, W)``, ``cond`` -- ``(B, cond_ch, H, W)``.

        Returns the predicted noise, same shape as ``x``.
        """
        b, n_frames = x.shape[:2]
        h, w = x.shape[-2:]
        emb = self.t_mlp(self.t_embed(t)).view(b, -1, 1, 1)
        cond_map = self.cond_proj(cond) + emb

        inp = x.reshape(b * n_frames, *x.shape[2:])
        c = cond_map.repeat_interleave(n_frames, dim=0)
        s0 = self.stem(inp) + c
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        m = self.mid(s2)
        u2 = self.up2(
            torch.cat([F.interpolate(m, size=s1.shape[-2:], mode="nearest"), s1], dim=1)
        )
        u1 = self.up1(
            torch.cat([F.interpolate(u2, size=s0.shape[-2:], mode="nearest"), s0], dim=1)
        )
        out = self.out(u1)
        return out.reshape(b, n_frames, 1, h, w)


class GlobalContextNetwork(nn.Module):
    """Scans the input frames and the backbone output into a global context.

    The context is a spatial map that is injected into the denoiser, so the
    residual is modelled relative to what the deterministic branch already
    captured over the whole domain.  A global average-pooled term is added
    back to the map, which gives the denoiser a domain-wide summary of the
    situation in addition to the local one.
    """

    def __init__(self, in_c: int, in_len: int, out_len: int, dim: int = 32) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.net = nn.Sequential(
            ConvBlock(in_c * in_len + out_len, dim, 3, stride=2),
            ConvBlock(dim, dim, 3, residual=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self, x_radar: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Args: ``x_radar`` -- ``(B, T, C, H, W)``, ``mu`` -- ``(B, out_len, 1, H, W)``.

        Returns ``(B, dim, H/2, W/2)``.
        """
        b = x_radar.shape[0]
        frames = x_radar.reshape(b, -1, *x_radar.shape[-2:])
        back = mu.reshape(b, -1, *mu.shape[-2:])
        y = self.net(torch.cat([frames, back], dim=1))
        pooled = self.pool(y)
        return y + pooled


@register_model("diffcast")
class DiffCastAdapter(BasePrecipitationModel):
    """Deterministic mean forecast plus residual diffusion.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        sample_steps: number of DDIM steps used at inference.
        nwp_channels: width of the NWP conditioning map.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        sample_steps: int = 6,
        dim: int = 64,
        hidden: int = 128,
        n_blocks: int = 2,
        unet_base: int = 32,
        nwp_hidden: int = 64,
        nwp_channels: int = 32,
        train_timesteps: int = 1000,
        residual_clip: float = 20.0,
        **kwargs: object,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.sample_steps = int(sample_steps)
        self.train_timesteps = train_timesteps
        self.residual_clip = float(residual_clip)

        self.backbone = SimVPBackbone(
            in_radar_c, in_len, out_len, dim=dim, hidden=hidden, n_blocks=n_blocks
        )
        # NWPEncoder appends a `hidden`-wide temporal embedding to each slice.
        self.nwp_encoder = NWPEncoder(in_nwp_c, nwp_channels, hidden=nwp_hidden)
        self.nwp_to_cond = nn.Conv2d(nwp_channels, nwp_channels, 1)
        self.global_context = GlobalContextNetwork(in_radar_c, in_len, out_len, dim=32)
        self.unet = ResidualUNet(in_ch=1, cond_ch=nwp_channels + 32, base=unet_base)
        self.rain_head = RainHead(dim, hidden=32)

        betas = torch.linspace(1e-4, 0.02, train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0), persistent=False)

    # -- helpers ----------------------------------------------------------
    def _ddim_sample(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, int, int, int, int],
        steps: int,
    ) -> torch.Tensor:
        """DDIM sampling of the residual field, starting from all zeros.

        Starting from zero means the diffusion starts from the "no residual"
        hypothesis, matching the residual formulation of the paper where the
        deterministic branch already provides the mean.  The loop is
        differentiable, so the residual branch trains end to end.

        The intermediate ``x0`` estimate is clamped (``dynamic
        thresholding``): the DDIM update divides by ``sqrt(alpha_cumprod)``,
        which is tiny at large timesteps and would otherwise let an
        imperfectly trained denoiser blow the residual up.
        """
        b = shape[0]
        z = torch.zeros(shape, device=cond.device, dtype=cond.dtype)
        alphas = self.alphas_cumprod
        ts = torch.linspace(
            0, self.train_timesteps - 1, steps, device=cond.device
        ).round().long().flip(0)
        for i, ti in enumerate(ts):
            t_batch = torch.full((b,), int(ti), device=cond.device, dtype=torch.long)
            eps = self.unet(z, t_batch, cond)
            a_t = alphas[ti]
            a_prev = alphas[ts[i + 1]] if i + 1 < len(ts) else torch.ones_like(a_t)
            x0 = (z - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-8)
            x0 = x0.clamp(-self.residual_clip, self.residual_clip)
            z = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps
        return z

    # -- forward ----------------------------------------------------------
    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """Predict future accumulated rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``.

        Returns:
            ``(B, T_out, 1, H, W)`` non-negative rainfall in millimetres.
        """
        self._check_shapes(x_radar, x_nwp)
        b = x_radar.shape[0]
        h, w = x_radar.shape[-2:]

        # (1) deterministic backbone -> mean forecast mu.
        # RainHead already exponentiates, so `mu` is directly in millimetres
        # and the residual is modelled in the same (mm) space as the paper.
        feats = self.backbone(x_radar)  # (B, out_len, dim, H, W)
        mu = self.rain_head(feats.reshape(b * self.out_len, -1, h, w))
        mu = mu.reshape(b, self.out_len, 1, h, w)

        # (2) global context from the inputs and the backbone output
        ctx = self.global_context(x_radar, mu)

        # (3) NWP conditioning, concatenated to the denoiser conditioning
        nwp_feat = self.nwp_to_cond(self.nwp_encoder(x_nwp))
        nwp_feat = F.interpolate(
            nwp_feat, size=ctx.shape[-2:], mode="bilinear", align_corners=False
        )
        cond = torch.cat([ctx, nwp_feat], dim=1)

        # (4) residual diffusion, sampled from zeros.  The residual is modelled
        # on the half-resolution conditioning grid: the residual field is smooth
        # by construction, and it keeps the DDIM loop cheap enough for CPU.
        rh, rw = cond.shape[-2:]
        residual = self._ddim_sample(
            cond, (b, self.out_len, 1, rh, rw), self.sample_steps
        )
        residual = F.interpolate(
            residual.reshape(b * self.out_len, 1, rh, rw),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, self.out_len, 1, h, w)

        # A hard clamp here froze the entire model: `mu` starts near
        # zero and the sampled residual is negative almost everywhere,
        # so every pixel sat below the floor and no parameter received
        # a gradient at all.
        out = floor_at_zero(mu + residual)
        return out
