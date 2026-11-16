"""CasCast adapter for the radar + NWP precipitation nowcasting benchmark.

Paper
-----
Gong, Bai, Ye, Xu, Liu, Dai, Yang and Ouyang, "CasCast: Skillful High-resolution
Precipitation Nowcasting via Cascaded Modelling", ICML 2024 (PMLR 235).
Official implementation: https://github.com/OpenEarthLab/CasCast

What this adapter implements
-----------------------------
The paper splits nowcasting into a deterministic part that captures the
mesoscale signal in pixel space and a probabilistic part that models the
small-scale detail in a compressed latent space.  The three-stage cascade is
reproduced here:

1. ``DeterministicBranch`` -- an EarthFormer-like space-time transformer over
   the radar history.  Temporal attention runs across frames at every spatial
   location, spatial attention runs across locations inside every frame, and a
   small autoregressive decoder rolls the history forward into ``out_len``
   blurry mean-forecast frames.
2. ``FrameAutoencoder`` -- a frame-wise convolutional autoencoder that maps a
   single rainfall frame to a 4-channel latent at 1/4 resolution and back.
3. ``LatentDiffusionDenoiser`` -- a DiT-style diffusion transformer over the
   latent sequence.  Every frame is denoised with cross-attention to its own
   conditioning (deterministic latent + NWP latent) and with temporal attention
   to the other frames, which is the "frame-wise guided" part of the paper.
   Inference runs a DDIM sampler with a small, configurable number of steps.

The NWP fields are an addition to the published model (which is radar-only).
They are encoded with the shared ``NWPEncoder`` and concatenated to the latent
conditioning that the diffusion transformer cross-attends to.

Substitutions and limitations
------------------------------
* The paper is trained on SEVIR VIL at 384x384 with pre-trained checkpoints;
  here the same mechanism runs on the 10-channel BJ radar composites at the
  benchmark resolution and every component is trained from scratch.
* The published model predicts VIL-like intensities.  Here the autoencoder and
  the final head live in the log1p(mm) space that ``RainHead`` uses.
* ``common.NWPEncoder`` concatenates a ``hidden``-wide temporal embedding to
  every NWP slice before its first convolution, so it has to be constructed
  with ``in_nwp_c + hidden`` input channels or the channel counts disagree.
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
        """Args: ``t`` -- ``(B,)`` integer timesteps.  Returns ``(B, dim)``."""
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


class SpaceTimeAttentionBlock(nn.Module):
    """EarthFormer-style space-time attention block.

    The input is ``(B, T, C, H, W)``.  Temporal attention mixes the frames at
    each spatial location, spatial attention mixes the locations inside each
    frame.  Both are plain multi-head self-attention over the reshaped token
    axis, which is the "separable space-time attention" of the paper.
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        # temporal attention: one sequence of T tokens per spatial location
        y = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        y = self.attn_t(self.norm_t(y), self.norm_t(y), self.norm_t(y), need_weights=False)[0]
        x = x + y.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)
        # spatial attention: one sequence of H*W tokens per frame
        y = x.permute(0, 1, 3, 4, 2).reshape(b * t, h * w, c)
        y = self.attn_s(self.norm_s(y), self.norm_s(y), self.norm_s(y), need_weights=False)[0]
        x = x + y.reshape(b, t, h, w, c).permute(0, 1, 4, 2, 3)
        z = x.permute(0, 1, 3, 4, 2)
        z = self.mlp(self.norm_m(z))
        return x + z.permute(0, 1, 4, 2, 3)


class DeterministicDecoderBlock(nn.Module):
    """One autoregressive decode step: self-attention, cross-attention to the
    encoded history, then a feed-forward layer."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_c = nn.LayerNorm(dim)
        self.attn_c = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        y = self.attn_s(self.norm_s(q), self.norm_s(q), self.norm_s(q), need_weights=False)[0]
        q = q + y
        y = self.attn_c(self.norm_c(q), memory, memory, need_weights=False)[0]
        q = q + y
        return q + self.mlp(self.norm_m(q))


class DeterministicBranch(nn.Module):
    """Pixel-space deterministic branch: a blurry mean forecast.

    The radar history is embedded into ``(B, T, C, H/4, W/4)`` tokens, refined
    by space-time attention, and then decoded autoregressively: each step
    consumes the previously predicted log-rainfall frame and cross-attends to
    the encoded history.
    """

    def __init__(
        self,
        in_radar_c: int,
        in_len: int,
        out_len: int,
        dim: int = 64,
        n_blocks: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.dim = dim
        self.embed = nn.Conv2d(in_radar_c, dim, kernel_size=4, stride=4)
        self.hist_time = nn.Parameter(torch.zeros(1, in_len, dim))
        nn.init.normal_(self.hist_time, std=0.02)
        self.blocks = nn.ModuleList(
            [SpaceTimeAttentionBlock(dim, num_heads) for _ in range(n_blocks)]
        )
        self.dec_embed = nn.Conv2d(1, dim, kernel_size=3, padding=1)
        self.step_time = nn.Parameter(torch.zeros(1, out_len, dim))
        nn.init.normal_(self.step_time, std=0.02)
        self.dec_block = DeterministicDecoderBlock(dim, num_heads)
        self.frame_head = nn.Sequential(
            ConvBlock(dim, dim, 3),
            nn.Conv2d(dim, 1, 1),
        )

    def forward(self, x_radar: torch.Tensor) -> torch.Tensor:
        """Args: ``x_radar`` -- ``(B, T, C, H, W)``.

        Returns the blurry mean forecast in log1p(mm) space, ``(B, out_len, 1, H, W)``.
        """
        b, t, c, h, w = x_radar.shape
        tok = self.embed(x_radar.reshape(b * t, c, h, w))
        hh, ww = tok.shape[-2:]
        x = tok.reshape(b, t, self.dim, hh, ww) + self.hist_time[:, :t].view(1, t, self.dim, 1, 1)
        for blk in self.blocks:
            x = blk(x)
        memory = x.permute(0, 1, 3, 4, 2).reshape(b, t * hh * ww, self.dim)

        out = []
        prev = torch.zeros(b, 1, hh, ww, device=x_radar.device, dtype=x_radar.dtype)
        for step in range(self.out_len):
            q = self.dec_embed(prev) + self.step_time[:, step].view(1, self.dim, 1, 1)
            q = q.permute(0, 2, 3, 1).reshape(b, hh * ww, self.dim)
            q = self.dec_block(q, memory)
            q = q.permute(0, 2, 1).reshape(b, self.dim, hh, ww)
            log_rain = self.frame_head(q)  # (B, 1, H/4, W/4)
            # The recurrent input stays at the token resolution; only the
            # emitted frame is upsampled to the radar grid.
            prev = log_rain
            out.append(
                F.interpolate(log_rain, size=(h, w), mode="bilinear", align_corners=False)
            )
        return torch.stack(out, dim=1)


class FrameAutoencoder(nn.Module):
    """Frame-wise convolutional autoencoder: ``(1, H, W)`` <-> ``(Cz, H/4, W/4)``.

    The encoder halves the resolution twice and compresses to ``latent_ch``
    channels; the decoder restores the resolution with nearest-neighbour
    upsampling plus convolutions.  It is applied frame by frame, exactly as
    in the paper, where the autoencoder defines the low-dimensional space in
    which the diffusion model operates.
    """

    def __init__(self, latent_ch: int = 4, feat_ch: int = 32, width: int = 32) -> None:
        super().__init__()
        self.latent_ch = latent_ch
        self.feat_ch = feat_ch
        self.encoder = nn.Sequential(
            ConvBlock(1, width, 3),
            ConvBlock(width, width, 3, stride=2),
            ConvBlock(width, width, 3, stride=2),
            nn.Conv2d(width, latent_ch, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            ConvBlock(latent_ch, width, 3),
            ConvBlock(width, width, 3),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(width, width, 3),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(width, width, 3),
            nn.Conv2d(width, feat_ch, 3, padding=1),
        )

    def encode(self, x: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """Compress frames ``(B, T, 1, H, W)`` to ``(B, T, Cz, h, w)``."""
        b, t = x.shape[:2]
        z = self.encoder(x.reshape(b * t, 1, *x.shape[-2:]))
        if z.shape[-2:] != grid:
            z = F.interpolate(z, size=grid, mode="bilinear", align_corners=False)
        return z.reshape(b, t, self.latent_ch, *grid)

    def decode(self, z: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Expand latents ``(B, T, Cz, h, w)`` to features ``(B, T, feat_ch, H, W)``."""
        b, t = z.shape[:2]
        f = self.decoder(z.reshape(b * t, self.latent_ch, *z.shape[-2:]))
        if f.shape[-2:] != size:
            f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
        return f.reshape(b, t, self.feat_ch, *size)


class DiffusionTransformerBlock(nn.Module):
    """DiT-style block over a latent sequence.

    The input is ``(B, T, N, C)`` with ``N`` spatial tokens per frame.  Each
    block applies spatial self-attention, cross-attention to the per-frame
    conditioning tokens (the "frame-wise guidance" of the paper), temporal
    attention across frames and a feed-forward layer.  The timestep embedding
    enters through an adaLN-style scale/shift.
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_c = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn_c = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn_t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)
        self.gate_s = nn.Parameter(torch.ones(1))
        self.gate_c = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        b, t, n, c = x.shape
        p = self.ada(t_emb).view(b, 1, 1, 4 * c)
        shift_s, scale_s, shift_c, scale_c = torch.split(p, c, dim=-1)

        y = self.norm_s(x) * (1 + scale_s) + shift_s
        y = self.attn_s(y.reshape(b * t, n, c), y.reshape(b * t, n, c), y.reshape(b * t, n, c), need_weights=False)[0]
        x = x + self.gate_s * y.reshape(b, t, n, c)

        y = self.norm_c(x) * (1 + scale_c) + shift_c
        y = self.attn_c(
            y.reshape(b * t, n, c),
            cond.reshape(b * t, cond.shape[2], c),
            cond.reshape(b * t, cond.shape[2], c),
            need_weights=False,
        )[0]
        x = x + self.gate_c * y.reshape(b, t, n, c)

        y = x.transpose(1, 2).reshape(b * n, t, c)
        y = self.attn_t(self.norm_t(y), self.norm_t(y), self.norm_t(y), need_weights=False)[0]
        x = x + y.reshape(b, n, t, c).transpose(1, 2)

        return x + self.mlp(self.norm_m(x))


class LatentDiffusionDenoiser(nn.Module):
    """Diffusion transformer that denoises the latent sequence.

    The denoiser predicts the noise of a noised latent sequence conditioned on
    (a) the deterministic branch latent of the same frame, (b) the NWP latent
    and (c) the diffusion timestep.
    """

    def __init__(
        self,
        latent_ch: int,
        cond_ch: int,
        dim: int = 64,
        n_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Conv2d(latent_ch, dim, 3, padding=1)
        self.cond_proj = nn.Conv2d(cond_ch, dim, 3, padding=1)
        self.t_embed = SinusoidalTimestepEmbedding(dim)
        self.blocks = nn.ModuleList(
            [DiffusionTransformerBlock(dim, num_heads) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_proj = nn.Conv2d(dim, latent_ch, 3, padding=1)

    def forward(
        self,
        z_t: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Args: ``z_t`` -- ``(B, T, Cz, h, w)``, ``cond`` -- ``(B, T, Cc, h, w)``.

        Returns the predicted noise with the shape of ``z_t``.
        """
        b, n_frames = z_t.shape[:2]
        x = self.in_proj(z_t.reshape(b * n_frames, *z_t.shape[2:]))
        n = x.shape[-2] * x.shape[-1]
        x = x.reshape(b * n_frames, self.dim, n).transpose(1, 2).reshape(b, n_frames, n, self.dim)
        c = self.cond_proj(cond.reshape(b * n_frames, *cond.shape[2:]))
        c = c.reshape(b * n_frames, self.dim, n).transpose(1, 2).reshape(b, n_frames, n, self.dim)
        t_emb = self.t_embed(t).view(b, self.dim)
        for blk in self.blocks:
            x = blk(x, c, t_emb)
        x = self.out_proj(
            self.norm_out(x).reshape(b * n_frames, n, self.dim).transpose(1, 2).reshape(
                b * n_frames, self.dim, *z_t.shape[-2:]
            )
        )
        return x.reshape(b, n_frames, *z_t.shape[2:])


@register_model("cascast_nwp")
class CasCastNWPAdapter(BasePrecipitationModel):
    """Cascaded deterministic + latent-diffusion nowcasting adapter.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        sample_steps: number of DDIM steps used at inference.
        noise_seed: seed of the (fixed) initial noise, so inference is
            reproducible while gradients still flow through the sampler.
        dim: token width of the transformers.
        n_blocks: number of space-time blocks in the deterministic branch.
        n_diff_layers: number of diffusion transformer blocks.
        latent_ch: width of the autoencoder latent.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        sample_steps: int = 6,
        noise_seed: int = 0,
        dim: int = 64,
        n_blocks: int = 2,
        n_diff_layers: int = 2,
        latent_ch: int = 4,
        nwp_hidden: int = 64,
        nwp_channels: int = 32,
        train_timesteps: int = 1000,
        latent_clip: float = 5.0,
        **kwargs: object,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.sample_steps = int(sample_steps)
        self.noise_seed = int(noise_seed)
        self.latent_ch = latent_ch
        self.train_timesteps = train_timesteps
        self.latent_clip = float(latent_clip)

        self.deterministic = DeterministicBranch(
            in_radar_c, in_len, out_len, dim=dim, n_blocks=n_blocks
        )
        self.autoencoder = FrameAutoencoder(latent_ch=latent_ch, feat_ch=32)
        # NWPEncoder appends a `hidden`-wide temporal embedding to each slice.
        self.nwp_encoder = NWPEncoder(
            in_nwp_c, nwp_channels, hidden=nwp_hidden
        )
        self.nwp_to_latent = nn.Conv2d(nwp_channels, latent_ch, 1)
        self.denoiser = LatentDiffusionDenoiser(
            latent_ch=latent_ch,
            cond_ch=2 * latent_ch,
            dim=dim,
            n_layers=n_diff_layers,
        )
        self.rain_head = RainHead(32 + 1)

        betas = torch.linspace(1e-4, 0.02, train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0), persistent=False)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _grid(h: int, w: int) -> Tuple[int, int]:
        """Latent grid of the autoencoder for an arbitrary input resolution."""
        return (max(1, (h + 3) // 4), max(1, (w + 3) // 4))

    def _ddim_sample(
        self, cond: torch.Tensor, steps: int, size: Tuple[int, int]
    ) -> torch.Tensor:
        """DDIM sampling loop over the latent sequence.

        The loop is differentiable (no ``no_grad``) and uses a fixed seed for
        the initial noise, so the adapter is deterministic at evaluation time
        while still trainable end to end.  The intermediate ``x0`` estimate
        is clamped, because the DDIM update divides by ``sqrt(alpha_cumprod)``
        which is tiny at large timesteps.
        """
        b, n_frames = cond.shape[:2]
        shape = (b, n_frames, self.latent_ch, size[0], size[1])
        gen = torch.Generator(device=cond.device)
        gen.manual_seed(self.noise_seed)
        z = torch.randn(shape, generator=gen, device=cond.device, dtype=cond.dtype)

        alphas = self.alphas_cumprod
        ts = torch.linspace(
            0, self.train_timesteps - 1, steps, device=cond.device
        ).round().long().flip(0)
        for i, ti in enumerate(ts):
            t_batch = torch.full((b,), int(ti), device=cond.device, dtype=torch.long)
            eps = self.denoiser(z, cond, t_batch)
            a_t = alphas[ti]
            a_prev = alphas[ts[i + 1]] if i + 1 < len(ts) else torch.ones_like(a_t)
            x0 = (z - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-8)
            x0 = x0.clamp(-self.latent_clip, self.latent_clip)
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
        grid = self._grid(h, w)

        # (1) deterministic pixel-space branch -> blurry mean forecast
        det_log = self.deterministic(x_radar)

        # (2) frame-wise autoencoder -> latent conditioning
        z_det = self.autoencoder.encode(det_log, grid)
        nwp_feat = self.nwp_encoder(x_nwp)
        nwp_lat = self.nwp_to_latent(
            F.interpolate(nwp_feat, size=grid, mode="bilinear", align_corners=False)
        )
        cond = torch.cat(
            [z_det, nwp_lat.unsqueeze(1).expand(-1, self.out_len, -1, -1, -1)], dim=2
        )

        # (3) latent diffusion with a DDIM sampler
        z0 = self._ddim_sample(cond, self.sample_steps, grid)
        feat = self.autoencoder.decode(z0, (h, w))

        final_in = torch.cat([feat, det_log], dim=2)
        rain = self.rain_head(final_in.reshape(b * self.out_len, -1, h, w))
        rain = rain.reshape(b, self.out_len, 1, h, w)
        # A hard clamp would cut the gradient wherever the rain is
        # still negative, and the whole model sits on that side of the
        # boundary until it has learned to rain at all.
        return floor_at_zero(rain)
