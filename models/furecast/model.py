"""FURECast adapter for the radar + NWP precipitation nowcasting benchmark.

Paper
-----
"FURECast: Advancing Convective Precipitation Nowcasting via 3D Polarimetric
Radar Data and Physics-Constrained Deep Learning Model", Geophysical Research
Letters, 2026.

What this adapter implements
-----------------------------
The paper's architecture is an encoder-translator-decoder network with two
ingredients that matter for the forecast quality:

* an **encoder with late fusion** -- every radar variable gets its own small
  convolutional branch, the branches are only merged at the deepest (most
  semantic) level of each branch, and the fused map is re-weighted by a
  squeeze-and-excitation channel-attention block;
* a **translator** -- a cascade of Inception-style multi-scale blocks that
  evolve an internal state through time.  The state is consumed by a decoder
  that gathers the encoder features, the translator features and the NWP
  fields, and a multi-task head that predicts the rainfall field together with
  the CAPPI altitude profile;
* a **physics constraint** -- a self-consistency regulariser over the CAPPI
  altitude levels, exposed to the trainer as
  :meth:`FURECastAdapter.self_consistency_loss` and weighted by the
  ``physics_weight`` attribute.

Substitutions and limitations
------------------------------
* The paper works on 3D polarimetric radar volumes.  This benchmark provides
  the ten 2D BJ radar composites (CAP20..CAP70, CR, HBR, ET, VIL), so the
  vertical structure is represented by the CAPPI altitude levels rather than by
  a full radar volume.  The CAPPI consistency constraint is applied to the
  CAP20..CAP70 channels accordingly.
* The NWP fields are an addition to the published model, which is radar-only.
  They enter through the shared ``NWPEncoder`` and are fused into the decoder.
* ``common.NWPEncoder`` concatenates a ``hidden``-wide temporal embedding to
  every NWP slice before its first convolution, so it has to be constructed
  with ``in_nwp_c + hidden`` input channels or the channel counts disagree.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    ChannelAttention,
    ConvBlock,
    NWPEncoder,
    RainHead,
    floor_at_zero,
)

#: The CAPPI altitude levels of the BJ radar product, in channel order.
CAPPI_LEVELS = ("CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70")


def _gn_safe(channels: int) -> int:
    """Round a channel count to a value ``common.ConvBlock`` accepts.

    ``ConvBlock`` normalises with ``GroupNorm(min(32, out_ch), out_ch)``,
    which requires the channel count to be a multiple of 32 once it exceeds
    32.  Snapping keeps the adapter usable with any requested width.
    """
    if channels <= 32:
        return max(channels, 1)
    return int(math.ceil(channels / 32.0) * 32)


class DilatedConvBlock(nn.Module):
    """Conv -> group norm -> SiLU with an optional dilation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class InceptionBlock(nn.Module):
    """Multi-scale block: 1x1, 3x3, 5x5 (dilated) and pooled branches, concatenated.

    The dilated 3x3 branch replaces the classic 5x5 convolution while the
    average-pooled branch keeps the resolution, so the block is shape
    preserving for arbitrary feature-map sizes.
    """

    def __init__(self, dim: int, branch_dim: Optional[int] = None) -> None:
        super().__init__()
        b = branch_dim or max(dim // 4, 8)
        self.b1 = DilatedConvBlock(dim, b, kernel_size=1)
        self.b3 = DilatedConvBlock(dim, b, kernel_size=3)
        self.b5 = DilatedConvBlock(dim, b, kernel_size=3, dilation=2)
        self.bp = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            DilatedConvBlock(dim, b, kernel_size=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * b, dim, 1),
            nn.GroupNorm(min(32, dim), dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)
        return self.fuse(y)


class VariableBranch(nn.Module):
    """Late-fusion branch dedicated to a single radar variable."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, width, 3),
            ConvBlock(width, width, 3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` -- ``(N, 1, H, W)``.  Returns ``(N, width, H/2, W/2)``."""
        return self.net(x)


class LateFusionEncoder(nn.Module):
    """Encoder with one branch per radar variable and a SE block on top.

    The branches stay separate until the deepest level of the branch, which
    is where the variables are concatenated and projected to a common width.
    A squeeze-and-excitation block then re-weights the fused channels.
    """

    def __init__(self, n_vars: int, width: int = 8, dim: int = 48) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.branches = nn.ModuleList([VariableBranch(width) for _ in range(n_vars)])
        self.fuse = nn.Sequential(
            ConvBlock(n_vars * width, dim, 1),
            ChannelAttention(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` -- ``(B, T, C, H, W)``.  Returns ``(B, T, dim, H/2, W/2)``."""
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        feats = []
        for i, branch in enumerate(self.branches):
            if i < c:
                feats.append(branch(flat[:, i : i + 1]))
            else:  # pragma: no cover - defensive, one branch per channel
                feats.append(branch(torch.zeros_like(flat[:, :1])))
        y = torch.cat(feats, dim=1)
        y = self.fuse(y)
        return y.reshape(b, t, *y.shape[1:])


class InceptionCascade(nn.Module):
    """Cascade of Inception blocks that refines the translator state."""

    def __init__(self, dim: int, n_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([InceptionBlock(dim) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = x + blk(x)
        return x


class FURECastTranslator(nn.Module):
    """Translator: an internal state that evolves through the radar history.

    The state is updated once per input frame with the cascade of Inception
    blocks.  After the history has been consumed the state is rolled forward
    ``out_len`` times, each step being fed the prediction of the previous
    step, which is how the translator produces the forecast sequence.
    """

    def __init__(
        self, dim: int = 48, enc_dim: int = 48, n_blocks: int = 2, n_cappi: int = 6
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_cappi = n_cappi
        self.enc_proj = nn.Conv2d(enc_dim, dim, 1)
        self.fb_proj = nn.Conv2d(1 + n_cappi, dim, 1)
        self.cascade = InceptionCascade(dim, n_blocks)

    def step(self, state: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """One translator step: inject ``drive`` into ``state`` and refine it."""
        return self.cascade(state + drive)

    def encode_history(
        self, enc: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Consume the encoded radar history, one frame at a time."""
        for t in range(enc.shape[1]):
            state = self.step(state, self.enc_proj(enc[:, t]))
        return state

    def feedback(
        self, rain_log: torch.Tensor, cappi: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        """Drive the next rollout step from the previous prediction.

        The prediction is produced at the radar resolution while the state
        lives at half resolution, so it is resized to the state grid.
        """
        x = torch.cat([rain_log, cappi], dim=1)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.fb_proj(x)


class FURECastDecoder(nn.Module):
    """Decoder that gathers encoder, translator and NWP features.

    The multi-task head produces the rainfall field (through ``RainHead``) and
    the CAPPI altitude profile that the physics constraint operates on.  The
    decoder runs at half resolution and upsamples to the radar grid.
    """

    def __init__(self, dim: int, enc_dim: int, nwp_dim: int, n_cappi: int = 6) -> None:
        super().__init__()
        self.n_cappi = n_cappi
        self.fuse = nn.Sequential(
            ConvBlock(dim + enc_dim + nwp_dim, dim, 3),
            ConvBlock(dim, dim, 3, residual=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(dim, dim, 3),
            ConvBlock(dim, dim, 3, residual=True),
        )
        self.rain_head = RainHead(dim, hidden=32)
        self.cappi_head = nn.Sequential(
            ConvBlock(dim, dim, 3),
            nn.Conv2d(dim, n_cappi, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        enc_last: torch.Tensor,
        nwp: torch.Tensor,
        size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(rain_mm, cappi)`` at the radar grid resolution."""
        x = self.fuse(torch.cat([state, enc_last, nwp], dim=1))
        x = self.up(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        rain = self.rain_head(x)
        cappi = self.cappi_head(x)
        return rain, cappi


@register_model("furecast")
class FURECastAdapter(BasePrecipitationModel):
    """Encoder-translator-decoder nowcasting adapter with a physics constraint.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        dim: feature width of the translator state and the decoder.
        physics_weight: default weight the trainer should give to
            :meth:`self_consistency_loss` when adding it to the total loss.
        n_cappi: number of CAPPI altitude levels predicted by the multi-task head.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        dim: int = 48,
        branch_width: int = 8,
        n_inception: int = 2,
        nwp_hidden: int = 64,
        n_cappi: int = 6,
        physics_weight: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.dim = _gn_safe(dim)
        dim = self.dim
        self.n_cappi = n_cappi
        self.physics_weight = float(physics_weight)
        self._last_cappi: Optional[torch.Tensor] = None

        self.encoder = LateFusionEncoder(in_radar_c, width=branch_width, dim=dim)
        # NWPEncoder appends a `hidden`-wide temporal embedding to each slice.
        self.nwp_encoder = NWPEncoder(in_nwp_c, dim, hidden=nwp_hidden)
        self.nwp_proj = nn.Conv2d(dim, dim, 1)
        self.translator = FURECastTranslator(
            dim=dim, enc_dim=dim, n_blocks=n_inception, n_cappi=n_cappi
        )
        self.decoder = FURECastDecoder(dim, enc_dim=dim, nwp_dim=dim, n_cappi=n_cappi)

    # -- physics constraint ------------------------------------------------
    def self_consistency_loss(self, pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Physics self-consistency penalty over the CAPPI altitude levels.

        The penalty has two parts.

        *Ordering.*  Higher CAPPI levels are assumed to carry larger-scale
        hydrometeor loading, so the level means must not decrease with height.
        The violation of that order is penalised with a soft hinge.

        *Gradient consistency.*  The vertical profile across the CAPPI levels
        must be smooth, which is penalised through the second difference
        along the level axis.

        Args:
            pred: predicted CAPPI stack, ``(B, T, n_cappi, H, W)``.  When
                omitted the prediction cached by the last forward pass is used.

        Returns:
            A finite scalar penalty.  Multiply it by ``physics_weight`` (or
            use the value of that attribute) when building the total loss.
        """
        cappi = pred if pred is not None else self._last_cappi
        if cappi is None:
            return torch.zeros(())
        if cappi.dim() == 4:
            # (B, C, H, W) -> (B, 1, C, H, W)
            cappi = cappi.unsqueeze(1)
        if cappi.dim() != 5:
            raise ValueError(
                "expected CAPPI predictions with shape (B, T, C, H, W), got "
                f"{tuple(cappi.shape)}"
            )
        levels = cappi[:, :, : self.n_cappi]
        if levels.shape[2] < 2:
            return levels.sum() * 0.0

        # Ordering: the level means must not decrease with altitude.
        means = levels.mean(dim=(3, 4))  # (B, T, n_cappi)
        diff = means[:, :, 1:] - means[:, :, :-1]
        ordering = F.softplus(-diff).mean()

        # Gradient consistency: the vertical profile has to be smooth.
        if levels.shape[2] >= 3:
            second = levels[:, :, 2:] - 2.0 * levels[:, :, 1:-1] + levels[:, :, :-2]
            gradient = second.pow(2).mean()
        else:
            gradient = levels.sum() * 0.0
        return ordering + gradient

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
        b, t = x_radar.shape[:2]
        h, w = x_radar.shape[-2:]

        # (a) encoder with late fusion, one branch per radar variable
        enc = self.encoder(x_radar)
        enc_last = enc[:, -1]
        enc_half = enc_last.shape[-2:]

        # NWP features, fused into the decoder
        nwp_feat = self.nwp_proj(self.nwp_encoder(x_nwp))
        nwp_feat = F.interpolate(
            nwp_feat, size=enc_half, mode="bilinear", align_corners=False
        )

        # (b) translator: consume the history, then roll the state forward
        state = enc.new_zeros(b, self.dim, *enc_half)
        state = self.translator.encode_history(enc, state)

        rains: list[torch.Tensor] = []
        cappis: list[torch.Tensor] = []
        for _ in range(self.out_len):
            rain, cappi = self.decoder(state, enc_last, nwp_feat, (h, w))
            rains.append(rain)
            cappis.append(cappi)
            rain_log = torch.log1p(floor_at_zero(rain))
            state = self.translator.step(
                state, self.translator.feedback(rain_log, cappi, enc_half)
            )

        out = torch.stack(rains, dim=1)
        self._last_cappi = torch.stack(cappis, dim=1)
        # The hard clamp cut the gradient wherever the prediction was still
        # negative, so those pixels could never be taught.
        return floor_at_zero(out)
