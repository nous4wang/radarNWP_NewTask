"""MMF-RNN adapter for the radar + NWP precipitation nowcasting benchmark.

Paper
-----
"MMF-RNN: A Multimodal Fusion Model for Precipitation Nowcasting", IEEE
Transactions on Geoscience and Remote Sensing, 2025.

What this adapter implements
-----------------------------
The paper fuses two *modalities* -- radar imagery and ground-station gauge
observations -- with a recurrent network and adds a multimodal-consistency
regulariser so that the fused representation does not drift away from the
individual modalities.

This benchmark has no gauge network, so the second modality is **substituted
by the NWP fields**.  The substitution keeps the mechanism intact: two
independent encoders, bidirectional cross-attention fusion with a learned
mixing gate, a recurrent backbone that rolls the state forward in time, and
the two auxiliary losses of the paper.

Concretely:

1. ``RadarBranch`` / ``NWPBranch`` -- two independent convolutional encoders,
   one per modality, each producing a feature map per time step.
2. ``CrossModalFusion`` -- cross-attention in *both* directions
   (``CrossAttention2d``) followed by a learned sigmoid gate that mixes the
   two attended representations.
3. ``ConvGRUCell`` -- the recurrent core.  The paper uses an RNN-style
   backbone; a convolutional GRU is used here because it keeps the spatial
   layout of the radar grid, which the cross-attention fusion relies on.
4. ``multimodal_consistency_loss`` -- penalises the disagreement between the
   single-modality features and the fused feature.
5. ``bdw_loss`` -- the block-based dynamic weighted loss of the paper: hard
   spatial blocks are up-weighted according to their error quantile.

Substitutions and limitations
------------------------------
* Gauges are replaced by NWP fields (see above).
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
    CrossAttention2d,
    NWPEncoder,
    RainHead,
    floor_at_zero,
)


def _gn_safe(channels: int) -> int:
    """Round a channel count to a value ``common.ConvBlock`` accepts.

    ``ConvBlock`` normalises with ``GroupNorm(min(32, out_ch), out_ch)``,
    which requires the channel count to be a multiple of 32 once it exceeds
    32.  Snapping keeps the adapter usable with any requested width.
    """
    if channels <= 32:
        return max(channels, 1)
    return int(math.ceil(channels / 32.0) * 32)


class RadarBranch(nn.Module):
    """Convolutional encoder for the radar modality.

    The features are produced at half resolution: the cross-attention fusion
    and the recurrent core both work on the spatial layout, and running them
    on the full grid costs an order of magnitude more for no accuracy gain at
    the encoder level.  The head restores the radar resolution at the end.
    """

    def __init__(self, in_c: int, dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_c, dim, 3, stride=2),
            ConvBlock(dim, dim, 3, residual=True),
            ConvBlock(dim, dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` -- ``(B, T, C, H, W)``.  Returns ``(B, T, dim, H/2, W/2)``."""
        b, t, c, h, w = x.shape
        y = self.net(x.reshape(b * t, c, h, w))
        return y.reshape(b, t, *y.shape[1:])


class NWPBranch(nn.Module):
    """Convolutional encoder for the NWP modality.

    The two NWP slices are encoded separately and then aggregated by the
    shared ``NWPEncoder``, which keeps the analysis hour and the forecast
    hour distinguishable instead of averaging them away.
    """

    def __init__(self, in_c: int, dim: int = 48, hidden: int = 64) -> None:
        super().__init__()
        # NWPEncoder appends a `hidden`-wide temporal embedding to each slice.
        self.encoder = NWPEncoder(in_c, dim, hidden=hidden)
        self.net = nn.Sequential(
            ConvBlock(dim, dim, 3, stride=2),
            ConvBlock(dim, dim, 3, residual=True),
        )

    def forward(self, x_nwp: torch.Tensor) -> torch.Tensor:
        """Args: ``x_nwp`` -- ``(B, T_nwp, C, H, W)``.  Returns ``(B, dim, H/2, W/2)``."""
        return self.net(self.encoder(x_nwp))


class CrossModalFusion(nn.Module):
    """Bidirectional cross-attention fusion with a learned mixing gate.

    The radar feature queries the NWP feature and the NWP feature queries the
    radar feature; the two attended maps are then mixed by a per-channel
    sigmoid gate, so the model can decide per channel and per pixel how much
    each modality contributes.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.radar_to_nwp = CrossAttention2d(dim, dim)
        self.nwp_to_radar = CrossAttention2d(dim, dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, f_radar: torch.Tensor, f_nwp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the fused map and the mixing gate."""
        r2n = self.radar_to_nwp(f_radar, f_nwp)
        n2r = self.nwp_to_radar(f_nwp, f_radar)
        g = self.gate(torch.cat([r2n, n2r], dim=1))
        return g * r2n + (1.0 - g) * n2r, g


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell.

    A GRU over feature maps instead of vectors: the gates are computed with
    3x3 convolutions so the hidden state keeps the spatial layout of the
    radar grid.  This is the recurrent core of the adapter (see the module
    docstring for why a ConvGRU is used in place of the paper's plain RNN).
    """

    def __init__(self, in_ch: int, hidden_ch: int) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gate_conv = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, 3, padding=1)
        self.cand_conv = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: ``x`` -- ``(B, C_in, H, W)``, ``h`` -- ``(B, C_hidden, H, W)``."""
        gates = self.gate_conv(torch.cat([x, h], dim=1))
        z, r = gates.chunk(2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        cand = torch.tanh(self.cand_conv(torch.cat([x, r * h], dim=1)))
        h_new = (1.0 - z) * h + z * cand
        return h_new, h_new


@register_model("mmf_rnn")
class MMFRNNAdapter(BasePrecipitationModel):
    """Dual-branch multimodal fusion + ConvGRU nowcasting adapter.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        dim: feature width of both branches and of the recurrent state.
        hidden: hidden width of the recurrent state.
    """

    #: Extra weight multiplier applied to the hard blocks by ``bdw_loss``.
    _bdw_boost = 3.0

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        dim: int = 48,
        hidden: int = 48,
        nwp_hidden: int = 64,
        **kwargs: object,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.dim = _gn_safe(dim)
        self.hidden = _gn_safe(hidden)
        dim = self.dim
        hidden = self.hidden
        self.radar_branch = RadarBranch(in_radar_c, dim)
        self.nwp_branch = NWPBranch(in_nwp_c, dim, hidden=nwp_hidden)
        self.fusion = CrossModalFusion(dim)
        self.cell = ConvGRUCell(dim, hidden)
        # A learned "future frame" embedding lets the recurrent core advance
        # beyond the observed history without feeding it real observations.
        self.time_embed = nn.Parameter(torch.zeros(1, out_len, hidden))
        nn.init.normal_(self.time_embed, std=0.02)
        self.head = nn.Sequential(
            ConvBlock(hidden, dim, 3),
            ConvBlock(dim, dim, 3, residual=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(dim, dim, 3),
        )
        self.rain_head = RainHead(dim, hidden=32)

    # -- internals --------------------------------------------------------
    def _encode(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """Per-frame fused features, ``(B, T, dim, H, W)``."""
        f_radar = self.radar_branch(x_radar)
        f_nwp = self.nwp_branch(x_nwp)
        if f_nwp.shape[-2:] != f_radar.shape[-2:]:
            f_nwp = F.interpolate(
                f_nwp, size=f_radar.shape[-2:], mode="bilinear", align_corners=False
            )
        fused = []
        for t in range(f_radar.shape[1]):
            ft, _ = self.fusion(f_radar[:, t], f_nwp)
            fused.append(ft)
        return torch.stack(fused, dim=1)

    # -- auxiliary losses -------------------------------------------------
    def multimodal_consistency_loss(
        self, x_radar: torch.Tensor, x_nwp: torch.Tensor
    ) -> torch.Tensor:
        """Penalise disagreement between the single-modality and fused features.

        The fused map is a gated blend of two cross-attended maps: one in
        which the radar feature queries the NWP feature, one in which the NWP
        feature queries the radar feature.  Each of those carries the content
        of a single modality, so the fused map must agree with both of them;
        otherwise cross-attention silently drops one modality.  The
        disagreement is measured as a per-pixel cosine distance.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``.

        Returns:
            A finite scalar penalty.
        """
        x_nwp = self._check_shapes(x_radar, x_nwp)
        f_radar = self.radar_branch(x_radar)[:, -1]
        f_nwp = self.nwp_branch(x_nwp)
        if f_nwp.shape[-2:] != f_radar.shape[-2:]:
            f_nwp = F.interpolate(
                f_nwp, size=f_radar.shape[-2:], mode="bilinear", align_corners=False
            )
        fused, _ = self.fusion(f_radar, f_nwp)
        # The two single-modality contributions the fused map is built from.
        r2n = self.fusion.radar_to_nwp(f_radar, f_nwp)
        n2r = self.fusion.nwp_to_radar(f_nwp, f_radar)

        def _disagreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """1 - cosine similarity per pixel between two feature maps."""
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            return (1.0 - (a * b).sum(dim=1)).mean()

        return _disagreement(fused, r2n) + _disagreement(fused, n2r)

    def bdw_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        block: int = 8,
        quantile: float = 0.75,
    ) -> torch.Tensor:
        """Block-based dynamic weighted loss.

        The field is split into square blocks.  The per-block error of the
        hard blocks -- those above the ``quantile`` of the block-error
        distribution -- receives a larger weight, so the loss concentrates on
        the parts of the domain the model gets wrong.

        Args:
            pred: prediction, ``(B, T, 1, H, W)`` or ``(B, T, H, W)``.
            target: target with the same shape as ``pred``.
            block: side length of the blocks in pixels.
            quantile: error quantile above which a block counts as hard.

        Returns:
            A finite scalar loss.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, got "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.dim() < 4:
            raise ValueError(
                "pred and target must have at least 4 dimensions "
                f"(B, T, [C], H, W), got {tuple(pred.shape)}"
            )
        # Normalise to (N, 1, H, W) with N = B * T frames.
        p = pred.reshape(-1, 1, *pred.shape[-2:])
        t = target.reshape(-1, 1, *target.shape[-2:])
        h, w = p.shape[-2:]
        bh = max(1, h // block)
        bw = max(1, w // block)
        # Crop to a multiple of the block size, then fold the field into blocks.
        p = p[..., : bh * block, : bw * block]
        t = t[..., : bh * block, : bw * block]
        p = p.reshape(p.shape[0], 1, bh, block, bw, block)
        t = t.reshape(t.shape[0], 1, bh, block, bw, block)
        err = (p - t).abs().mean(dim=(1, 3, 5))  # (N, bh, bw)
        flat = err.reshape(err.shape[0], -1)
        thresh = torch.quantile(flat, quantile, dim=1, keepdim=True)
        hard = (flat > thresh).to(flat.dtype)
        weights = 1.0 + self._bdw_boost * hard
        weights = weights.reshape(err.shape)
        weights = weights / weights.mean().clamp_min(1e-6)
        per_block = err * weights
        return per_block.mean()

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
        h_out, w_out = x_radar.shape[-2:]
        fused = self._encode(x_radar, x_nwp)

        # Recurrent core over the observed history.
        h = fused.new_zeros(b, self.hidden, *fused.shape[-2:])
        for t in range(fused.shape[1]):
            h, _ = self.cell(fused[:, t], h)

        # Roll the state forward to produce the forecast frames.
        last = fused[:, -1]
        outs = []
        for step in range(self.out_len):
            h, _ = self.cell(last + self.time_embed[:, step].view(1, -1, 1, 1), h)
            feat = self.head(h)
            feat = self.up(feat)
            if feat.shape[-2:] != (h_out, w_out):
                feat = F.interpolate(
                    feat, size=(h_out, w_out), mode="bilinear", align_corners=False
                )
            outs.append(self.rain_head(feat))
        out = torch.stack(outs, dim=1)
        # The hard clamp cut the gradient wherever the prediction was
        # still negative, so those pixels could never be taught.
        return floor_at_zero(out)
