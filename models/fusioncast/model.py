"""FusionCast adapter: asymmetric cross-modal fusion with future radar priors.

Paper: "FusionCast: Enhancing Precipitation Nowcasting with Asymmetric
Cross-Modal Fusion and Future Radar Priors" (arXiv:2603.13298).

The adapter implements the three mechanisms of the paper:

* **Two ConvLSTM branches.**  A radar branch and a NWP (PWV) branch each keep
  their own hidden and cell state and are updated with a ConvLSTM cell
  (forget / input / output gates computed by convolutions).
* **Asymmetric gated fusion.**  The NWP branch receives a *spatial* attention
  gate (channels are pooled and a convolution produces a per-pixel gate),
  while the radar branch receives a *channel* attention gate (global average
  and max pooling feed an MLP that produces a per-channel gate).  The two
  gates are multiplied elementwise, a residual is added, and the fusion is
  applied to both the hidden state and the cell state.
* **Residual log-rain decoder.**  The rainfall is decoded in log space,

      log1p(y_next) = log1p(cur) + max_log_delta * tanh(delta / max_log_delta),

  with ``max_log_delta = 4.0``, so that the decoder starts from a persistence
  behaviour and can only move the rainfall by a bounded amount in log space.

Decoding is autoregressive: every step feeds the previous radar frame back
into the radar branch (self-conditioning), which is what the "future radar
prior" of the paper refers to.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    ConvBlock,
    MLP,
    NWPEncoder,
    RainHead,
    floor_at_zero,
)



def init_rain_head(head: RainHead) -> None:
    """Bias the rain head so that it starts close to the identity on rain.

    The training target is ``log1p(mm)``, so the head predicts in log space
    and exponentiates at the end.  Starting the log-space output near zero
    means the initial prediction is a small rainfall rather than an arbitrary
    value, which keeps the early training signal well conditioned.
    """
    final = head.net[-1]
    if isinstance(final, nn.Conv2d) and final.bias is not None:
        nn.init.zeros_(final.bias)
        nn.init.normal_(final.weight, std=0.01)


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell: forget / input / output gates plus a candidate."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cell state.

        Args:
            x: ``(B, C_in, H, W)`` input.
            h: ``(B, C_hidden, H, W)`` hidden state.
            c: ``(B, C_hidden, H, W)`` cell state.

        Returns:
            ``(h_new, c_new)``.
        """
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class AsymmetricGatedFusion(nn.Module):
    """Asymmetric gated fusion of the radar and NWP branches.

    The NWP branch is gated *spatially* and the radar branch is gated
    *channel-wise*.  The two gates are multiplied elementwise, a residual is
    added, and the fused hidden / cell states are returned.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Spatial gate on the NWP branch: pool channels, then a conv.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Channel gate on the radar branch: global average + max pool, then MLP.
        self.channel_gate = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.residual = nn.Conv2d(2 * channels, channels, 1)

    def forward(
        self,
        h_radar: torch.Tensor,
        c_radar: torch.Tensor,
        h_nwp: torch.Tensor,
        c_nwp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse the two branches.

        Args:
            h_radar: ``(B, C, H, W)`` radar hidden state.
            c_radar: ``(B, C, H, W)`` radar cell state.
            h_nwp: ``(B, C, H, W)`` NWP hidden state.
            c_nwp: ``(B, C, H, W)`` NWP cell state.

        Returns:
            ``(fused_hidden, fused_cell)``.
        """
        # Spatial gate from the NWP branch.
        nwp_stats = torch.cat(
            [h_nwp.mean(dim=1, keepdim=True), h_nwp.amax(dim=1, keepdim=True)], dim=1
        )
        g_spatial = self.spatial_gate(nwp_stats)  # (B, 1, H, W)

        # Channel gate from the radar branch.
        radar_stats = torch.cat(
            [h_radar.mean(dim=(2, 3)), h_radar.amax(dim=(2, 3))], dim=1
        )
        g_channel = self.channel_gate(radar_stats)  # (B, C)
        g_channel = g_channel.unsqueeze(-1).unsqueeze(-1)

        # Elementwise product of the two asymmetric gates.
        gate = g_spatial * g_channel

        fused_hidden = gate * h_radar + self.residual(torch.cat([h_radar, h_nwp], dim=1))
        fused_cell = gate * c_radar + self.residual(torch.cat([c_radar, c_nwp], dim=1))
        return fused_hidden, fused_cell


@register_model("fusioncast")
class FusionCastAdapter(BasePrecipitationModel):
    """FusionCast adapter for the radar + NWP precipitation benchmark.

    Args:
        in_radar_c: number of radar channels per history frame.
        in_nwp_c: number of NWP channels per NWP frame.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        channels: width of the ConvLSTM states.
        hidden: width of the internal convolutions.
        max_log_delta: bound of the log-space rainfall increment.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        channels: int = 32,
        hidden: int = 32,
        max_log_delta: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.channels = channels
        self.max_log_delta = max_log_delta

        # Radar branch: the encoder turns radar frames into ConvLSTM inputs.
        self.radar_encoder = ConvBlock(in_radar_c, channels, 3)
        self.radar_cell = ConvLSTMCell(channels, channels)
        # Maps a predicted rainfall frame back into the latent space, so the
        # predicted frame can be fed back as the next radar prior.
        self.rain_to_radar = nn.Conv2d(1, channels, 1)

        # NWP branch.
        self.nwp_encoder = NWPEncoder(in_nwp_c, channels, hidden=hidden)
        self.nwp_cell = ConvLSTMCell(channels, channels)

        self.fusion = AsymmetricGatedFusion(channels)

        # Decoder: the fused state produces the log-space rainfall increment.
        self.delta_head = nn.Sequential(
            ConvBlock(channels, hidden, 3),
            MLP(hidden, ratio=2),
            nn.Conv2d(hidden, 1, 1),
        )
        # The persistence anchor: the rainfall estimated from the current radar
        # frame.  RainHead exponentiates, so it returns millimetres.
        self.rain_head = RainHead(channels, hidden=hidden)
        init_rain_head(self.rain_head)
        # Start the decoder as a pure persistence model: with delta == 0 the
        # log-space residual vanishes and every frame equals the current one.
        #
        # Only the *bias* may be zeroed. Zeroing the weight as well would
        # make the increment identically zero for every input, and since
        # delta = delta_head(fused_hidden), the zero gradient would propagate
        # back through the whole fusion branch: the asymmetric gates, both
        # ConvLSTM cells and the NWP encoder were left with no gradient at
        # all, so the model could only ever learn the direct encoder-to-head
        # path. Zeroing the bias gives the same starting point and keeps the
        # gradient alive.
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor,
        gt_rain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict future accumulated rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``.
            gt_rain: optional ground-truth rainfall ``(B, T_out, 1, H, W)`` in
                millimetres, used for teacher forcing during training.  It
                stands in for the "future radar prior" of the paper: this
                benchmark has no future radar observation, so the future
                rainfall is what the decoder is conditioned on.  It is mapped
                into the latent space by the same projection the model uses
                for its own predictions, so the two paths stay consistent.
                When it is omitted the decoder is self-conditioned on its own
                previous output.

        Returns:
            ``(B, T_out, 1, H, W)`` non-negative rainfall in millimetres.
        """
        self._check_shapes(x_radar, x_nwp)
        b, t, _, h, w = x_radar.shape
        device = x_radar.device

        # --- encode the inputs -------------------------------------------
        radar_feats = self.radar_encoder(
            x_radar.reshape(b * t, *x_radar.shape[2:])
        ).reshape(b, t, self.channels, h, w)

        # The NWP encoder produces one fused field; the NWP branch is updated
        # with it at every step so that it always has a valid guidance.
        nwp_feat = self.nwp_encoder(x_nwp)  # (B, C, H, W)
        if nwp_feat.shape[-2:] != (h, w):
            nwp_feat = F.interpolate(
                nwp_feat, size=(h, w), mode="bilinear", align_corners=False
            )

        # --- run the two ConvLSTM branches over the history ---------------
        h_radar = torch.zeros(b, self.channels, h, w, device=device, dtype=x_radar.dtype)
        c_radar = torch.zeros_like(h_radar)
        h_nwp = torch.zeros_like(h_radar)
        c_nwp = torch.zeros_like(h_radar)

        for step in range(t):
            h_radar, c_radar = self.radar_cell(radar_feats[:, step], h_radar, c_radar)
            h_nwp, c_nwp = self.nwp_cell(nwp_feat, h_nwp, c_nwp)

        # --- autoregressive residual log-rain decoding --------------------
        # The current rainfall estimate comes from the last radar frame: it is
        # the persistence anchor of the decoder.
        # A hard floor here would cut the gradient from the rain head
        # and, through it, from the whole encoder.
        current = floor_at_zero(self.rain_head(radar_feats[:, -1]))

        frames: list[torch.Tensor] = []
        for step in range(self.out_len):
            # The asymmetric fusion acts on both the hidden and the cell state.
            fused_hidden, fused_cell = self.fusion(h_radar, c_radar, h_nwp, c_nwp)
            delta = self.delta_head(fused_hidden)

            # log1p(y_next) = log1p(cur) + max_log_delta * tanh(delta / max_log_delta)
            log_next = torch.log1p(current) + self.max_log_delta * torch.tanh(
                delta / self.max_log_delta
            )
            # The floor is straight-through so that a frame which dips below
            # zero rain can still be pulled back up by the loss.
            current = floor_at_zero(torch.expm1(log_next))
            frames.append(current)

            # The fused states carry the cross-modal information forward.
            h_radar, c_radar = fused_hidden, fused_cell

            # The decoded frame becomes the next radar prior, so the decoder
            # keeps its own trajectory (the "future radar prior" of the paper).
            # With teacher forcing the observed rainfall is used instead.
            if gt_rain is not None and step < gt_rain.shape[1]:
                next_feat = self.rain_to_radar(gt_rain[:, step])
            else:
                next_feat = self.rain_to_radar(current)
            h_radar, c_radar = self.radar_cell(next_feat, h_radar, c_radar)
            h_nwp, c_nwp = self.nwp_cell(nwp_feat, h_nwp, c_nwp)

        out = torch.stack(frames, dim=1)
        return floor_at_zero(out)
