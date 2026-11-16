"""Route R4: a semi-Lagrangian advection with a source and sink residual.

The diagnosis this model answers is the one the master document reached
by three independent routes: a scalar regression in log space
under-predicts the rain, under-prediction flattens the small scales, and
the flattened small scales are what the heavy-rain score collapses on.
The spectrum measurements put the survival of the sub-ten-kilometre
wavelengths at 0.9 to 26 per cent.

A displacement-based model attacks that directly. The rainfall field is
carried forward by a motion field rather than re-predicted pixel by
pixel, so the peak arrives at the peak instead of being averaged away.

Two things about this model have to be said plainly, because the
proposal it replaces did not say them.

The first is that a bilinear resampling is **not conservative**. It
dilutes the dense cells and fills the sparse ones, so a "spectrum
preserving" warp is not one. The claim has to be measured rather than
asserted, and the measurement is part of the protocol: the total
rainfall before and after the warp is recorded for every patch.

The second is that the network can ignore the advection entirely. If
the source and sink residual is added on top without a gate, the
cheapest way to reduce the loss is to set the residual to the whole
answer and the motion field to nothing, and the model degenerates into
the baseline it was meant to beat. So there is a gate, and the gate is
reported: the mean weight the model puts on the advection is a direct
measurement of whether it is using it.

Three arms share these weights and differ only in which parts are
switched on:

``use_advection`` off   the residual alone -- the baseline behaviour.
``use_source_sink`` off the advection alone -- the physical part on its own.
both on                 the full model, with the gate between them.

The inference-time contract is the one every baseline shares: the model
takes the radar history and the NWP state, and nothing else.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    QPEQPFBackbone,
    floor_at_zero,
    make_warp_grid,
)


@register_model("simvp_v2_lagrangian")
class SimVPv2Lagrangian(BasePrecipitationModel):
    """Advection plus a learned source and sink, with a gate between them.

    Args:
        use_advection: whether the motion field is used at all. Off, the
            model is the residual alone.
        use_source_sink: whether the residual is used. Off, the model is
            the advection alone.
        max_disp: the largest displacement the motion field may ask
            for, in pixels per frame. A frame is six minutes, so a limit
            of six pixels is thirty-six kilometres per hour.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        use_advection: bool = True,
        use_source_sink: bool = True,
        max_disp: float = 6.0,
    ):
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.backbone = QPEQPFBackbone(
            in_radar_c, in_nwp_c, in_len, out_len, hid_c
        )
        self.out_len = out_len
        self.use_advection = bool(use_advection)
        self.use_source_sink = bool(use_source_sink)
        self.max_disp = float(max_disp)

        # The three heads sit on the future features the backbone already
        # computes, so the encoder and the time translator are shared with
        # the baseline and the comparison is about the last step.
        #
        # The motion head emits two channels. The order is fixed here and
        # not left to the training: ``make_warp_grid`` reads the flow as
        # (dy, dx), and a model that emitted them the other way round
        # would advect along the axes without any error message.
        self.flow_head = nn.Sequential(
            nn.Conv2d(hid_c, hid_c // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid_c // 2, 2, kernel_size=1),
        )
        # The residual is a change in the rain rate, so it is free to be
        # negative.
        self.source_sink_head = nn.Sequential(
            nn.Conv2d(hid_c, hid_c // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid_c // 2, 1, kernel_size=1),
        )
        # The gate says how much of this frame comes from the motion and
        # how much from the residual.
        self.gate_head = nn.Sequential(
            nn.Conv2d(hid_c, hid_c // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid_c // 2, 1, kernel_size=1),
        )

        #: The mean weight the model put on the advection, from the last
        #: forward pass. Reported so that "the model uses the physics" is
        #: a measurement rather than an assumption.
        self.last_gate_mean: float | None = None
        #: The mean rainfall total before and after the warp, from the
        #: last forward pass. The ratio is one for a conservative warp.
        self.last_mass_ratio: float | None = None

    @property
    def last_qpe_pred(self) -> torch.Tensor | None:
        """The inverted history rainfall, for the objective to read."""
        return self.backbone.last_qpe_pred

    def _future_features(
        self,
        x_radar: torch.Tensor,
        qpe_feat_flat: torch.Tensor,
        history_rain: torch.Tensor,
        nwp_forecast: torch.Tensor,
    ) -> torch.Tensor:
        """The future features, from the backbone's own encoder.

        This is the backbone's extrapolation path up to the point where
        the rainfall head is applied, so the advection model and the
        baseline see the same encoding of the same input.
        """
        b, t_in, _, h, w = x_radar.shape
        backbone = self.backbone
        qpe_feat = qpe_feat_flat.view(b, t_in, backbone.hid_c, h, w)
        rain_log = torch.log1p(history_rain.clamp_min(0.0))
        nwp_exp = nwp_forecast.unsqueeze(1).expand(-1, t_in, -1, -1, -1)

        qpf_in = torch.cat([qpe_feat, rain_log, x_radar, nwp_exp], dim=2)
        down_feat = backbone.spatial_down(qpf_in.reshape(b * t_in, -1, h, w))
        _, c_down, h_d, w_d = down_feat.shape

        fut_down = backbone.translator(
            down_feat.view(b, t_in * c_down, h_d, w_d)
        )
        fut_down = backbone.out_time_proj(fut_down).view(
            b * self.out_len, backbone.hid_c * 2, h_d, w_d
        )
        return backbone.spatial_up(fut_down)

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast the next ten six-minute accumulations.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)`` or ``None`` when ablated.

        Returns:
            ``(B, T_out, 1, H, W)`` in millimetres per 6-minute
            accumulation.
        """
        x_nwp = self.resolve_nwp(x_radar, x_nwp)
        b, t_in, _, h, w = x_radar.shape

        qpe_feat, qpe_rain = self.backbone.invert(x_radar, x_nwp[:, 0])
        history_rain = qpe_rain.clamp_min(0.0)
        feats = self._future_features(
            x_radar, qpe_feat, history_rain, x_nwp[:, -1]
        )

        flow = self.flow_head(feats).view(b, self.out_len, 2, h, w)
        flow = flow.clamp(-self.max_disp, self.max_disp)
        delta = self.source_sink_head(feats).view(b, self.out_len, 1, h, w)
        gate = torch.sigmoid(self.gate_head(feats)).view(
            b, self.out_len, 1, h, w
        )

        if not self.use_advection:
            gate = torch.zeros_like(gate)
        if not self.use_source_sink:
            delta = torch.zeros_like(delta)
            gate = torch.ones_like(gate)

        # The starting field is the inverted rainfall at the issue time.
        # It is the same field the baseline extrapolates from, so the two
        # differ only in how the future is produced.
        r0 = history_rain[:, -1]

        preds = []
        mass_before = r0.sum()
        mass_after = r0.new_zeros(())
        for step in range(self.out_len):
            if self.use_advection:
                grid = make_warp_grid(flow[:, step], (h, w))
                advected = F.grid_sample(
                    r0,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            else:
                advected = torch.zeros_like(r0)
            g = gate[:, step]
            blended = g * advected + (1.0 - g) * delta[:, step]
            preds.append(floor_at_zero(blended))
            mass_after = mass_after + blended[:, 0].sum()

        self.last_gate_mean = float(gate.detach().mean())
        if float(mass_before.detach()) > 1e-6:
            self.last_mass_ratio = float(
                (mass_after / (self.out_len * mass_before)).detach()
            )

        return torch.stack(preds, dim=1)
