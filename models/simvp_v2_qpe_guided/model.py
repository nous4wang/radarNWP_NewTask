"""Route R1: the QPE-QPF split, with the history rainfall as a teacher.

The end-to-end model is asked to learn two different things at once. The
first is the inversion -- turning the radar's reflectivity into the rain
rate that reached the ground -- and the measured correlation between the
two at the same instant is +0.17, so the inversion is hard on its own.
The second is the advection -- carrying the rain forward in space and
predicting its growth and decay -- and the future frames carry a
displacement error of a few pixels that the loss charges twice, once as
a false alarm and once as a miss.

The two fight each other. What the archive offers is a way to teach the
first without the second: the radar and the ground rainfall share their
timestamps, so the eleven history instants have an observed rainfall
each, with no displacement between the radar and the truth.

So the model is split. A small branch inverts the radar and the NWP
state into a rainfall field for each of the eleven history frames, and
that estimate is supervised directly against the observed history
rainfall. The main branch then works in the space of that estimate, and
the two branches share one rainfall head.

The shared head is the point. The head that produces the history
estimate is supervised against observations that carry no displacement
error, so it is anchored to the true rain distribution and keeps its
sharpness. The head that produces the future forecast is the same head,
so it inherits that anchoring instead of learning to hedge.

At inference the model takes exactly what every other baseline takes:
the radar history and the NWP state. The observed history rainfall is
used during training only, which is what makes the comparison with the
other baselines a fair one.
"""
from __future__ import annotations

import torch

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import QPEQPFBackbone


@register_model("simvp_v2_qpe_guided")
class SimVPv2QPEGuided(BasePrecipitationModel):
    """Route R1: the history rainfall supervises the inversion.

    The inference-time contract is unchanged -- the model takes the radar
    history and the NWP state, and nothing else. The observed history
    rainfall reaches the loss through ``last_qpe_pred``, which the
    training objective picks up and supervises against ``y_hist``.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
    ):
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.backbone = QPEQPFBackbone(
            in_radar_c, in_nwp_c, in_len, out_len, hid_c
        )
        self.out_len = out_len

    @property
    def last_qpe_pred(self) -> torch.Tensor | None:
        """The inverted history rainfall, for the objective to read."""
        return self.backbone.last_qpe_pred

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
        qpe_feat, qpe_rain = self.backbone.invert(x_radar, x_nwp[:, 0])
        return self.backbone.extrapolate(
            x_radar, qpe_feat, qpe_rain, x_nwp[:, -1]
        )
