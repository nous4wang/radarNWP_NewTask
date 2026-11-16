"""Route R2: the observed history rainfall as an input.

In many operational settings the rainfall at the issue time is already
known -- from a gauge network, from a merged radar-gauge product, from
the analysis step of a numerical model. When it is, a nowcast that
ignores it is working with less than it has.

This model takes the observed history rainfall as an input channel and
asks the question an operational reader would ask: **how much does that
help when the observation is late?**

The lag is measured in frames of six minutes. With ``obs_lag_steps = 0``
the whole eleven-frame history is observed, which is the upper bound.
With ``obs_lag_steps = 2`` the last twelve minutes are missing and the
inversion branch fills them in. With ``obs_lag_steps = 11`` nothing is
observed at inference and the model degenerates to route R1.

The filling is the point. A model that simply took the observations
would collapse the moment they stopped arriving; this one falls back on
its own inversion, and the lag sweep measures what that fallback is
worth.

**This changes the task.** At inference the model consumes ground
observations, so its scores are not comparable with a pure radar
extrapolation. It is a different setting, it is reported separately, and
its baseline is another member of this family rather than the radar
baselines.
"""
from __future__ import annotations

import torch

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import QPEQPFBackbone


@register_model("simvp_v2_assimilated")
class SimVPv2Assimilated(BasePrecipitationModel):
    """Route R2: assimilate the observed history rainfall, with a lag.

    Args:
        obs_lag_steps: how many of the most recent history frames are
            unavailable at inference. Each frame is six minutes, so 2 is
            a twelve-minute delay and 4 is twenty-four. The value 11
            means no observation is used at all.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        obs_lag_steps: int = 0,
    ):
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.backbone = QPEQPFBackbone(
            in_radar_c, in_nwp_c, in_len, out_len, hid_c
        )
        self.out_len = out_len
        self.obs_lag_steps = obs_lag_steps

        # The observed history rainfall, published by the evaluation and
        # the training loop. The model reads it from here rather than
        # from its forward signature, because the signature is fixed by
        # the seam every baseline shares: ``model(x_radar, x_nwp)``.
        # Changing that would make the model unevaluable.
        self.obs_y_hist: torch.Tensor | None = None

    @property
    def last_qpe_pred(self) -> torch.Tensor | None:
        """The inverted history rainfall, for the objective to read."""
        return self.backbone.last_qpe_pred

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor | None = None,
        y_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast the next ten six-minute accumulations.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)`` or ``None`` when ablated.
            y_hist: the observed history rainfall, when the caller has
                it. The evaluation passes it through ``obs_y_hist``
                instead, so that the shared seam stays intact.

        Returns:
            ``(B, T_out, 1, H, W)`` in millimetres per 6-minute
            accumulation.
        """
        x_nwp = self.resolve_nwp(x_radar, x_nwp)
        b, t_in, _, h, w = x_radar.shape

        # The inversion branch runs first, because it is what fills the
        # lagged slots.
        qpe_feat, qpe_rain = self.backbone.invert(x_radar, x_nwp[:, 0])

        observed = y_hist if y_hist is not None else self.obs_y_hist
        if self.obs_lag_steps >= t_in:
            # No observation is available at all: every slot is older than
            # the lag allows. The history rainfall channel is zeroed rather
            # than filled with the inverted estimate, because filling it
            # makes this arm a copy of the inversion-only arm -- the two
            # then have identical weights, and a sweep over the lag has
            # only three distinct arms instead of four.
            #
            # Zeroing is also the honest statement: the model is told the
            # rainfall history is unavailable, and it has the radar to work
            # with. It is not the same as a history of zero rain, but the
            # difference cannot be expressed without a separate flag
            # channel, and adding one would change every other arm and
            # invalidate the baselines that were measured against it.
            history_rain = torch.zeros_like(qpe_rain)
        elif observed is None:
            # The caller supplied no observation. The history channel is
            # empty, which is the same statement the no-observation arm
            # makes. The evaluation is where a missing publication is
            # caught: it knows which arm expects an observation, and it
            # refuses to score an arm that was quietly given none.
            history_rain = torch.zeros_like(qpe_rain)
        else:
            if observed.shape[1] != t_in:
                raise ValueError(
                    f"the observed history has {observed.shape[1]} frames "
                    f"but the radar history has {t_in}"
                )
            valid_end = t_in - self.obs_lag_steps
            history_rain = torch.cat(
                [
                    observed[:, :valid_end].clamp_min(0.0),
                    qpe_rain[:, valid_end:].clamp_min(0.0),
                ],
                dim=1,
            )

        return self.backbone.extrapolate(
            x_radar, qpe_feat, history_rain, x_nwp[:, -1]
        )
