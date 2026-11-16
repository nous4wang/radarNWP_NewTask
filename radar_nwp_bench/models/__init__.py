"""The model seam: one interface, many baselines.

Every baseline implements the same contract, so the training loop, the
verification metrics and the data pipeline never learn a model's
particularities. That is what makes a comparison across eight architectures
meaningful: the models differ, the harness does not.

    forward(x_radar, x_nwp) -> y_rain

        x_radar : (B, T_in, C_radar, H, W) float32
            Radar history. T_in frames of 6 minutes, C_radar physical channels.
        x_nwp   : (B, T_nwp, C_nwp, H, W) float32
            NWP fields already resampled onto the radar grid and aligned by
            valid time.
        returns : (B, T_out, 1, H, W) float32
            Predicted 6-minute accumulated ground rainfall in millimetres.

Baselines are reached by name, never by import:

    from radar_nwp_bench.models import build_model, list_models
    model = build_model("mocast", in_radar_c=10, in_nwp_c=27)

A new baseline registers itself and needs no change here -- see
:mod:`radar_nwp_bench.models.registry`.
"""
from .base import BasePrecipitationModel
from .registry import (
    build_model,
    list_models,
    model_class,
    register_model,
    registry_snapshot,
)

__all__ = [
    "BasePrecipitationModel",
    "build_model",
    "list_models",
    "model_class",
    "register_model",
    "registry_snapshot",
]
