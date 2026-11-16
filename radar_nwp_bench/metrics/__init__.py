"""Meteorological verification metrics and training losses."""
from .losses import PrecipitationLoss, WeightedCRPSLoss, inverse_mse, weighted_bce
from .precipitation import (
    DEFAULT_THRESHOLDS,
    PrecipitationVerifier,
    VerificationCounts,
    contingency_counts,
    fractions_skill_score,
    verify,
)
from .stitching import PatchStitcher

__all__ = [
    "PrecipitationLoss",
    "WeightedCRPSLoss",
    "inverse_mse",
    "weighted_bce",
    "DEFAULT_THRESHOLDS",
    "PrecipitationVerifier",
    "VerificationCounts",
    "contingency_counts",
    "fractions_skill_score",
    "verify",
    "PatchStitcher",
]
