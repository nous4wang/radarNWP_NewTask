"""FusionCast baseline adapter.

Dual ConvLSTM branches with asymmetric gated fusion and a residual log-rain
decoder.
"""
from __future__ import annotations

from .model import FusionCastAdapter

__all__ = ["FusionCastAdapter"]
