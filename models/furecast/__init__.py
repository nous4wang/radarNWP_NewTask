"""FURECast baseline adapter.

Paper: "FURECast: Advancing Convective Precipitation Nowcasting via 3D
Polarimetric Radar Data and Physics-Constrained Deep Learning Model",
Geophysical Research Letters, 2026.
"""
from __future__ import annotations

from .model import FURECastAdapter

__all__ = ["FURECastAdapter"]
