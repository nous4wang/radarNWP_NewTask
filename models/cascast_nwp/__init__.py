"""CasCast baseline adapter.

Paper: "CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded
Modelling", ICML 2024.  Official implementation:
https://github.com/OpenEarthLab/CasCast
"""
from __future__ import annotations

from .model import CasCastNWPAdapter

__all__ = ["CasCastNWPAdapter"]
