"""DiffCast baseline adapter.

Paper: "DiffCast: A Unified Framework via Residual Diffusion for Precipitation
Nowcasting", CVPR 2024.  Official implementation:
https://github.com/DeminYu98/DiffCast

Note: the published model is radar-only.  This adapter adds an NWP
conditioning path to the residual denoiser.
"""
from __future__ import annotations

from .model import DiffCastAdapter

__all__ = ["DiffCastAdapter"]
