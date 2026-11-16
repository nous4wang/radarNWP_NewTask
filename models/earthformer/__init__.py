"""EarthFormer baseline.

A spatio-temporal transformer that factorises the attention into a temporal
branch over the frame axis and a spatial branch over the grid.
"""
from __future__ import annotations

from .model import EarthformerModel
from .model import EarthformerBlock
from .model import CuboidSelfAttention

__all__ = ['EarthformerModel', 'EarthformerBlock', 'CuboidSelfAttention']
