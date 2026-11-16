"""SimVP baseline (two variants).

SimVP predicts the next state with an MLP-Mixer-style core instead of a
recurrent cell. Two variants are registered: one on the radar alone, and
one that concatenates the NWP at the input.
"""
from __future__ import annotations

from .model import SimVPRadarOnly
from .model import SimVPEarlyConcat
from .model import SimVP_Core
from .model import BasicConv2d

__all__ = ['SimVPRadarOnly', 'SimVPEarlyConcat', 'SimVP_Core', 'BasicConv2d']
