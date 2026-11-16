"""Route R1: the QPE-QPF split, with the history rainfall as a teacher.

The model inverts the radar history into a rainfall field and then
extrapolates from it, with one rainfall head shared between the two
stages so that the zero-displacement supervision on the history frames
anchors the forecast.

The inference-time contract is unchanged: the model takes the radar
history and the NWP state, and nothing else.
"""
from __future__ import annotations

from .model import SimVPv2QPEGuided

__all__ = ["SimVPv2QPEGuided"]
