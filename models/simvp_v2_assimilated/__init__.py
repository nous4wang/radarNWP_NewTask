"""Route R2: the observed history rainfall as an input, with an operational lag.

The model takes the ground rainfall observed at the issue time as an
input channel, with a configurable delay so that the cost of a late
observation can be measured.

This changes the task: at inference the model consumes ground
observations, so its scores belong in a table of their own.
"""
from __future__ import annotations

from .model import SimVPv2Assimilated

__all__ = ["SimVPv2Assimilated"]
