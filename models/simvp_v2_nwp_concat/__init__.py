"""SimVP-v2 with the NWP concatenated onto the radar input.

Ablation row M1 of the NWP-injection study: the plainest way of giving
the model the NWP is to paste the fields onto the input channels and let
the first convolution work out what they mean.
"""
from __future__ import annotations

from .model import SimVPv2NWPConcat

__all__ = ['SimVPv2NWPConcat']
