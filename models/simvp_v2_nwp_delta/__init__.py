"""SimVP-v2 with the NWP difference field in place of the NWP.

Ablation row M3 of the NWP-injection study: the model sees how the NWP
changes between its analysis hour and its forecast hour, and nothing else
of the NWP.
"""
from __future__ import annotations

from .model import SimVPv2NWPDelta

__all__ = ['SimVPv2NWPDelta']
