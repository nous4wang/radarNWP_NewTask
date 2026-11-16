"""SimVP-v2 with the quantum-inspired NWP tendency plugin.

The reference integration of
:class:`~radar_nwp_bench.models.common.QuantumInspiredTendencyCoupler`:
the backbone and the head are the ones SimVP-v2 already had, and the only
line added is the call to the coupler.
"""
from __future__ import annotations

from .model import SimVPv2QuantumNWP

__all__ = ['SimVPv2QuantumNWP']
