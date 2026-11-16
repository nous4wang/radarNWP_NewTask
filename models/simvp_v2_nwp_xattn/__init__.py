"""SimVP-v2 with the NWP fused by a conventional spatial cross-attention.

Ablation row M2 of the NWP-injection study: the radar features query an
encoded NWP feature map instead of the NWP riding on the input channels.
The attention block is ordinary multi-head attention with no Koopman or
quantum-inspired machinery, so the row measures what a competent
conventional fusion achieves.
"""
from __future__ import annotations

from .model import SimVPv2NWPCrossAttention

__all__ = ['SimVPv2NWPCrossAttention']
