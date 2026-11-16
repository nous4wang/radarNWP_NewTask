"""Hybrid Quantum Manifold Architecture.

The shallow concatenation that M1 proved worth keeping, a deep Koopman
tendency over the two NWP slices, and a hierarchical projection head
that predicts a nested distribution over rain rates instead of a single
number -- because the single number came out systematically too small.
"""
from __future__ import annotations

from .model import SimVPv2HybridQuantumManifold

__all__ = ['SimVPv2HybridQuantumManifold']
