"""MoCast baseline adapter (AAAI 2026).

Motion branch with physics guidance (PMM) + mixture-of-experts source/sink
branch (MSM), fused by differentiable advection.
"""
from __future__ import annotations

from .model import MoCastAdapter

__all__ = ["MoCastAdapter"]
