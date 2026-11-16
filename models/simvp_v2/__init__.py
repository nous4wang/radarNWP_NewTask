"""SimVP v2 baseline.

A SimVP variant whose core is built from ConvNeXt blocks.
"""
from __future__ import annotations

from .model import SimVPv2Model
from .model import ConvNeXtBlock

__all__ = ['SimVPv2Model', 'ConvNeXtBlock']
