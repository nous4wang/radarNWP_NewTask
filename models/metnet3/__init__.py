"""MetNet-3 baseline adapter.

U-Net backbone with MaxViT blocks, lead-time conditioning and NWP
cross-attention fusion at the bottleneck.
"""
from __future__ import annotations

from .model import MetNet3Adapter

__all__ = ["MetNet3Adapter"]
