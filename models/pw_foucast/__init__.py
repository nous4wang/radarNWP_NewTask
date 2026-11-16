"""PW-FouCast baseline adapter (IJCNN 2026).

AFNO-based spectral encoder/decoder with Pangu-guided frequency modulation,
a frequency memory and inverted frequency attention.
"""
from __future__ import annotations

from .model import PWFouCastAdapter

__all__ = ["PWFouCastAdapter"]
