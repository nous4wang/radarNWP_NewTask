"""Optical-flow advection baseline.

An analytical baseline in the spirit of pySTEPS: the motion is taken
from the last two radar frames and the field is advected, then converted to
rainfall through a Z-R relation. It has no trainable parameters, so the
harness evaluates it rather than fitting it.
"""
from __future__ import annotations

from .model import PySTEPSAdvection

__all__ = ['PySTEPSAdvection']
