"""Route R4: a semi-Lagrangian advection with a learned source and sink.

The rainfall field is carried forward by a motion field instead of being
re-predicted pixel by pixel, and a gate decides how much of each frame
comes from the motion and how much from the learned residual. The two
switches in the configuration give the three arms of the ablation.
"""
from __future__ import annotations

from .model import SimVPv2Lagrangian

__all__ = ["SimVPv2Lagrangian"]
