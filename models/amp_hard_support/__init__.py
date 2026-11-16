"""E45 / MASTER 4.47 -- the amp_hard_support arm.

The implementation lives in ``model.py`` and registers itself with the
benchmark's open model registry.  See
``docs/experiments/E45-gate-direction-amount-separation-baseline.md``
for the pre-registered criteria; this package ships **no performance
claim**.
"""
from .model import AmpHardSupportModel

__all__ = ["AmpHardSupportModel"]
