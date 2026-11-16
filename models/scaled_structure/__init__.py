"""E45 / MASTER 4.47 -- the scaled_structure arm.

The implementation lives in ``model.py`` and registers itself with the
benchmark's open model registry under two names: ``scaled_structure``
(the amount is carried by a new scalar head) and ``scaled_structure_own``
(the amount is carried by the model's own ordinary decode head).  The
structure operator is E44-A2's additive decomposition, reused verbatim:
the low-rank projection of the decoder's output map plus the
independently predicted sparse head, so the shape the arm reshapes is
one whose improvement was MEASURED in E44.  See
``docs/experiments/E45-gate-direction-amount-separation-baseline.md``
for the pre-registered criteria; this package ships **no performance
claim**.
"""
from .model import ScaledStructureModel, ScaledStructureOwnModel

__all__ = ["ScaledStructureModel", "ScaledStructureOwnModel"]
