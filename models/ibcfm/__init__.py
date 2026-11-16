"""E39 / MASTER 4.41 -- IBCFM adapter package.

The implementation lives in ``model.py`` and registers itself with the
benchmark's open model registry.  See ``docs/experiments/E39-*.md`` for the
pre-registered criteria; this package ships **no performance claim**.
"""
from .model import IBCFMModel  # noqa: F401

__all__ = ["IBCFMModel"]
