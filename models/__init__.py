"""Baseline model implementations, one package per model.

Each subdirectory holds a ``model.py`` that defines a class deriving from
:class:`radar_nwp_bench.models.base.BasePrecipitationModel` and registers
itself with ``@register_model("<name>")``.

This module is deliberately free of imports from ``radar_nwp_bench`` at module
level: the registry discovers the model files by path, and a model file that
imports the framework must not find the framework half-initialised.
"""
