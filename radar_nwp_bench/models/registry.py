"""The model registry.

An **open** registry: a baseline registers itself with the
``@register_model("name")`` decorator, and the harness reaches it by name.
Nothing in the harness enumerates the available models, so adding a baseline
never means editing the framework.

    from radar_nwp_bench.models import register_model
    from radar_nwp_bench.models.base import BasePrecipitationModel

    @register_model("my_model")
    class MyModel(BasePrecipitationModel):
        ...

Model implementations live in the repository's top-level ``models/`` directory,
one package per baseline (``models/<name>/model.py``). They are discovered
lazily, so importing the framework does not import every model.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

from .base import BasePrecipitationModel

#: name -> class. Populated by the ``register_model`` decorator.
_REGISTRY: dict[str, type] = {}

#: Where to look for model implementations, in order of preference.
def _candidate_roots() -> list[Path]:
    """Directories that may contain ``<name>/model.py`` packages."""
    roots: list[Path] = []
    # repository root: <repo>/radar_nwp_bench/models/registry.py -> <repo>
    here = Path(__file__).resolve()
    if len(here.parents) >= 3:
        repo = here.parents[2]
        roots.append(repo / "models")
        roots.append(repo)
    # the package's own models directory
    roots.append(here.parent)
    # the current working directory
    roots.append(Path.cwd() / "models")
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        if r not in seen and r.is_dir():
            seen.add(r)
            unique.append(r)
    return unique


def register_model(name: str) -> Callable[[type], type]:
    """Class decorator that registers a baseline under ``name``.

    Args:
        name: the name the harness will use, e.g. ``"mocast"``.

    Returns:
        The decorator, which returns the class unchanged.
    """
    def decorator(cls: type) -> type:
        if not issubclass(cls, BasePrecipitationModel):
            raise TypeError(
                f"{cls.__name__} must subclass BasePrecipitationModel"
            )
        key = name.lower()
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            # The same file may be imported twice under different module
            # names: once by the registry, which loads it by path, and once
            # as a package (``from models.cascast_nwp import ...``). The
            # second import produces a *different* class object with the
            # same name, so identity cannot be used to tell the two apart.
            # The module file is the thing that identifies a baseline, so
            # that is what is compared.
            if _source_of(existing) == _source_of(cls):
                return cls
            raise ValueError(f"model name {name!r} is already registered")
        _REGISTRY[key] = cls
        return cls

    return decorator


def _source_of(cls: type) -> str:
    """The file a class was defined in, for identity across re-imports."""
    try:
        return str(Path(inspect.getfile(cls)).resolve())
    except (TypeError, OSError):  # pragma: no cover - builtins only
        return f"{cls.__module__}.{cls.__qualname__}"


def _import_model_file(path: Path) -> None:
    """Import a model implementation by file path.

    The module is loaded under a unique name so that two directories holding
    the same baseline name cannot collide, and so that a file which fails to
    import does not poison the registry.
    """
    module_name = f"_rnb_model_{path.parent.name}"
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _discover() -> None:
    """Import every model implementation that has not been registered yet."""
    for root in _candidate_roots():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            model_file = entry / "model.py"
            if not model_file.is_file():
                continue
            if entry.name.lower() in _REGISTRY:
                continue
            _import_model_file(model_file)


def list_models() -> list[str]:
    """Names of every registered baseline."""
    _discover()
    return sorted(_REGISTRY)


def model_class(name: str) -> type | None:
    """The class a name is registered to, without building it.

    A caller that needs to know what a baseline accepts before it
    constructs one -- to check whether a setting applies to it, say --
    needs the class rather than an instance, and building the instance
    to find out would already have failed.

    Args:
        name: one of :func:`list_models`.

    Returns:
        The class, or ``None`` when the name is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        _discover()
    return _REGISTRY.get(key)


def build_model(name: str, **kwargs: Any) -> BasePrecipitationModel:
    """Instantiate a baseline by name.

    Args:
        name: one of :func:`list_models`.
        **kwargs: forwarded to the baseline's constructor.

    Raises:
        ValueError: when ``name`` is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        _discover()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown baseline {name!r}; available: {', '.join(list_models())}"
        )
    return _REGISTRY[key](**kwargs)


def registry_snapshot() -> dict[str, str]:
    """Map of registered name -> implementing class name."""
    _discover()
    return {k: v.__name__ for k, v in sorted(_REGISTRY.items())}


__all__ = [
    "BasePrecipitationModel",
    "register_model",
    "build_model",
    "list_models",
    "registry_snapshot",
]
