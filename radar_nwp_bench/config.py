"""Configuration files: load, resolve, and layer.

The repository ships YAML files under ``configs/``:

* ``configs/default.yaml`` -- the shared data / train / eval defaults;
* ``configs/data/<region>.yaml`` -- a region's data settings and audit facts;
* ``configs/models/<name>.yaml`` -- one baseline's constructor kwargs, plus
  the paper it implements and its upstream licence.

Those files used to be documentation only: nothing in the harness read them,
so a model could ship a set of hyperparameters that were silently ignored.
This module makes them real. The layering order, from lowest to highest
precedence:

1. the YAML file;
2. the script's own argparse defaults;
3. the command line.

Explicit command-line flags always win, so a config file sets the *starting
point* rather than locking a run to it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

#: Where configuration files are looked up.
#: ``<repo>/radar_nwp_bench/config.py`` -> the repository root is parents[1].
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

#: Keys of a model config that describe the tensor shapes rather than a
#: hyperparameter. They are owned by the dataset, not by the YAML file.
SHAPE_KEYS = frozenset({"in_radar_c", "in_nwp_c", "in_len", "out_len"})


def config_root() -> Path:
    """The directory holding the shipped configuration files."""
    return CONFIG_ROOT


def resolve_config_path(spec: str | Path) -> Path:
    """Turn a ``--config`` argument into a concrete file path.

    Accepts a path to a file, or a bare name that is resolved against the
    shipped configurations:

    * ``mocast`` -> ``configs/models/mocast.yaml``
    * ``default`` -> ``configs/default.yaml``
    * ``bj`` -> ``configs/data/bj.yaml``

    Raises:
        FileNotFoundError: when nothing matches.
    """
    candidate = Path(spec)
    if candidate.is_file():
        return candidate

    name = str(spec)
    options = [
        CONFIG_ROOT / "models" / f"{name}.yaml",
        CONFIG_ROOT / "data" / f"{name}.yaml",
        CONFIG_ROOT / f"{name}.yaml",
        CONFIG_ROOT / f"{name}.yml",
    ]
    for option in options:
        if option.is_file():
            return option

    raise FileNotFoundError(
        f"no configuration file for {spec!r}; looked in "
        + ", ".join(str(o) for o in options)
    )


def load_config(spec: str | Path | None) -> dict[str, Any]:
    """Read one configuration file.

    Args:
        spec: a path, a bare configuration name, or ``None`` for "no file".

    Returns:
        The parsed mapping, or an empty mapping when ``spec`` is ``None``.

    Raises:
        FileNotFoundError: when the file cannot be resolved.
        ValueError: when the file is not a YAML mapping.
    """
    if spec is None:
        return {}
    path = resolve_config_path(spec)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a dependency
        raise RuntimeError(
            "PyYAML is required to read configuration files"
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    data.setdefault("_path", str(path))
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file without any of the name resolution above."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


#: Which YAML section holds which argparse destination.
#:
#: ``radar_vars`` and ``nwp_vars`` are in here because they decide the
#: *channel count* of the model's input. A file that lists a subset of the
#: NWP variables must be honoured identically by training and by evaluation,
#: or the evaluation builds a model with 27 NWP channels and fails to load a
#: checkpoint that was trained with 3.
_SECTION_MAP: tuple[tuple[str, str, str], ...] = (
    ("data", "data_root", "root"),
    ("data", "packed", "packed"),
    ("data", "patch_size", "patch_size"),
    ("data", "in_len", "in_len"),
    ("data", "out_len", "out_len"),
    ("data", "alignment", "alignment"),
    ("data", "radar_vars", "radar_vars"),
    ("data", "nwp_vars", "nwp_vars"),
    ("train", "epochs", "epochs"),
    ("train", "batch_size", "batch_size"),
    ("train", "num_workers", "num_workers"),
    ("train", "lr", "lr"),
    ("train", "weight_decay", "weight_decay"),
    ("train", "seed", "seed"),
    ("train", "device", "device"),
)


def argparse_defaults(
    config: dict[str, Any],
    section: str = "train",
    extra: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Translate a configuration file into argparse defaults.

    Args:
        config: the parsed configuration.
        section: ``"train"`` or ``"eval"``; the data section is always read.
        extra: additional ``(section, key)`` pairs to resolve.

    Returns:
        A mapping of argparse destination names to values. Destinations the
        configuration does not mention are absent, so the caller can leave
        them at their own defaults.
    """
    data = config.get("data") or {}
    train = config.get("train") or {}
    groups = {"data": data, "train": train, "eval": config.get("eval") or {}}

    resolved: dict[str, Any] = {}
    for group, key, dest in _SECTION_MAP:
        source = groups.get(group) or {}
        if key in source:
            resolved[dest] = source[key]
    for group, key in extra:
        source = groups.get(group) or {}
        if key in source:
            resolved[key] = source[key]
    return resolved


def model_kwargs(
    config: dict[str, Any],
    shapes: dict[str, int] | None = None,
    overrides: str | None = None,
) -> dict[str, Any]:
    """Extract the constructor kwargs for a baseline from a configuration.

    Args:
        config: the parsed configuration (its ``model`` section is used).
        shapes: the tensor shapes the dataset dictates; they override
            whatever the file says, because the shapes come from the data.
        overrides: a JSON object whose entries win over the file, for
            ad-hoc runs.

    Returns:
        Keyword arguments for the baseline's constructor.
    """
    kwargs: dict[str, Any] = dict((config.get("model") or {}))
    kwargs.pop("_path", None)
    for key in SHAPE_KEYS:
        kwargs.pop(key, None)
    if shapes:
        kwargs.update(shapes)
    if overrides:
        extra = json.loads(overrides)
        if not isinstance(extra, dict):
            raise ValueError("--model-kwargs must be a JSON object")
        kwargs.update(extra)
    return kwargs


__all__ = [
    "CONFIG_ROOT",
    "SHAPE_KEYS",
    "argparse_defaults",
    "config_root",
    "load_config",
    "load_yaml",
    "model_kwargs",
    "resolve_config_path",
]
