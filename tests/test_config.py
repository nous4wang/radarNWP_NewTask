"""Tests for the configuration files.

The YAML files under ``configs/`` used to be documentation: nothing read
them, so a baseline could ship a set of hyperparameters that were silently
ignored at training time. These tests pin that they are now real, and that
the layering is the one the harness advertises:

    command line  >  configuration file  >  script defaults
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.config import (  # noqa: E402
    SHAPE_KEYS,
    argparse_defaults,
    config_root,
    load_config,
    model_kwargs,
    resolve_config_path,
)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def test_config_root_points_at_the_shipped_configs():
    root = config_root()
    assert root.is_dir(), root
    assert (root / "default.yaml").is_file()


def test_a_bare_name_resolves_to_the_right_file():
    assert resolve_config_path("default") == config_root() / "default.yaml"
    assert resolve_config_path("mocast") == config_root() / "models" / "mocast.yaml"
    assert resolve_config_path("bj") == config_root() / "data" / "bj.yaml"


def test_an_explicit_path_is_used_as_given():
    path = config_root() / "models" / "mocast.yaml"
    assert resolve_config_path(path) == path


def test_an_unknown_name_is_reported_clearly():
    try:
        resolve_config_path("no_such_configuration")
    except FileNotFoundError as exc:
        assert "no_such_configuration" in str(exc)
    else:
        raise AssertionError("an unknown configuration must raise")


# ---------------------------------------------------------------------------
# layering
# ---------------------------------------------------------------------------
def test_the_default_file_supplies_the_documented_values():
    defaults = argparse_defaults(load_config("default"), "train")
    assert defaults["epochs"] == 10
    assert defaults["batch_size"] == 2
    assert defaults["lr"] == 2.0e-4
    assert defaults["alignment"] == "forecast"
    assert defaults["in_len"] == 11
    assert defaults["out_len"] == 10


def test_the_data_section_is_read_for_both_train_and_eval():
    config = load_config("default")
    train = argparse_defaults(config, "train")
    evaluate = argparse_defaults(config, "eval")
    assert train["patch_size"] == evaluate["patch_size"]
    assert train["batch_size"] == 2


def test_model_kwargs_come_from_the_model_file():
    kwargs = model_kwargs(load_config("mocast"))
    assert kwargs["channels"] == 32
    assert kwargs["num_experts"] == 4


def test_shapes_override_the_file_because_the_data_owns_them():
    """The tensor shapes come from the dataset, never from a YAML file."""
    kwargs = model_kwargs(
        load_config("mocast"),
        shapes={"in_radar_c": 10, "in_nwp_c": 27, "in_len": 11, "out_len": 10},
    )
    assert kwargs["in_radar_c"] == 10
    assert kwargs["in_nwp_c"] == 27
    assert kwargs["in_len"] == 11
    assert kwargs["out_len"] == 10


def test_shape_keys_are_never_taken_from_the_file():
    """A file that claims a different shape must not be able to set one.

    The shapes come from the dataset -- the number of radar channels and the
    number of NWP variables are facts about the data, not tunable
    hyperparameters. A configuration file that disagrees is overridden, and a
    shape the caller does not supply is simply left unset rather than
    invented.
    """
    config = {"model": {"in_len": 3, "out_len": 4, "in_radar_c": 1, "in_nwp_c": 2,
                        "channels": 8}}
    kwargs = model_kwargs(config, shapes={"in_len": 11, "out_len": 10})
    assert kwargs["in_len"] == 11
    assert kwargs["out_len"] == 10
    assert kwargs["channels"] == 8
    # the file's channel counts are discarded, not merely overridden
    assert "in_radar_c" not in kwargs
    assert "in_nwp_c" not in kwargs
    for key in SHAPE_KEYS:
        assert key not in config["model"] or kwargs.get(key) != config["model"][key]


def test_json_overrides_win_over_the_file():
    kwargs = model_kwargs(
        load_config("mocast"), overrides=json.dumps({"channels": 64})
    )
    assert kwargs["channels"] == 64


def test_a_non_object_override_is_rejected():
    try:
        model_kwargs(load_config("mocast"), overrides="[1, 2, 3]")
    except ValueError:
        pass
    else:
        raise AssertionError("--model-kwargs must reject a non-object")


# ---------------------------------------------------------------------------
# every shipped model configuration is usable
# ---------------------------------------------------------------------------
def test_every_model_configuration_builds_its_model():
    """A configuration that cannot build its model is worse than none."""
    import inspect

    from radar_nwp_bench.models import build_model, list_models

    known = set(list_models())
    for path in sorted((config_root() / "models").glob("*.yaml")):
        name = path.stem
        if name not in known:
            continue
        config = load_config(path)
        kwargs = model_kwargs(
            config,
            shapes={
                "in_radar_c": 10,
                "in_nwp_c": 27,
                "in_len": 11,
                "out_len": 10,
            },
        )
        model = build_model(name, **kwargs)
        accepted = set(inspect.signature(model.__class__.__init__).parameters)
        unexpected = [
            k for k in kwargs
            if k not in accepted and k not in ("self", "args", "kwargs")
        ]
        assert not unexpected, (
            f"{name}.yaml passes {unexpected}, which the adapter does not accept"
        )


def test_every_model_configuration_records_its_paper():
    """A baseline without a paper reference is an unexplained baseline."""
    for path in sorted((config_root() / "models").glob("*.yaml")):
        config = load_config(path)
        assert config.get("paper"), f"{path.name} has no paper reference"
        assert "license" in config, f"{path.name} has no licence field"


# ---------------------------------------------------------------------------
# the scripts really use the files
# ---------------------------------------------------------------------------
def _run_train(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py")] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_the_training_script_reports_the_configuration_it_used():
    """The run must say which file it read, so it is auditable."""
    proc = _run_train(
        [
            "--model", "mocast",
            "--data-root", "unused",
            "--config", "mocast",
            "--epochs", "1", "--max-steps", "1",
        ]
    )
    # the data root does not exist, so the run fails -- but it must fail
    # *after* reporting the configuration, which proves the file was read.
    assert "config:" in proc.stdout, proc.stdout + proc.stderr


def test_the_command_line_overrides_the_configuration_file():
    """A file sets the starting point; an explicit flag wins."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rnb_train_cfg", REPO_ROOT / "scripts" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_train_cfg"] = module
    spec.loader.exec_module(module)

    ap = module.build_argparser()
    args, _ = module._apply_config(
        ap, ["--model", "mocast", "--data-root", "x", "--config", "default"]
    )
    assert args.epochs == 10, "the file's value must be used when no flag is given"
    assert args.patch_size == 128

    ap = module.build_argparser()
    args, _ = module._apply_config(
        ap,
        ["--model", "mocast", "--data-root", "x", "--config", "default",
         "--epochs", "3", "--patch-size", "64"],
    )
    assert args.epochs == 3, "an explicit flag must beat the file"
    assert args.patch_size == 64


def test_a_model_configuration_is_found_without_naming_it():
    """``--model mocast`` alone must pick up ``configs/models/mocast.yaml``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rnb_train_cfg2", REPO_ROOT / "scripts" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_train_cfg2"] = module
    spec.loader.exec_module(module)

    source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "load_config(args.model)" in source, (
        "the training script must fall back to the model's own configuration"
    )


def test_a_missing_data_root_is_a_clear_error_not_a_traceback():
    proc = _run_train(["--model", "mocast", "--config", "mocast"])
    assert proc.returncode != 0
    assert "--data-root is required" in proc.stderr, proc.stderr
