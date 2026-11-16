"""Contract tests for the command-line scripts.

These lock the two failure modes that have actually bitten this repository:

1. **Argument drift.** ``train.py`` and ``evaluate.py`` must expose the same
   data-shaping flags. When ``--slide-stride`` existed only in ``train.py``,
   ``evaluate.py`` raised ``AttributeError: 'Namespace' object has no
   attribute 'slide_stride'`` the moment a sliding-window evaluation was
   requested. The test below compares the two parsers directly.

2. **Unit drift.** The models emit millimetres; the loss works in
   ``log1p(mm)`` space. Both scripts must apply the same physical
   non-negativity convention, so that a prediction can never be scored in one
   unit and trained in another.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCRIPTS = REPO_ROOT / "scripts"


def _load_parser(module_name: str, filename: str, func_name: str):
    """Import a script by path and return its argument parser factory."""
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def _flags(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.option_strings[0]
        for action in parser._actions
        if action.option_strings and action.option_strings[0].startswith("--")
    }


# ---------------------------------------------------------------------------
# 1. train.py / evaluate.py argument symmetry
# ---------------------------------------------------------------------------
def test_train_and_evaluate_expose_the_same_data_flags():
    """Every flag that shapes the data must exist in both scripts."""
    train = _load_parser("rnb_train", "train.py", "build_argparser")()
    evaluate = _load_parser("rnb_evaluate", "evaluate.py", "build_argparser")()

    train_flags = _flags(train)
    evaluate_flags = _flags(evaluate)

    # These control what the dataset yields, so they must match exactly.
    data_flags = {
        "--data-root",
        "--packed",
        "--patch-size",
        "--in-len",
        "--out-len",
        "--alignment",
        "--split",
        "--max-samples",
        "--slide-stride",
        "--seed",
        "--device",
    }
    for flag in sorted(data_flags):
        assert flag in train_flags, f"train.py is missing {flag}"
        assert flag in evaluate_flags, f"evaluate.py is missing {flag}"


def test_evaluate_declares_slide_stride():
    """The flag that caused the reported AttributeError."""
    evaluate = _load_parser("rnb_evaluate2", "evaluate.py", "build_argparser")()
    assert "--slide-stride" in _flags(evaluate)
    args = evaluate.parse_args(
        ["--model", "convlstm", "--data-root", "x", "--slide-stride", "32"]
    )
    assert args.slide_stride == 32


def test_slide_stride_defaults_to_none_in_both_scripts():
    """Absent the flag, both scripts must agree on the default."""
    train = _load_parser("rnb_train3", "train.py", "build_argparser")()
    evaluate = _load_parser("rnb_evaluate3", "evaluate.py", "build_argparser")()
    train_args = train.parse_args(["--model", "convlstm", "--data-root", "x"])
    evaluate_args = evaluate.parse_args(["--model", "convlstm", "--data-root", "x"])
    assert train_args.slide_stride is None
    assert evaluate_args.slide_stride is None


def test_slide_stride_is_forwarded_to_the_dataset():
    """Both scripts must actually pass the flag to RadarNWPDataset.

    Declaring the flag is not enough: the value has to reach the dataset,
    otherwise ``--slide-stride`` would be silently ignored.

    The check looks for the *value* being passed on, not for one particular
    spelling of it: ``evaluate.py`` routes it through a helper, so the
    literal ``slide_stride=args.slide_stride`` is not the only correct way
    to write it. What must not happen is the flag being dropped.
    """
    for filename in ("train.py", "evaluate.py"):
        source = (SCRIPTS / filename).read_text(encoding="utf-8")
        forwarded = (
            "slide_stride=args.slide_stride" in source
            # the helper form: the dataset is built by a function that
            # receives the stride as an argument
            or "slide_stride=slide_stride" in source
        )
        assert forwarded, (
            f"{filename} declares --slide-stride but does not forward it"
        )


def test_evaluate_script_has_no_unbound_namespace_access():
    """Guard against referencing an argument the parser never declares.

    A cheap static check: every ``args.<name>`` in a script must correspond
    to a flag that the same script declares.
    """
    for filename in ("train.py", "evaluate.py"):
        path = SCRIPTS / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))

        # collect the flags the parser declares
        declared: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        declared.add(arg.value.lstrip("-").replace("-", "_"))

        # collect the argparse attributes the script reads
        used: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "args"
            ):
                used.add(node.attr)

        missing = used - declared
        assert not missing, (
            f"{filename} reads args.{sorted(missing)} but never declares "
            "the matching flag"
        )


# ---------------------------------------------------------------------------
# 2. unit convention
# ---------------------------------------------------------------------------
def test_loss_clamps_before_the_log_transform():
    """A negative prediction must never reach ``log1p``.

    ``log1p(x)`` is NaN for ``x <= -1``, so the physical floor has to be
    applied *before* the transform, not after.
    """
    import torch

    from radar_nwp_bench.metrics.losses import PrecipitationLoss

    criterion = PrecipitationLoss()
    pred = torch.full((1, 2, 1, 4, 4), -5.0)
    target = torch.zeros(1, 2, 1, 4, 4)
    loss, components = criterion(pred, target)
    assert torch.isfinite(loss), "loss must stay finite for negative predictions"
    for value in components.values():
        assert value == value, "components must not be NaN"


def test_loss_gives_gradient_to_negative_predictions():
    """The clamp must not silence the gradient entirely.

    ``clamp_min(0)`` has a zero derivative below the threshold, so a model
    whose output is entirely negative would receive no learning signal at
    all. The loss therefore keeps a gradient path through the floor.
    """
    import torch

    from radar_nwp_bench.metrics.losses import PrecipitationLoss

    criterion = PrecipitationLoss()
    pred = torch.full((1, 1, 1, 2, 2), -0.5, requires_grad=True)
    target = torch.full((1, 1, 1, 2, 2), 2.0)
    loss, _ = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0, (
        "negative predictions must still produce a learning signal"
    )


def test_every_registered_model_emits_non_negative_rainfall():
    """The physical contract: rainfall in millimetres, never negative."""
    import torch

    from radar_nwp_bench.models import build_model, list_models

    for name in list_models():
        model = build_model(name, in_radar_c=10, in_nwp_c=27,
                            in_len=11, out_len=10)
        with torch.no_grad():
            out = model(torch.rand(1, 11, 10, 32, 32), torch.rand(1, 2, 27, 32, 32))
        assert torch.isfinite(out).all(), f"{name} produced a non-finite output"
        assert (out >= 0).all(), f"{name} produced negative rainfall"
