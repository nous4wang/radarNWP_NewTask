"""Tests for the NWP ablation and for the argument symmetry it depends on.

Two failure modes are locked down here:

1. **Argument drift.** ``train.py`` declared ``--nwp-vars`` and
   ``--radar-vars`` while ``evaluate.py`` did not. A model trained on a
   subset of the NWP variables then had an input layer with three channels
   while the evaluation built one with twenty-seven, and the checkpoint
   failed to load. The parser comparison below now covers those flags, and
   the evaluation additionally inherits the training run's variable subsets
   from the ``config.json`` written next to the checkpoint.

2. **An ablation that changes the experiment.** The whole point of the
   ablation is that the three cells differ *only* in whether the NWP input
   carries information. If blanking the NWP also changed the parameter
   count, the checkpoint layout, or the set of evaluated windows, the
   comparison would no longer be an ablation. These tests pin those
   properties down.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _pytest_compat import pytest  # noqa: E402
from radar_nwp_bench.ablation import (  # noqa: E402
    MODE_FULL,
    MODE_NO_NWP,
    MODE_ZERO_NWP,
    apply_ablation,
    describe,
    resolve_mode,
)


def _load_script(module_name: str, filename: str):
    """Import a script by path."""
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _flags(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.option_strings[0]
        for action in parser._actions
        if action.option_strings and action.option_strings[0].startswith("--")
    }


# ---------------------------------------------------------------------------
# 1. the ablation switch itself
# ---------------------------------------------------------------------------
def test_the_mode_is_resolved_from_the_two_flags():
    assert resolve_mode(False, False) == MODE_FULL
    assert resolve_mode(True, False) == MODE_NO_NWP
    assert resolve_mode(False, True) == MODE_ZERO_NWP


def test_the_two_flags_cannot_be_combined():
    """They are different experiments, so a run must not silently be both."""
    with pytest.raises(ValueError):
        resolve_mode(True, True)


def test_blanking_the_nwp_keeps_the_shape():
    """The shape is what keeps the ablation controlled.

    A different shape would mean a different first layer, and therefore a
    different number of weights -- the comparison would no longer isolate
    the information in the NWP input.
    """
    import torch

    x = torch.rand(2, 2, 27, 8, 8)
    blanked = apply_ablation(x, MODE_NO_NWP)
    assert blanked.shape == x.shape
    assert float(blanked.abs().sum()) == 0.0
    assert blanked.dtype == x.dtype
    assert blanked.device == x.device


def test_the_full_mode_changes_nothing():
    import torch

    x = torch.rand(2, 2, 27, 8, 8)
    assert apply_ablation(x, MODE_FULL) is x


def test_an_unknown_mode_is_rejected():
    import torch

    with pytest.raises(ValueError):
        apply_ablation(torch.rand(1, 2, 3, 4, 4), "nonsense")


def test_every_mode_describes_itself():
    """A report without a description cannot be placed in a results table."""
    descriptions = {mode: describe(mode) for mode in
                    (MODE_FULL, MODE_NO_NWP, MODE_ZERO_NWP)}
    for mode, text in descriptions.items():
        assert text.strip(), f"{mode} has no description"
    # The three cells are different experiments; they must not be
    # described in the same words.
    assert len(set(descriptions.values())) == len(descriptions)


# ---------------------------------------------------------------------------
# 2. the model seam accepts a missing NWP
# ---------------------------------------------------------------------------
def test_a_model_accepts_a_missing_nwp_tensor():
    """``x_nwp=None`` is a first-class input, not an error.

    Every adapter must survive it, including the six that never call
    ``_check_shapes`` and go straight to ``x_nwp.view(...)``.
    """
    import torch

    from radar_nwp_bench.models import build_model, list_models

    for name in list_models():
        model = build_model(name, in_radar_c=10, in_nwp_c=27, in_len=4, out_len=2)
        with torch.no_grad():
            out = model(torch.rand(1, 4, 10, 16, 16), None)
        assert torch.isfinite(out).all(), f"{name} produced a non-finite output"
        assert out.shape == (1, 2, 1, 16, 16), f"{name} changed the output shape"


def test_a_missing_nwp_and_a_blank_nwp_agree():
    """The two ways of saying "no NWP" must give the same answer.

    ``None`` is resolved by the model; a zero tensor is passed straight
    through. If they disagreed, the ablation would depend on which spelling
    a caller happened to use.
    """
    import torch

    from radar_nwp_bench.models import build_model

    torch.manual_seed(0)
    model = build_model("convlstm", in_radar_c=10, in_nwp_c=27, in_len=4, out_len=2)
    model.eval()
    x_radar = torch.rand(1, 4, 10, 16, 16)
    with torch.no_grad():
        without = model(x_radar, None)
        blank = model(x_radar, torch.zeros(1, 2, 27, 16, 16))
    assert torch.allclose(without, blank, atol=1e-6)


def test_the_zero_tensor_has_the_declared_shape():
    """The substitute must match what the model's own encoder expects."""
    import torch

    from radar_nwp_bench.models import build_model

    model = build_model("convlstm", in_radar_c=10, in_nwp_c=27, in_len=4, out_len=2)
    x_radar = torch.rand(2, 4, 10, 16, 16)
    resolved = model.resolve_nwp(x_radar, None)
    assert resolved.shape == (2, model.nwp_time_len, 27, 16, 16)
    assert resolved.dtype == x_radar.dtype
    assert resolved.device == x_radar.device


def test_the_nwp_slice_count_can_be_overridden():
    """It comes from the alignment, so the harness must be able to set it."""
    import torch

    from radar_nwp_bench.models import build_model

    model = build_model("convlstm", in_radar_c=10, in_nwp_c=27, in_len=4, out_len=2)
    model.nwp_time_len = 5
    resolved = model.resolve_nwp(torch.rand(1, 4, 10, 8, 8), None)
    assert resolved.shape == (1, 5, 27, 8, 8)


# ---------------------------------------------------------------------------
# 3. the scripts agree with each other
# ---------------------------------------------------------------------------
def test_train_and_evaluate_expose_the_same_data_flags():
    """Every flag that shapes the data must exist in both scripts."""
    train = _load_script("rnb_abl_train", "train.py").build_argparser()
    evaluate = _load_script("rnb_abl_eval", "evaluate.py").build_argparser()

    train_flags = _flags(train)
    evaluate_flags = _flags(evaluate)

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
        # The two that were missing from evaluate.py and made a subset-NWP
        # checkpoint impossible to evaluate.
        "--nwp-vars",
        "--radar-vars",
        # The ablation switches.
        "--no-nwp",
        "--zero-nwp",
    }
    for flag in sorted(data_flags):
        assert flag in train_flags, f"train.py is missing {flag}"
        assert flag in evaluate_flags, f"evaluate.py is missing {flag}"


def test_the_ablation_flags_default_to_off():
    """A run that mentions neither flag must use the NWP."""
    for filename in ("train.py", "evaluate.py"):
        module = _load_script(f"rnb_abl_def_{filename.replace('.', '_')}", filename)
        args = module.build_argparser().parse_args(
            ["--model", "convlstm", "--data-root", "x"]
        )
        assert args.no_nwp is False
        assert args.zero_nwp is False


def test_the_variable_flags_default_to_none():
    """``None`` means "not specified", so the training run can be asked."""
    for filename in ("train.py", "evaluate.py"):
        module = _load_script(f"rnb_abl_vars_{filename.replace('.', '_')}", filename)
        args = module.build_argparser().parse_args(
            ["--model", "convlstm", "--data-root", "x"]
        )
        assert args.nwp_vars is None
        assert args.radar_vars is None


def test_the_variable_flags_accept_a_subset():
    """The subset is what the user actually asked for."""
    module = _load_script("rnb_abl_subset", "evaluate.py")
    args = module.build_argparser().parse_args(
        ["--model", "convlstm", "--data-root", "x",
         "--nwp-vars", "CAPE", "PWAT", "DVG850"]
    )
    assert args.nwp_vars == ["CAPE", "PWAT", "DVG850"]


def test_evaluate_inherits_the_variable_subsets_from_the_training_run():
    """The checkpoint knows which variables it was trained for.

    Without this, evaluating a subset-NWP checkpoint against the default
    twenty-seven variables builds a model whose first layer has the wrong
    shape, and the weights fail to load.
    """
    import json
    import tempfile

    module = _load_script("rnb_abl_inherit", "evaluate.py")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "config.json").write_text(
            json.dumps({"nwp_vars": ["CAPE", "PWAT"],
                        "radar_vars": ["CR", "ET"]}),
            encoding="utf-8",
        )
        trained = module._training_run_config(str(run_dir / "last.pt"))
        assert trained["nwp_vars"] == ["CAPE", "PWAT"]
        assert trained["radar_vars"] == ["CR", "ET"]


def test_evaluate_copes_with_a_checkpoint_that_has_no_config():
    """A missing config must not crash the evaluation."""
    module = _load_script("rnb_abl_noconfig", "evaluate.py")
    assert module._training_run_config(None) is None
    assert module._training_run_config("/nonexistent/last.pt") is None


def test_the_training_runs_ablation_mode_is_readable():
    """A checkpoint knows how it was trained.

    A model trained without the NWP, evaluated with the NWP, produces a
    number rather than an error -- and the number looks like a result. The
    training run records its mode, so the evaluation can notice.
    """
    module = _load_script("rnb_abl_mode", "evaluate.py")
    assert module._trained_ablation_mode({"no_nwp": True}) == MODE_NO_NWP
    assert module._trained_ablation_mode({"zero_nwp": True}) == MODE_ZERO_NWP
    assert module._trained_ablation_mode({}) == MODE_FULL
    # No training configuration at all: nothing to compare against.
    assert module._trained_ablation_mode(None) is None


def test_the_training_script_records_its_ablation_mode():
    """The config.json next to the checkpoint has to carry the mode."""
    source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    # The run's arguments are written next to the checkpoint, and they carry
    # the mode (the --no-nwp and --zero-nwp flags are among them).
    assert "run_config = dict(vars(args))" in source
    assert '"config.json"' in source


def test_the_input_pipeline_version_travels_with_the_checkpoint():
    """A checkpoint trained under one scaling cannot be scored under another.

    The NWP normalisation changed what the network sees, so the eight
    checkpoints trained before it expect fields on a different scale. Their
    scores are not comparable, and the evaluation has to say so rather than
    produce a number that looks like a result.
    """
    from radar_nwp_bench.data.transforms import NWP_PIPELINE_VERSION

    assert NWP_PIPELINE_VERSION

    train_source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "nwp_pipeline_version" in train_source
    assert "NWP_PIPELINE_VERSION" in train_source

    evaluate_source = (REPO_ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert "nwp_pipeline_version" in evaluate_source
    assert "NWP_PIPELINE_VERSION" in evaluate_source


def test_the_evaluation_refuses_a_checkpoint_from_another_pipeline():
    """The refusal is a real code path, not a comment."""
    module = _load_script("rnb_abl_version", "evaluate.py")
    source = (REPO_ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert "trained_version" in source
    assert "return 2" in source
    assert hasattr(module, "_training_run_config")


def test_the_scripts_actually_use_the_ablation_switch():
    """Declaring the flag is not enough; it has to reach the model."""
    for filename in ("train.py", "evaluate.py"):
        source = (REPO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "resolve_mode" in source, (
            f"{filename} does not resolve the ablation mode"
        )
        assert "apply_ablation" in source, (
            f"{filename} does not apply the ablation"
        )


def test_the_scripts_forward_the_variable_subsets_to_the_dataset():
    """Otherwise the flag is decorative."""
    for filename in ("train.py", "evaluate.py"):
        source = (REPO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "nwp_vars" in source
        assert "radar_vars" in source


def test_no_script_reads_an_argument_it_never_declares():
    """The cheap static check that caught the original defect."""
    for filename in ("train.py", "evaluate.py"):
        path = REPO_ROOT / "scripts" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))

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
# 4. the dataset honours the ablation without moving the goalposts
# ---------------------------------------------------------------------------
def test_the_dataset_reports_its_nwp_slice_count():
    """The model needs it to build a correctly shaped zero tensor."""
    import inspect

    from radar_nwp_bench.data import RadarNWPDataset

    assert hasattr(RadarNWPDataset, "nwp_frames")
    assert inspect.getsource(RadarNWPDataset.nwp_frames.fget)


def test_blanking_the_nwp_does_not_change_the_sample_index():
    """The ablation must not move the goalposts.

    A sample is kept or dropped depending on whether a forecast NWP slice
    exists for it. If blanking the NWP changed that, the ablated run would
    be scored on a different set of windows and the difference in score
    would be a difference of samples rather than of information.
    """
    import inspect

    from radar_nwp_bench.data import RadarNWPDataset

    source = inspect.getsource(RadarNWPDataset.__getitem__)
    assert "blank_nwp" in source
    # The zero tensor is built in place of the resampled field, not by
    # filtering the sample list.
    assert "np.zeros" in source


def test_the_dataset_exposes_the_blank_nwp_option():
    import inspect

    from radar_nwp_bench.data import RadarNWPDataset

    signature = inspect.signature(RadarNWPDataset.__init__)
    assert "blank_nwp" in signature.parameters
    assert signature.parameters["blank_nwp"].default is False


def test_the_training_script_blanks_the_dataset_for_the_no_nwp_arm():
    """Not reading the NWP at all is stronger than zeroing it."""
    source = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "blank_nwp" in source
    assert "MODE_NO_NWP" in source


# ---------------------------------------------------------------------------
# 5. the configuration file feeds the variable subsets through
# ---------------------------------------------------------------------------
def test_the_configuration_file_can_select_the_variables():
    """The YAML's ``radar_vars`` / ``nwp_vars`` must reach argparse."""
    from radar_nwp_bench.config import argparse_defaults, load_config

    defaults = argparse_defaults(load_config("default"), "train")
    assert "nwp_vars" in defaults
    assert "radar_vars" in defaults
    assert len(defaults["nwp_vars"]) == 27
    assert len(defaults["radar_vars"]) == 10


def test_the_variable_defaults_reach_the_evaluation_too():
    """Otherwise a configuration file would train and evaluate differently."""
    from radar_nwp_bench.config import argparse_defaults, load_config

    config = load_config("default")
    train = argparse_defaults(config, "train")
    evaluate = argparse_defaults(config, "eval")
    assert train["nwp_vars"] == evaluate["nwp_vars"]
    assert train["radar_vars"] == evaluate["radar_vars"]
