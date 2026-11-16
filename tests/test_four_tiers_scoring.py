"""The scorer must not write two arms into the same report.

This locks a failure that actually happened. The training stamps its run
directory with the wall clock at launch, so ten seeds of one arm -- and
two arms that start in the same second -- end up sharing a handful of
directory names. The scorer used to key its output on that name alone,
so the second arm to finish overwrote the first one's report: the
sixteen evaluations of the second stage produced six reports, and the
six were whichever arm happened to finish last.

The output directory is therefore keyed on the arm directory above the
run directory, and two runs that share a name must not share an output
path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TOOLS = REPO_ROOT / "tools"


def _load_scorer():
    """Import the scorer by path."""
    path = TOOLS / "_eval_four_tiers.py"
    spec = importlib.util.spec_from_file_location("_rnb_eval_four_tiers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_rnb_eval_four_tiers"] = module
    spec.loader.exec_module(module)
    return module


def _make_run(root: Path, arm: str, run_name: str, seed: int) -> Path:
    """A run directory with a configuration and one epoch."""
    run_dir = root / f"{arm}_s{seed}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"model": "simvp_v2_nwp_concat", "model_kwargs": {}}),
        encoding="utf-8",
    )
    (run_dir / "epoch_00.pt").write_bytes(b"not a real checkpoint")
    return run_dir


def test_two_arms_with_the_same_run_name_get_different_output_paths():
    """The core of the defect: a shared name must not mean a shared file."""
    scorer = _load_scorer()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Two arms, two seeds, one shared name -- which is what the
        # wall-clock stamp produces when the runs start together.
        _make_run(root, "fss35", "simvp_v2_nwp_concat_1793694077", 0)
        _make_run(root, "fss35", "simvp_v2_nwp_concat_1793694077", 2)
        _make_run(root, "fss35", "simvp_v2_nwp_concat_1793694075", 1)
        _make_run(root, "scale3", "simvp_v2_nwp_concat_1793694077", 0)

        runs = scorer.find_runs(root)
        arms = [arm for arm, _ in runs]
        assert len(runs) == 4, f"expected four runs, found {len(runs)}"

        # The arm is the directory above the run directory, and the two
        # seeds of one arm are told apart by it.
        assert "fss35_s0" in arms and "fss35_s2" in arms
        assert "scale3_s0" in arms

        # The output path is built from the arm and the run directory,
        # so no two runs share one.
        paths = []
        for arm, run_dir in runs:
            paths.append((root / "eval" / arm / run_dir.name).resolve())
        assert len(set(paths)) == len(paths), (
            "two runs were given the same output directory"
        )


def test_the_arm_is_the_directory_above_the_run_directory():
    """The arm is read from the parent, not from the run's own name."""
    scorer = _load_scorer()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_run(root, "lag11", "simvp_v2_assimilated_1793701814", 3)
        runs = scorer.find_runs(root)
        assert len(runs) == 1
        arm, run_dir = runs[0]
        assert arm == "lag11_s3"
        assert run_dir.name == "simvp_v2_assimilated_1793701814"


def test_a_run_without_a_configuration_is_reported_not_scored():
    """A directory with no configuration is skipped, not guessed at."""
    scorer = _load_scorer()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        broken = root / "fss35_s0" / "simvp_v2_nwp_concat_1793694077"
        broken.mkdir(parents=True)
        (broken / "epoch_00.pt").write_bytes(b"not a real checkpoint")
        assert scorer.read_meta(broken) is None
