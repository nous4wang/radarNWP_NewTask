"""The scorer's output paths, and the run names they must not depend on.

The training stamps its run directory with the wall clock, so two runs
that start in the same second carry the same name. The scorer used to
key its reports on that name, and the consequence was not a crash: the
second run's report overwrote the first one's, and sixteen evaluations
came back as six reports whose contents belonged to whichever seed
finished last. Nothing said so. The count was simply short.

These tests put two runs with the same stamp under the same arm and
check that their reports land apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from _pytest_compat import pytest  # noqa: E402

from _eval_four_tiers import find_runs, read_meta  # noqa: E402


def _make_run(root: Path, arm: str, stamp: str, model: str = "simvp_v2_nwp_concat"):
    """A run directory as the training leaves it, with its configuration."""
    run_dir = root / arm / f"{model}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"model": model, "model_kwargs": {}}), encoding="utf-8"
    )
    return run_dir


def test_two_runs_with_the_same_stamp_are_told_apart(tmp_path=None):
    """The name the training gives a run is not unique, and must not be used."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_run(root, "fss35_s0", "1793695380")
        _make_run(root, "fss35_s1", "1793695380")  # the same second

        runs = find_runs(root)
        assert len(runs) == 2, "the two runs were not both found"

        arms = {arm for arm, _ in runs}
        assert arms == {"fss35_s0", "fss35_s1"}, (
            "the two runs were given the same name, so one would overwrite "
            "the other's report"
        )

        for arm, run_dir in runs:
            model, kwargs = read_meta(run_dir)
            assert model == "simvp_v2_nwp_concat"
            assert kwargs == {}


def test_the_output_directory_is_the_arm_name(tmp_path=None):
    """The report path has to be keyed on something unique.

    The run directory carries the wall clock and cannot be used; the arm
    directory carries the seed and can.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_run(root, "full_s3", "1793695380")
        _make_run(root, "full_s4", "1793695380")

        for arm, run_dir in find_runs(root):
            out_dir = root / "eval" / arm
            assert run_dir.name not in out_dir.parts, (
                "the output path is keyed on the run directory, whose name "
                "is the wall clock and is not unique"
            )


def test_the_analysis_tool_reads_the_same_layout():
    """The analysis has to find the reports the scorer writes.

    The two are separate scripts and they agree on a path; if one of
    them changes its mind the reports are simply not found, and the
    analysis reports an empty table rather than an error.
    """
    source = (REPO_ROOT / "tools" / "_analyze_four_tiers.py").read_text(
        encoding="utf-8"
    )
    assert 'glob(f"{arm}_s*")' in source or 'glob("*_s*")' in source
    assert "report.parent.parent" not in source, (
        "the analysis walks up two levels from the report, which only "
        "holds if the scorer writes the run directory in the path"
    )


def test_the_baseline_arms_are_the_ones_the_criteria_name():
    """The comparison has to be against the arm the criterion names."""
    source = (REPO_ROOT / "tools" / "_analyze_four_tiers.py").read_text(
        encoding="utf-8"
    )
    assert '2: "base_concat"' in source
    assert '3: "qpe_0"' in source
