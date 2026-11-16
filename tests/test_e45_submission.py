"""E45 -- the submission plumbing's own regression tests.

Every test here corresponds to a bug that actually happened during
E45's first run, and every one of them was SILENT: the jobs ran, the
logs looked healthy, and the results were wrong or missing.  That is
why they are tests of the plumbing and not of the science.

The four bugs:

1. ``--export`` splits its list on COMMAS, so an arm list passed that
   way arrives truncated to its first element.  Every training job then
   trained exactly one arm, and four arms per seed never ran at all.
2. The training script had lost its ``#SBATCH --array`` directive, so
   it ran as a single job with no array task id.
3. The topology flags travelled through environment variables the
   submitter never exported, so both topology arms received 0.0 --
   which means "off" -- and neither tested the mechanism it was named
   after.
4. The mechanism probe was called with ``--runs`` (an argument it does
   not have) and with ``model=`` inside the arm spec (an inline option
   it does not accept), so the job died on its first probe.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
TOOLS = REPO / "tools"

#: The repository root as bash spells it on this machine.  The bash
#: that Python starts is the PortableGit one, which mounts the drive
#: at /mnt/<letter> rather than /<letter>.
BASH_REPO = "/mnt/r/1radarNWPprediction/radar-nwp-bench"


def _bash(script: str) -> str:
    """Run a snippet inside the repository, in bash, and return its output.

    The tests shell out to bash, and bash on this machine mounts the
    repository at /r/<...> while Python sees a Windows path.  Neither
    can be handed to the other as a working directory -- the Windows
    one has backslashes that bash reads as escapes, and the /r one is
    not a directory as far as Windows is concerned -- so the whole
    operation is given to bash in one piece, with the repository
    entered from bash's own point of view.
    """
    result = subprocess.run(
        ["bash", "-c", f"cd {BASH_REPO} && {script}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"the bash snippet failed:\n{result.stderr}")
    return result.stdout


def _load_probe():
    """The mechanism probe, loaded as a module (it has no package)."""
    path = TOOLS / "_e45_mechanism_probe.py"
    spec = importlib.util.spec_from_file_location("_e45_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# the arm table
# --------------------------------------------------------------------------

def _arms_table():
    """The arm table, parsed out of the file that defines it.

    The file is read rather than sourced, because the bash that Python
    starts here (the PortableGit one) does not pass a sourced function
    into a command substitution, while the remote bash does.  Parsing
    the file tests the same thing the scripts read -- the rows and
    their order -- without depending on which bash is in front of it.
    The remote bash reads the same rows, which is checked by the
    scripts' own run.
    """
    text = _read(SCRIPTS / "_e45_arms.sh")
    block = re.search(
        r"E45_ARM_TABLE=\((.*?)\n\)", text, re.S
    )
    assert block, "the arm table was not found in _e45_arms.sh"
    rows = {}
    for line in block.group(1).splitlines():
        line = line.strip().strip('"')
        if not line:
            continue
        arm, model, loss, train_flags, probe_flags = line.split("|")
        rows[len(rows)] = {
            "arm": arm, "model": model, "loss": loss,
            "train_flags": train_flags, "probe_flags": probe_flags,
        }
    return len(rows), rows


def test_arm_table_has_seven_arms_in_the_pre_registered_order():
    n_arms, rows = _arms_table()
    assert n_arms == 7
    assert [rows[i]["arm"] for i in range(7)] == [
        "B0", "A1", "A2", "A3a", "A3b", "A4", "A4b",
    ]


def test_topology_flags_belong_to_the_arms_not_the_environment():
    """Bug 3: the flags are part of the arm's definition."""
    _, rows = _arms_table()
    assert rows[3]["train_flags"] == "--topo-clamp-mm 20"
    assert rows[4]["train_flags"] == (
        "--topo-truth-smooth-sigma 1.0 --topo-area-normalize"
    )
    # No other arm carries a topology flag.
    for index in (0, 1, 2, 5, 6):
        assert rows[index]["train_flags"] == ""


def test_every_non_topology_arm_uses_the_load_bearing_loss():
    _, rows = _arms_table()
    for index in (0, 1, 2, 5, 6):
        assert rows[index]["loss"] == "balanced"
    assert rows[3]["loss"] == "topo"
    assert rows[4]["loss"] == "topo"


def test_probe_flags_mirror_the_training_flags():
    """The probe rebuilds the arm's term, so it needs the same flags."""
    _, rows = _arms_table()
    assert rows[3]["probe_flags"] == "topo_clamp_mm=20.0"
    assert rows[4]["probe_flags"] == (
        "topo_truth_smooth_sigma=1.0,topo_area_normalize=1"
    )


def test_arm_table_index_out_of_range_fails():
    script = (
        'source scripts/_e45_arms.sh\n'
        'e45_arm 7 && echo REACHED || echo REFUSED\n'
    )
    out = _bash(script)
    assert "REFUSED" in out


# --------------------------------------------------------------------------
# the submission plumbing
# --------------------------------------------------------------------------

def test_no_arm_list_travels_through_export():
    """Bug 1: --export splits on commas, so a list cannot go through it."""
    for name in ("e45_train.sbatch", "e45_stage2.sbatch", "_submit_e45.sbatch"):
        text = _read(SCRIPTS / name)
        for line in text.splitlines():
            # Only the COMMAND lines count.  The comments explain the
            # bug and quote the old form on purpose.
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for match in re.finditer(r"--export=([^\s\\]+)", stripped):
                exported = match.group(1)
                # Only SEED (a single integer) may be exported.  Any
                # other name is a variable that would have to carry a
                # list.
                for item in exported.split(","):
                    key = item.split("=", 1)[0]
                    assert key in ("ALL", "SEED", "START_SEED"), (
                        f"{name} exports {key!r}; only SEED may cross --export"
                    )


def test_training_script_has_the_array_directive():
    """Bug 2: without --array there is no array task id."""
    text = _read(SCRIPTS / "e45_train.sbatch")
    assert re.search(r"^#SBATCH --array=0-6$", text, re.M)


def test_training_script_refuses_to_run_without_an_array_task_id():
    text = _read(SCRIPTS / "e45_train.sbatch")
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "must run as an array task" in text


def test_training_script_reads_the_arm_table():
    text = _read(SCRIPTS / "e45_train.sbatch")
    assert "scripts/_e45_arms.sh" in text
    # The arm table is the only place the index -> arm mapping appears.
    assert not re.search(r'^\s*\d\)\s*ARM="', text, re.M)


def test_stage2_reads_the_same_arm_table():
    text = _read(SCRIPTS / "e45_stage2.sbatch")
    assert "scripts/_e45_arms.sh" in text


def test_probe_is_called_with_arguments_it_has():
    """Bug 4: --runs does not exist, and model= is not an inline option."""
    text = _read(SCRIPTS / "e45_stage2.sbatch")
    assert "--runs " not in text
    assert "model=" not in text


def test_probe_accepts_the_arm_specs_the_stage2_builds():
    probe = _load_probe()
    for spec_text in (
        "A1", "A2", "A3a|topo_clamp_mm=20.0",
        "A3b|topo_truth_smooth_sigma=1.0,topo_area_normalize=1",
        "A4", "A4b",
    ):
        name, options = probe._parse_arm_spec(spec_text)
        assert name in probe.ARMS
        if name == "A3a":
            assert options["topo_clamp_mm"] == 20.0
        if name == "A3b":
            assert options["topo_truth_smooth_sigma"] == 1.0
            assert options["topo_area_normalize"] is True


def test_probe_rejects_a_model_option_inside_the_arm_spec():
    """The old form is rejected, so it cannot come back unnoticed."""
    probe = _load_probe()
    with pytest.raises(SystemExit):
        probe._parse_arm_spec("A1|model=amp_monotone_gate")


def test_submitter_stops_on_a_refused_submission():
    """A refusal must not turn into an endless retry loop."""
    text = _read(SCRIPTS / "_submit_e45.sbatch")
    assert "SUBMIT REFUSED" in text
    # The refusal path returns non-zero, and the caller exits.
    assert "FATAL" in text


def test_submitter_reads_a_padded_job_state():
    """sacct pads its column, so the state arrives with spaces on both sides.

    A pattern of COMPLETED* does not match " COMPLETED " (the pattern
    has to match the whole word), so the value is trimmed before it is
    matched.  Getting this wrong made the submitter declare a finished
    training array FAILED and stop the whole batch.
    """
    text = _read(SCRIPTS / "_submit_e45.sbatch")
    assert "tr -d '[:space:]'" in text
    # The match itself is on the exact word.
    assert "COMPLETED) return 0 ;;" in text


def test_submitter_takes_the_job_id_from_the_first_field():
    """sbatch --parsable prints "jobid;cluster"."""
    text = _read(SCRIPTS / "_submit_e45.sbatch")
    assert 'first="${out%%;*}"' in text


def test_submitter_runs_the_seeds_one_at_a_time():
    """Two seeds' arrays would already be fourteen submitted sub-tasks."""
    text = _read(SCRIPTS / "_submit_e45.sbatch")
    assert "for SEED in $(seq" in text
    assert "wait_for" in text


# --------------------------------------------------------------------------
# the scripts parse
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "_e45_arms.sh", "e45_train.sbatch", "e45_stage2.sbatch",
        "e45_summary.sbatch", "_submit_e45.sbatch",
    ],
)
def test_scripts_parse_as_bash(name):
    result = subprocess.run(
        ["bash", "-c", f"cd {BASH_REPO} && bash -n scripts/{name}"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_merge_tool_merges_per_seed_files(tmp_path):
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "_e45_merge_under_test", TOOLS / "_e45_merge_struct.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first = tmp_path / "s0.json"
    second = tmp_path / "s1.json"
    first.write_text(json_str({"tags": {"A1_s0_ep00": {"pred": {"x": 1.0}}}}),
                    encoding="utf-8")
    second.write_text(json_str({"tags": {"A1_s1_ep00": {"pred": {"x": 2.0}}}}),
                     encoding="utf-8")
    out = tmp_path / "all.json"

    import json as _json
    argv = sys.argv
    sys.argv = ["merge", "--inputs", str(first), str(second), "--out", str(out)]
    try:
        assert module.main() == 0
    finally:
        sys.argv = argv
    merged = _json.loads(out.read_text(encoding="utf-8"))
    assert set(merged["tags"]) == {"A1_s0_ep00", "A1_s1_ep00"}


def json_str(obj) -> str:
    import json as _json
    return _json.dumps(obj)
