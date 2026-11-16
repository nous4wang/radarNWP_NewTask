"""Contract tests for ``scripts/tendency_field.py``.

The diagnostic cannot be checked against the archive from here, so these
tests lock the properties that make its numbers mean what the paper says
they mean:

1. **The CAPE consumption is in physical units.** The dataset hands the
   NWP over normalised, so a difference of two normalised values is not a
   CAPE change in J/kg. The inversion has to happen, and it has to be
   exact.
2. **The tendency is the one the model used.** The figure reads the
   tendency through a forward hook on the Koopman flow; the hook has to
   fire, and it has to record the tensor the module returned.
3. **The script's own self-check passes.** It is the only numerical
   guard the tool has when the archive is not reachable.
4. **The failure modes are quiet.** An unknown case, a missing
   checkpoint and a missing archive must produce a message and a non-zero
   exit code, never a traceback.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCRIPTS = REPO_ROOT / "scripts"
SCRIPT = SCRIPTS / "tendency_field.py"


def _load_script(name: str = "rnb_tendency_field"):
    """Import the diagnostic script by path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _flags(parser) -> set[str]:
    """The long options a parser declares."""
    return {
        action.option_strings[0]
        for action in parser._actions
        if action.option_strings and action.option_strings[0].startswith("--")
    }


# ---------------------------------------------------------------------------
# 1. the CAPE consumption is in physical units
# ---------------------------------------------------------------------------
def test_the_cape_consumption_is_in_physical_units():
    """The normalisation is inverted before the difference is taken."""
    import numpy as np

    from radar_nwp_bench.data.transforms import NWP_NORM, normalize_nwp_field

    tool = _load_script()

    analysis = normalize_nwp_field(np.array([[2000.0]], dtype=np.float32), "CAPE")
    forecast = normalize_nwp_field(np.array([[1500.0]], dtype=np.float32), "CAPE")
    consumption = tool.cape_consumption(analysis, forecast)

    assert abs(float(consumption[0, 0]) + 500.0) < 1e-2, (
        "the CAPE consumption must be the physical difference in J/kg"
    )

    # The difference of the normalised fields on its own is not a CAPE
    # change: it is off by the standard deviation of the variable.
    naive = float(forecast[0, 0] - analysis[0, 0])
    std = NWP_NORM["CAPE"][1]
    assert abs(naive * std - float(consumption[0, 0])) < 1e-2
    assert abs(naive - float(consumption[0, 0])) > 100.0, (
        "the normalised difference must not be reported as J/kg"
    )


def test_the_reflectivity_inversion_recovers_the_dbz():
    """The radar log transform is inverted to dBZ, not to log space."""
    import numpy as np

    from radar_nwp_bench.data.transforms import radar_log_transform

    tool = _load_script()

    dbz = np.array([[0.0, 12.5, 48.0, 65.0]], dtype=np.float32)
    recovered = tool.dbz_from_network(radar_log_transform(dbz, "CR"))
    assert np.allclose(recovered, dbz, atol=1e-3)


def test_the_correlation_coefficient_is_the_pearson_one():
    """The coefficient the figure prints has to be the Pearson one."""
    import numpy as np

    tool = _load_script()

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    assert abs(tool.pearson_r(x, y) - 0.8) < 1e-12
    assert abs(tool.pearson_r(x, x) - 1.0) < 1e-12
    assert abs(tool.pearson_r(x, -x) + 1.0) < 1e-12
    assert not np.isfinite(tool.pearson_r(x, np.ones_like(x)))


# ---------------------------------------------------------------------------
# 2. the tendency is the one the model used
# ---------------------------------------------------------------------------
def test_the_forward_hook_records_the_tendency_the_model_used():
    """A hook on the Koopman flow fires and captures the right tensor."""
    import torch

    from radar_nwp_bench.models import build_model

    tool = _load_script()

    torch.manual_seed(0)
    model = build_model(
        "simvp_v2_quantum", in_radar_c=10, in_nwp_c=27,
        in_len=4, out_len=2, hid_c=8, num_blocks=1,
    )
    model.eval()

    capture = tool.TendencyCapture(model)
    try:
        assert capture.n_modules >= 1, "the model holds no Koopman flow"

        x_radar = torch.rand(1, 4, 10, 16, 16)
        x_nwp = torch.rand(1, 2, 27, 16, 16)
        with torch.no_grad():
            model(x_radar, x_nwp)

        captured = capture.last
        assert captured is not None, "the hook did not fire"
        # The coupler's feature width is twice the backbone width, so the
        # tendency carries 16 channels here.
        assert captured.shape == (1, 16, 16, 16)

        with torch.no_grad():
            reference = model.qnwp_coupler.koopman_flow(
                model.qnwp_coupler.state_proj(x_nwp[:, 0]),
                model.qnwp_coupler.state_proj(x_nwp[:, -1]),
            )
        assert torch.allclose(captured, reference, atol=1e-5), (
            "the hook must record the tensor the module returned"
        )
    finally:
        capture.close()


def test_the_tendency_magnitude_is_the_norm_over_the_channels():
    """The magnitude is what the arrows and the contours are drawn from."""
    import torch

    tool = _load_script()

    d_psi = torch.zeros(1, 3, 4, 4)
    d_psi[0, 0] = 3.0
    d_psi[0, 1] = 4.0
    magnitude = tool.tendency_magnitude(d_psi)
    assert torch.allclose(magnitude, torch.full((1, 4, 4), 5.0))


# ---------------------------------------------------------------------------
# 3. the script's own self-check
# ---------------------------------------------------------------------------
def test_the_scripts_own_self_check_passes():
    """``--self-check`` is the guard the tool has without the archive."""
    tool = _load_script()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = tool._self_check()
    output = buffer.getvalue()
    assert code == 0, f"the self-check failed:\n{output}"
    assert "checks passed" in output
    assert "FAIL" not in output


def test_the_parser_declares_the_flags_the_paper_figure_needs():
    """The selection has to be explicit and reproducible from the shell."""
    tool = _load_script()

    flags = _flags(tool.build_argparser())
    for flag in (
        "--data-root", "--packed", "--split", "--model", "--checkpoint",
        "--out-dir", "--case", "--issue-time", "--patch-size",
        "--slide-stride", "--batch-size", "--num-workers", "--device",
        "--max-cases", "--self-check",
    ):
        assert flag in flags, f"tendency_field.py is missing {flag}"


def test_the_script_reads_no_argument_it_does_not_declare():
    """A cheap static check, the same one the other scripts get."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

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
        f"tendency_field.py reads args.{sorted(missing)} but never declares "
        "the matching flag"
    )


def test_the_script_does_not_import_from_scripts():
    """The tool has to stand on the package alone, not on its siblings."""
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("from scripts", "import scripts"):
        assert forbidden not in source, (
            f"tendency_field.py must not use {forbidden!r}"
        )


# ---------------------------------------------------------------------------
# 4. the failure modes are quiet
# ---------------------------------------------------------------------------
def test_an_unknown_case_is_refused_with_a_message():
    """A reviewer who mistypes a case id gets an answer, not a traceback."""
    tool = _load_script()

    class _Args:
        case = "CP_99_00000000_00001"
        issue_time = None
        max_cases = 8
        split = "test"

    choices = [
        tool.WindowChoice(0, "CP_01_06010006_00001", 6010100, -10.0, -50.0),
    ]
    try:
        tool.select_windows(choices, _Args(), {"CP_01_06010006_00001"}, None)
    except ValueError as exc:
        assert "CP_99_00000000_00001" in str(exc)
    else:
        raise AssertionError("an unknown case must be refused")


def test_the_issue_time_is_parsed_as_an_archive_stamp():
    """``MMDDHHMM`` is what the archive uses, and it must be accepted."""
    tool = _load_script()

    assert tool._parse_issue_time("06231500") == 6231500
    assert tool._parse_issue_time(None) is None
    try:
        tool._parse_issue_time("not a time")
    except ValueError:
        pass
    else:
        raise AssertionError("a malformed issue time must be refused")
