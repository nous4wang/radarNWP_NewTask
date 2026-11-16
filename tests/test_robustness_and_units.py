"""The properties the paper's claims rest on.

Each test here corresponds to a claim that is made in the write-up, and
each one failed at least once during development.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_parser(name: str, script: str, attr: str):
    """Load an argparse parser out of one of the scripts."""
    import importlib.util

    path = REPO_ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, attr)


def _flags(parser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }


# ---------------------------------------------------------------------------
# the partition
# ---------------------------------------------------------------------------
def test_split_cases_removes_the_irregular_cases_itself():
    """A caller cannot forget to exclude the cases with a hole.

    The robustness split is measured on cases the model never trained on.
    If the exclusion happened in the caller, one caller would eventually
    forget it and the robustness score would be contaminated by training.
    """
    from radar_nwp_bench.data.index import IRREGULAR_CASES, split_cases

    ids = [f"CP_01_{i:08d}_00001" for i in range(20)]
    irregular = sorted(IRREGULAR_CASES)[:3]
    parts = split_cases(ids + irregular)

    clean = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    assert not (clean & set(irregular)), "an irregular case stayed in the clean split"
    assert parts["robustness_missing"] == irregular


def test_the_irregular_cases_are_the_ones_that_actually_have_a_hole():
    """The list must not be a guess.

    The ids in an earlier revision did not exist in the archive at all, so
    the split excluded nothing and the robustness split raised an error.
    The names have the archive's own suffix.
    """
    from radar_nwp_bench.data.index import IRREGULAR_CASES

    assert len(IRREGULAR_CASES) > 0
    for case in IRREGULAR_CASES:
        assert case.startswith("CP_01_"), case
        assert case.endswith("_00001"), (
            f"{case} does not carry the archive's suffix"
        )


def test_the_three_splits_do_not_overlap():
    from radar_nwp_bench.data.index import split_cases

    ids = [f"CP_01_{i:08d}_00001" for i in range(30)]
    parts = split_cases(ids)
    assert not (set(parts["train"]) & set(parts["val"]))
    assert not (set(parts["train"]) & set(parts["test"]))
    assert not (set(parts["val"]) & set(parts["test"]))


# ---------------------------------------------------------------------------
# the timestamp arithmetic
# ---------------------------------------------------------------------------
def test_a_step_across_midnight_is_six_minutes():
    """The old arithmetic read it as 4566 minutes.

    The stamp is MMDD*10000 + HHMM, so (stamp // 100) * 60 + (stamp % 100)
    folds the month into the hours. The archive's own minute arithmetic
    handles the day and month rollover.
    """
    from radar_nwp_bench.data.index import add_minutes, minutes_between

    before = 6232354   # 06-23 23:54
    after = 6240000    # 06-24 00:00
    assert minutes_between(after, before) == 6
    assert add_minutes(before, 6) == after


def test_a_step_across_a_month_boundary_is_six_minutes():
    from radar_nwp_bench.data.index import add_minutes, minutes_between

    before = 5312354   # 05-31 23:54
    after = 6010000    # 06-01 00:00
    assert minutes_between(after, before) == 6
    assert add_minutes(before, 6) == after


def test_the_minutes_round_trip_over_the_whole_archive():
    from radar_nwp_bench.data.index import add_minutes, minutes_between

    # One stamp per month the archive covers, plus the boundaries. The
    # stamp is MMDD*10000 + HHMM, so the month and the day are packed
    # together and a shift has to carry across both.
    stamps = [
        5171710, 5312354, 6010000, 6232354, 6240000, 6300000,
        7000000, 7010000, 7312359, 8010000, 8122359, 9010000,
        9292359,
    ]
    for stamp in stamps:
        for delta in (0, 6, 12, 60, 360):
            shifted = add_minutes(stamp, delta)
            assert minutes_between(shifted, stamp) == delta, (stamp, delta)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------
def test_the_robustness_split_is_reachable_from_the_command_line():
    """It was documented as an evaluation mode and argparse refused it."""
    train = _load_parser("rnb_tr_rob", "train.py", "build_argparser")()
    evaluate = _load_parser("rnb_ev_rob", "evaluate.py", "build_argparser")()

    for parser in (train, evaluate):
        args = parser.parse_args(
            ["--model", "simvp_v2", "--data-root", "x",
             "--split", "robustness_missing"]
        )
        assert args.split == "robustness_missing"


def test_the_interpolation_can_be_switched_off():
    """The robustness claim needs a way to measure the tolerance itself."""
    train = _load_parser("rnb_tr_nointerp", "train.py", "build_argparser")()
    evaluate = _load_parser("rnb_ev_nointerp", "evaluate.py", "build_argparser")()

    for parser in (train, evaluate):
        assert "--no-interpolate" in _flags(parser)
        args = parser.parse_args(
            ["--model", "simvp_v2", "--data-root", "x"]
        )
        assert args.no_interpolate is False
        args = parser.parse_args(
            ["--model", "simvp_v2", "--data-root", "x", "--no-interpolate"]
        )
        assert args.no_interpolate is True


def test_the_dry_patch_filter_is_reachable_from_the_command_line():
    train = _load_parser("rnb_tr_dry", "train.py", "build_argparser")()
    assert "--filter-dry" in _flags(train)
    args = train.parse_args(
        ["--model", "simvp_v2", "--data-root", "x", "--filter-dry"]
    )
    assert args.filter_dry is True
    assert 0.0 <= args.dry_keep_prob <= 1.0
    assert args.rain_threshold > 0.0


# ---------------------------------------------------------------------------
# the units
# ---------------------------------------------------------------------------
def test_the_binary_term_can_express_a_dry_pixel():
    """log1p(mm) is never negative, so sigmoid() was never below 0.5.

    A dry pixel could therefore never be predicted with a confidence below
    one half, and the term saturated. The decision boundary now sits at
    the rain threshold, so a prediction below the threshold produces a
    logit below zero.
    """
    import torch

    from radar_nwp_bench.metrics.losses import PrecipitationLoss

    loss_fn = PrecipitationLoss(rain_threshold=0.1, bce_weight=1.0)

    dry = torch.zeros(1, 1, 1, 4, 4)
    wet = torch.full((1, 1, 1, 4, 4), 1.0)
    target_dry = torch.zeros(1, 1, 1, 4, 4)
    target_wet = torch.full((1, 1, 1, 4, 4), 1.0)

    # A dry prediction against a dry target is the best the term can do.
    _, components = loss_fn(dry, target_dry)
    dry_loss = components["bce"]

    # The same prediction against a wet target is penalised more.
    _, components = loss_fn(dry, target_wet)
    wet_loss = components["bce"]

    assert wet_loss > dry_loss, (
        "the binary term does not distinguish a dry from a wet target"
    )


def test_the_report_carries_the_false_alarm_rate_per_lead_time():
    """A paper that claims a lower false-alarm rate needs the rate."""
    import numpy as np

    from radar_nwp_bench.metrics.precipitation import PrecipitationVerifier

    verifier = PrecipitationVerifier(thresholds=(0.5,))
    pred = np.zeros((1, 3, 8, 8), dtype=np.float32)
    truth = np.zeros((1, 3, 8, 8), dtype=np.float32)
    # The first lead time gets the forecast right and the second one does
    # not, so the two must show different false-alarm rates.
    truth[:, 0, 0, 0] = 2.0
    pred[:, 0, 0, 0] = 2.0
    pred[:, 1, 0, 0] = 2.0
    verifier.update(pred, truth)

    report = verifier.report()
    assert "lead_scores" in report
    leads = report["lead_scores"]
    assert set(leads) == {"0", "1", "2"}
    for lead, per_threshold in leads.items():
        assert "csi" in per_threshold["0.5"]
        assert "far" in per_threshold["0.5"]
    # The first lead time is a hit with no false alarm; the second is a
    # pure false alarm.
    assert leads["0"]["0.5"]["far"] == 0.0
    assert leads["1"]["0.5"]["far"] == 1.0
    assert leads["0"]["0.5"]["csi"] == 1.0
    assert leads["1"]["0.5"]["csi"] == 0.0


# ---------------------------------------------------------------------------
# the analytical baseline
# ---------------------------------------------------------------------------
def test_the_analytical_baseline_sees_physical_dbz_not_the_compressed_scale():
    """The Z-R inversion is a relation in dBZ, and the input is compressed.

    The dataset hands every model a radar field that has been through
    ``log1p(dBZ) / 6``.  The physics baseline inverts Z-R to get rain, so it
    has to undo that transform first.  Fed the compressed value, a 45 dBZ
    storm is read as 0.5 dBZ and comes back as 0.004 mm per 6 minutes --
    two orders of magnitude below the weakest scoring threshold.  Every pixel
    is then a miss and the baseline scores exactly zero, which is what it
    did for the whole first matrix.
    """
    import numpy as np
    import torch

    from radar_nwp_bench.models import build_model
    from radar_nwp_bench.data.transforms import radar_log_transform

    model = build_model("pysteps_adv")
    assert getattr(model, "is_learnable", True) is False, (
        "pysteps_adv is an analytical baseline and must not train"
    )

    # A uniform 45 dBZ field, encoded the way the dataset encodes it.
    dbz = np.full((2, 32, 32), 0.0, dtype=np.float32)
    dbz[:] = 45.0
    encoded = radar_log_transform(dbz, "CR")
    x_radar = torch.zeros(1, 2, 10, 32, 32)
    x_radar[0, :, 6] = torch.from_numpy(encoded)

    with torch.no_grad():
        out = model(x_radar, torch.zeros(1, 2, 27, 32, 32))

    assert out.shape == (1, 10, 1, 32, 32)
    # The Z-R relation for 45 dBZ gives a rain rate of tens of mm/h, so a
    # 6-minute accumulation is well above 1 mm.  The broken version produced
    # 0.004 mm, which is below the weakest threshold in the whole benchmark.
    assert float(out.max()) > 1.0, (
        "the analytical baseline is not inverting Z-R on physical dBZ; "
        f"got {float(out.max()):.6f} mm for a 45 dBZ storm"
    )

    # And the round trip through the transform must be the identity.
    back = np.expm1(encoded * 6.0)
    assert np.allclose(back, dbz, atol=1e-3)


def test_the_analytical_baseline_advects_the_storm_forwards():
    """A storm that moves right must be forecast to the right of where it is.

    ``grid_sample`` reads the source at the coordinate it is handed, so
    sampling at ``base + d`` moves the content to ``x - d``.  The previous
    version sampled on the ``+`` side, which ran every storm backwards: the
    forecast centroid moved from 20 px towards 12 px while the storm itself
    was travelling from 20 px towards 28 px.
    """
    import numpy as np
    import torch

    from radar_nwp_bench.models import build_model

    model = build_model("pysteps_adv")
    h = w = 64
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    def blob(cx):
        return 45.0 * np.exp(-(((xx - cx) ** 2 + (yy - 32.0) ** 2) / 72.0))

    x_radar = torch.zeros(1, 2, 10, h, w)
    # The storm sits at x=20 in the previous frame and at x=22 in the last
    # one: it is travelling to the right at 2 px per frame.  The field goes
    # in the way the dataset encodes it.
    x_radar[0, 0, 6] = torch.from_numpy(
        np.log1p(np.clip(blob(20.0), 0.0, None)) / 6.0
    )
    x_radar[0, 1, 6] = torch.from_numpy(
        np.log1p(np.clip(blob(22.0), 0.0, None)) / 6.0
    )

    with torch.no_grad():
        out = model(x_radar, torch.zeros(1, 2, 27, h, w))

    def centroid(field):
        weight = field.clamp_min(0.0)
        total = weight.sum()
        return float((weight * xx).sum() / total)

    # The forecast has to move to the right, towards where the storm is
    # going, not to the left.
    c0 = centroid(out[0, 0, 0])
    c2 = centroid(out[0, 2, 0])
    assert c2 > c0, (
        "the analytical baseline advects the storm backwards: "
        f"centroid moved {c0:.1f} -> {c2:.1f}"
    )
    # Two frames of advection at 2 px per frame is about 4 px.
    assert 1.0 < (c2 - c0) < 8.0, (
        "the advection distance is off: "
        f"centroid moved {c0:.1f} -> {c2:.1f} over two frames"
    )


def test_a_degenerate_report_is_not_written_to_disk():
    """A score of zero everywhere is a broken model, not a weak one.

    The analytical baseline once produced a report in which every CSI was
    zero and every false-alarm rate was undefined: the forecast never
    crossed a threshold, because the Z-R inversion was applied to the
    compressed radar scale.  The report looked like a result -- the table
    printed "0.000" -- so it sat in the paper's main comparison for a whole
    matrix without anybody noticing that the row was a bug.

    The evaluation now refuses to write such a report.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "evaluate.py").read_text(encoding="utf-8")

    # The guard is in the script, and it fails the run rather than
    # returning a report.
    assert "degenerate" in source, (
        "evaluate.py has no guard against a report in which every CSI is zero"
    )
    assert "not a score" in source, (
        "the guard does not explain why an all-zero report is not a score"
    )
    # The guard has to run *before* the report is written.
    guard_at = source.index("degenerate")
    write_at = source.index("report written to")
    assert guard_at < write_at, (
        "the guard runs after the report is written, so it cannot stop it"
    )
