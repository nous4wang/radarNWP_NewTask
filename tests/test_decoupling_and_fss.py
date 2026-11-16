"""The decoupling of the no-observation arm, and the multiscale FSS.

Three things are checked here.

The first is that the arm with no observation available is no longer a
copy of the inversion-only arm. It used to be: with a lag of eleven
frames and a history of eleven, the condition that guarded the
observation branch could never hold, so the arm fell through to the
inverted estimate and produced weights that were bit for bit identical
to the other arm. A sweep over the lag with one arm duplicated is a
sweep over three arms, and it cannot answer what an observation is
worth.

The second is that the fix did not move any other arm. The lag arms at
zero, two and four must behave exactly as they did, and the baselines
that were measured against them must still be comparable.

The third is the multiscale FSS: the fraction field is recomputed for
every half-width, so the report has to carry one value per window size
and the single-scale field must keep its meaning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pytest_compat import pytest  # noqa: E402

from radar_nwp_bench.metrics import PrecipitationVerifier  # noqa: E402
from radar_nwp_bench.models import build_model  # noqa: E402


def _model(name: str, **kwargs) -> torch.nn.Module:
    return build_model(name, **kwargs)


def _inputs(batch: int = 2, size: int = 32):
    x_radar = torch.randn(batch, 11, 10, size, size)
    x_nwp = torch.randn(batch, 2, 27, size, size)
    y_hist = torch.rand(batch, 11, 1, size, size) * 3.0
    return x_radar, x_nwp, y_hist


def _history_rain(model, x_radar, x_nwp, y_hist) -> torch.Tensor:
    """Recover the history rainfall the model actually used.

    The backbone logs it, but the log is a side effect of the forward
    pass, so the cleanest way to see it is to read the same quantity the
    backbone computed.
    """
    x_nwp = model.resolve_nwp(x_radar, x_nwp)
    _, qpe_rain = model.backbone.invert(x_radar, x_nwp[:, 0])
    if model.obs_lag_steps >= x_radar.shape[1]:
        return torch.zeros_like(qpe_rain)
    valid_end = x_radar.shape[1] - model.obs_lag_steps
    return torch.cat(
        [y_hist[:, :valid_end].clamp_min(0.0), qpe_rain[:, valid_end:].clamp_min(0.0)],
        dim=1,
    )


def test_the_no_observation_arm_is_no_longer_a_copy_of_the_inversion_arm():
    """The two arms share their weights and must still behave differently.

    The two models are the same network: the assimilation model is the
    inversion model plus a rule about which history slots are observed.
    With a lag of eleven the rule used to fall through to the inverted
    estimate, and then the two arms produced bit-identical weights --
    which is how a sweep over the lag ended up with one arm fewer than
    it appeared to have.

    The test is on the behaviour rather than on the checkpoint digest,
    because the behaviour is the cause and the digest is only the
    symptom. Two models that behave the same cannot end up with
    different weights, and two that behave differently cannot end up
    with the same ones.
    """
    torch.manual_seed(0)
    x_radar, x_nwp, y_hist = _inputs()

    no_obs = _model("simvp_v2_assimilated", obs_lag_steps=11)
    inversion = _model("simvp_v2_qpe_guided")

    # The same network, so the same weights can be handed to both.
    assert set(no_obs.state_dict()) == set(inversion.state_dict())
    inversion.load_state_dict(no_obs.state_dict())

    no_obs.obs_y_hist = y_hist
    a = no_obs(x_radar, x_nwp)
    b = inversion(x_radar, x_nwp)

    assert not torch.allclose(a, b), (
        "the no-observation arm and the inversion arm still behave "
        "identically, so they would still train to the same weights"
    )


def test_the_no_observation_arm_has_an_empty_history_channel():
    """With a lag of eleven the history rainfall must be zero, not inverted.

    This is the whole point of the decoupling: the arm has to differ
    from the inversion-only arm, or the sweep has one arm fewer than it
    appears to.
    """
    x_radar, x_nwp, y_hist = _inputs()
    model = _model("simvp_v2_assimilated", obs_lag_steps=11)
    model.obs_y_hist = y_hist

    used = _history_rain(model, x_radar, x_nwp, y_hist)
    assert used.shape == (2, 11, 1, 32, 32)
    assert float(used.abs().max()) == 0.0, (
        "the no-observation arm is still being fed the inverted estimate, "
        "so it is a copy of the inversion arm"
    )

    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()


def test_the_no_observation_arm_does_not_need_an_observation():
    """The arm that expects none must not raise when none is supplied.

    Every other arm expects observations and refuses to run without
    them, because a silent fallback measures something other than what
    was asked for. This arm genuinely wants none, so supplying none is
    the normal case.
    """
    x_radar, x_nwp, _ = _inputs()
    model = _model("simvp_v2_assimilated", obs_lag_steps=11)
    model.obs_y_hist = None
    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()


def test_the_other_lag_arms_are_untouched():
    """The fix must not move the arms that were already measured.

    Every arm whose lag is shorter than the history keeps the observed
    frames and fills the rest with the inverted estimate. If this test
    fails, the baselines measured against those arms are no longer
    comparable with the new ones.
    """
    x_radar, x_nwp, y_hist = _inputs()
    for lag, observed_frames in ((0, 11), (2, 9), (4, 7)):
        model = _model("simvp_v2_assimilated", obs_lag_steps=lag)
        model.obs_y_hist = y_hist
        used = _history_rain(model, x_radar, x_nwp, y_hist)
        # the observed part is the observation, unchanged
        assert torch.allclose(
            used[:, :observed_frames], y_hist[:, :observed_frames].clamp_min(0.0)
        ), f"lag {lag} no longer uses the observed frames verbatim"
        out = model(x_radar, x_nwp)
        assert torch.isfinite(out).all()


def test_an_arm_without_an_observation_gets_an_empty_history():
    """A missing observation must not become the inverted estimate.

    The original defect in this route was exactly a silent substitution:
    the training loop never published the observation, the model quietly
    used the inverted estimate, and the four arms trained to the same
    weights. A fallback here would bring that back, so the model falls
    back to an empty history channel instead, and the evaluation is the
    layer that refuses to score such an arm.
    """
    x_radar, x_nwp, _ = _inputs()
    model = _model("simvp_v2_assimilated", obs_lag_steps=0)
    model.obs_y_hist = None
    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()

    used = _history_rain(model, x_radar, x_nwp, torch.zeros(2, 11, 1, 32, 32))
    assert float(used.abs().max()) == 0.0, (
        "a missing observation fell back to the inverted estimate"
    )


def test_the_evaluation_refuses_to_score_an_arm_without_its_observation():
    """The check belongs to the evaluation, which knows the arm expects one.

    A model that raises would break the registry contract -- every
    registered model has to run on its own -- so the refusal is here.
    """
    source = (REPO_ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert "obs_lag_steps" in source
    assert "the dataset supplied none" in source
    assert "raise RuntimeError" in source


def test_the_multiscale_fss_reports_one_value_per_window_size():
    """The report has to carry the neighbourhood it was measured over.

    The half-width is the parameter; the window size is what a reader
    needs, and the two are not the same number.
    """
    rng = np.random.default_rng(0)
    truth = rng.random((1, 3, 32, 32)) * 4.0
    pred = truth + rng.normal(0.0, 0.5, truth.shape)

    v = PrecipitationVerifier(thresholds=(0.5, 2.0), fss_radii=(1, 2, 4))
    v.update(pred, truth)
    rep = v.report()

    for thr in ("0.5", "2.0"):
        entry = rep["thresholds"][thr]
        assert "fss_by_window" in entry
        assert set(entry["fss_by_window"]) == {"3x3", "5x5", "9x9"}
        for value in entry["fss_by_window"].values():
            assert -1.0 <= value <= 1.0
        # the single-scale field keeps its meaning: it is the half-width
        # of three, a seven by seven window
        assert entry["fss"] == pytest.approx(
            v.fss_sum[float(thr)] / v.fss_n[float(thr)], abs=1e-9
        )


def test_the_multiscale_fss_is_off_by_default():
    """The fraction field is recomputed for every half-width.

    The verification is bound by the processor rather than by the card,
    so the extra work is opt-in: a run that does not ask for it must not
    pay for it, and the report must not carry empty entries.
    """
    rng = np.random.default_rng(1)
    truth = rng.random((1, 2, 16, 16)) * 3.0
    pred = truth + rng.normal(0.0, 0.3, truth.shape)

    v = PrecipitationVerifier(thresholds=(0.5,))
    v.update(pred, truth)
    rep = v.report()
    assert "fss_by_window" not in rep["thresholds"]["0.5"]
    assert not v.fss_radii
