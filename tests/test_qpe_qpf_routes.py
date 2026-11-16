"""Routes R1 and R2: the contracts that the models have to hold.

These are contract tests, not measurements. The measurements happen on
the server, where the archive is; here the point is that the pieces
fit together -- that the dataset hands out the history rainfall, that
the models take it the way they say they do, and that the objective can
read the inverted history.

The last test is the one that matters most for the paper. Route R2
consumes ground observations at inference, which is a different task
from a pure radar extrapolation. A model that changes the task has to
say so, because its scores cannot be placed next to the baselines'
without the reader drawing the wrong conclusion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _pytest_compat import pytest  # noqa: E402

from radar_nwp_bench.models import build_model  # noqa: E402
from radar_nwp_bench.models.common import QPEQPFBackbone  # noqa: E402


def _model(name: str, **kwargs):
    return build_model(
        name,
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10,
        hid_c=16, **kwargs,
    )


def test_the_backbone_is_registered_and_builds():
    assert build_model is not None
    model = _model("simvp_v2_qpe_guided")
    assert isinstance(model.backbone, QPEQPFBackbone)


def test_the_inference_contract_of_r1_is_unchanged():
    """R1 must take exactly what the baselines take.

    If it needed the history rainfall as an input, it would be route R2
    and the comparison against the baselines would be void.
    """
    model = _model("simvp_v2_qpe_guided")
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)

    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()
    assert float(out.detach().min()) >= 0.0


def test_the_inverted_history_is_published_for_the_objective():
    """The objective can only supervise the inversion if it can see it."""
    model = _model("simvp_v2_qpe_guided")
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)

    assert model.last_qpe_pred is None
    model(x_radar, x_nwp)
    assert model.last_qpe_pred is not None
    assert model.last_qpe_pred.shape == (2, 11, 1, 32, 32)


def test_the_two_stages_share_one_rainfall_head():
    """The shared head is the mechanism, so it has to be one head.

    Two heads would let the forecast stage hedge independently of the
    history stage, and the zero-displacement supervision would stop
    carrying over.
    """
    model = _model("simvp_v2_qpe_guided")
    heads = [
        module for name, module in model.named_modules()
        if name.endswith("shared_rain_head")
    ]
    assert len(heads) == 1


def test_the_history_supervision_reaches_the_gradient():
    """A term that does not reach the parameters is not a supervision."""
    from radar_nwp_bench.models.common import BalancedExtremePrecipLoss

    model = _model("simvp_v2_qpe_guided")
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    y_hist = torch.rand(2, 11, 1, 32, 32) * 3.0
    y_future = torch.rand(2, 10, 1, 32, 32) * 3.0

    pred = model(x_radar, x_nwp)
    loss = BalancedExtremePrecipLoss(alpha_csi=0.5)
    total = loss(pred, y_future) + 0.5 * loss(model.last_qpe_pred, y_hist)
    total.backward()

    grad = model.backbone.shared_rain_head.net[0].body[0].weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_the_adapted_objective_forwards_the_history():
    """The training loop reaches the loss through an adapter.

    The loss itself takes no keyword arguments, so the adapter is
    where the history term has to be attached.
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("rnb_r1_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_r1_train"] = module
    spec.loader.exec_module(module)

    model = _model("simvp_v2_qpe_guided")
    criterion = module._BalancedExtremeCriterion(alpha_csi=0.5, qpe_weight=0.5)
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    pred = model(x_radar, x_nwp)
    y_future = torch.rand(2, 10, 1, 32, 32) * 3.0
    y_hist = torch.rand(2, 11, 1, 32, 32) * 3.0

    value, components = criterion(
        pred, y_future, model=model, y_hist=y_hist, qpe_weight=0.5
    )
    assert "qpe_loss" in components
    assert components["total"] > components["qpe_loss"]


def test_the_adapted_objective_ignores_the_history_when_asked_to():
    """The ablation arm is the same model without the supervision."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("rnb_r1_train2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_r1_train2"] = module
    spec.loader.exec_module(module)

    model = _model("simvp_v2_qpe_guided")
    criterion = module._BalancedExtremeCriterion(alpha_csi=0.5, qpe_weight=0.0)
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    pred = model(x_radar, x_nwp)
    y_future = torch.rand(2, 10, 1, 32, 32) * 3.0
    y_hist = torch.rand(2, 11, 1, 32, 32) * 3.0

    _, components = criterion(
        pred, y_future, model=model, y_hist=y_hist, qpe_weight=0.0
    )
    assert "qpe_loss" not in components


@pytest.mark.parametrize("lag", [0, 2, 4, 11])
def test_r2_runs_at_every_lag(lag: int):
    """The whole point of R2 is the sweep over the lag.

    The observation goes in through ``obs_y_hist``, not as a third
    positional argument: the seam every baseline shares is
    ``model(x_radar, x_nwp)``, and a model that took a third positional
    argument could not be evaluated by the same code that evaluates the
    other nineteen.
    """
    model = _model("simvp_v2_assimilated", obs_lag_steps=lag)
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    y_hist = torch.rand(2, 11, 1, 32, 32) * 3.0

    model.obs_y_hist = y_hist
    out = model(x_radar, x_nwp)
    assert out.shape == (2, 10, 1, 32, 32)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert model.backbone.shared_rain_head.net[0].body[0].weight.grad is not None


def test_r2_refuses_a_third_positional_argument():
    """The seam is fixed, and a model that broke it would be unevaluable.

    The evaluation calls ``model(x_radar, x_nwp)`` for every baseline.
    A model that required a third positional argument would fail there
    before it read a single sample.
    """
    model = _model("simvp_v2_assimilated", obs_lag_steps=0)
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    with pytest.raises(TypeError):
        model(x_radar, x_nwp, torch.zeros(2, 11, 1, 32, 32))


def test_r2_takes_the_observation_through_the_model_not_the_signature():
    """The shared seam is ``model(x_radar, x_nwp)``.

    The observation is published on the model instead, so that the
    evaluation can hand it over without the baselines having to know
    about it.
    """
    model = _model("simvp_v2_assimilated", obs_lag_steps=0)
    assert hasattr(model, "obs_y_hist")

    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)
    y_hist = torch.rand(2, 11, 1, 32, 32) * 3.0

    model.obs_y_hist = y_hist
    with_obs = model(x_radar, x_nwp)

    # An arm that expects observations but is given none degrades to an
    # empty history channel rather than to the inverted estimate, which
    # would make it a copy of the inversion arm. The evaluation is the
    # layer that refuses to score such an arm.
    model.obs_y_hist = None
    without_obs = model(x_radar, x_nwp)

    # The observation does reach the forecast: an arm with a full
    # observation history and an arm with none at all differ.
    no_obs = _model("simvp_v2_assimilated", obs_lag_steps=11)
    no_obs.obs_y_hist = y_hist
    empty_history = no_obs(x_radar, x_nwp)

    assert not torch.allclose(with_obs, empty_history), (
        "the observation does not reach the forecast, so the lag sweep "
        "would measure nothing"
    )
    assert not torch.allclose(with_obs, without_obs)


def test_r2_declares_that_it_changes_the_task():
    """The scores of R2 are not comparable with the baselines'.

    That has to be visible in the code, because the paper's tables are
    generated from the configurations and a reader has to be able to
    tell which rows are in a different setting.
    """
    config = (REPO_ROOT / "configs" / "models" / "simvp_v2_assimilated.yaml")
    text = config.read_text(encoding="utf-8")
    assert "CHANGES THE TASK" in text
    assert "not comparable" in text


def test_the_history_rainfall_is_not_normalised():
    """It is a rainfall, in millimetres, like the target.

    The dataset normalises the target because the loss works in log
    space; the history rainfall is a physical field that the model
    consumes, so it has to stay in millimetres.
    """
    from radar_nwp_bench.data.dataset import RadarNWPDataset

    source = (REPO_ROOT / "radar_nwp_bench" / "data" / "dataset.py").read_text(
        encoding="utf-8"
    )
    # The history is read from the case and clipped, but never passed
    # through normalize_rain.
    assert "y_hist" in source
    segment = source.split("def _read_hist_label")[1]
    assert "normalize_rain" not in segment
