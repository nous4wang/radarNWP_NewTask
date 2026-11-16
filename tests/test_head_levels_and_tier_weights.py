"""The two invariants the hierarchical head has to hold.

Both were found by measurement on the archive rather than by reasoning,
and both are silent failures: the model trains, the loss falls, and the
score is simply wrong.

The first is the level at which a bin is represented. When the head
reconstructs the rain rate as the expectation over the bins, and each bin
is represented by its lower edge, then the bin that starts at 2.0 mm is
represented by 2.0 mm -- the verification threshold itself. A pixel the
head is confident about therefore reconstructs to exactly the threshold,
and whether it counts as a hit is decided by the rounding. Measured: the
field-wide tier loss produced a maximum prediction of 1.999797 mm, with
1.51 per cent of the pixels within a hundredth of a millimetre below
2.0 mm and none above it, so the score at 2.0 mm was exactly zero however
much probability mass the head put on the heavy bins.

Representing a bin by its midpoint fixes it: a pixel above a threshold
reconstructs to a rate above it.

The second is the tier weights. They are a property of the threshold they
belong to, so a run that scores a different set of thresholds must not
inherit the weights of a different tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss
from radar_nwp_bench.models.common import QuantumHierarchicalRainHead


# ---------------------------------------------------------------------------
# the level at which a bin is represented
# ---------------------------------------------------------------------------
def _levels(head: QuantumHierarchicalRainHead) -> torch.Tensor:
    """The representative value of each bin."""
    return head._bin_levels(torch.device("cpu"), torch.float32).flatten()


def test_a_bin_above_a_threshold_is_represented_above_it():
    """The head must be able to reconstruct a rate that crosses the line.

    With the lower edge as the representative, the bin that begins at
    2.0 mm is represented by 2.0 mm, so a prediction that the head is
    certain about lands exactly on the threshold and the score at that
    threshold is decided by the rounding.
    """
    thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)
    head = QuantumHierarchicalRainHead(
        in_channels=4, hidden_dim=32, num_bins=6, thresholds=thresholds,
        level_mode="midpoint",
    )
    levels = _levels(head)

    for index, threshold in enumerate(thresholds):
        # The bin that starts at the threshold is the one after the dry
        # bin and the previous thresholds.
        bin_index = index + 1
        assert levels[bin_index] > threshold, (
            f"the bin beginning at {threshold} mm is represented by "
            f"{levels[bin_index]:.3f} mm, which does not cross the "
            f"threshold it is supposed to be above"
        )


def test_the_floor_mode_is_the_one_that_fails():
    """The old construction is kept, and it is known to fail."""
    thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)
    head = QuantumHierarchicalRainHead(
        in_channels=4, hidden_dim=32, num_bins=6, thresholds=thresholds,
        level_mode="floor",
    )
    levels = _levels(head)
    # The bin beginning at 2.0 mm is represented by exactly 2.0 mm.
    assert levels[4] == pytest.approx(2.0)


def test_the_dry_bin_stays_at_zero():
    """The label's smallest positive value is a tenth of a millimetre.

    So the dry bin holds only zeros, and its representative must be zero
    as well: a dry pixel that reconstructs to a positive rate is a false
    alarm at every threshold.
    """
    head = QuantumHierarchicalRainHead(
        in_channels=4, hidden_dim=32, num_bins=6, level_mode="midpoint",
    )
    assert float(_levels(head)[0]) == 0.0


def test_the_reconstruction_agrees_with_the_probability():
    """The end-to-end statement the fix is about.

    When all the probability mass sits on the bin above the 2.0 mm
    threshold, the reconstructed rate has to be above 2.0 mm. With the
    lower edge as the representative it lands exactly on the threshold,
    which is what made the score at 2.0 mm zero.
    """
    thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)
    levels = {mode: _levels(QuantumHierarchicalRainHead(
        in_channels=4, hidden_dim=32, num_bins=6, thresholds=thresholds,
        level_mode=mode,
    )) for mode in ("midpoint", "floor")}

    # All the mass on the bin that begins at 2.0 mm.
    probs = torch.zeros(1, 6, 1, 1)
    probs[0, 4, 0, 0] = 1.0

    midpoint_rate = float((probs * levels["midpoint"].view(1, 6, 1, 1)).sum())
    floor_rate = float((probs * levels["floor"].view(1, 6, 1, 1)).sum())

    assert midpoint_rate > 2.0, (
        f"the midpoint construction reconstructs to {midpoint_rate} mm, "
        f"which does not cross the 2.0 mm threshold the bin begins at"
    )
    assert floor_rate == pytest.approx(2.0), (
        "the floor construction is expected to land exactly on the "
        "threshold, which is the failure being fixed"
    )


def test_a_leak_below_the_bin_does_not_push_the_rate_under():
    """The reason the midpoint matters: leakage is what broke it.

    The expectation of a distribution whose mass sits on a bin edge and
    leaks slightly below it is below the edge, so a pixel the head is
    almost certain about reconstructs to just under the threshold. That
    is what the measurement saw: a maximum prediction of 1.999797 mm.
    """
    thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)
    levels = {mode: _levels(QuantumHierarchicalRainHead(
        in_channels=4, hidden_dim=32, num_bins=6, thresholds=thresholds,
        level_mode=mode,
    )) for mode in ("midpoint", "floor")}

    # 99.98 per cent on the bin above 2.0 mm, the rest leaking below.
    probs = torch.zeros(1, 6, 1, 1)
    probs[0, 4, 0, 0] = 0.9998
    probs[0, 3, 0, 0] = 0.0002

    for mode, levels_t in levels.items():
        rate = float((probs * levels_t.view(1, 6, 1, 1)).sum())
        if mode == "floor":
            assert rate < 2.0, (
                "the floor construction is expected to fall under the "
                "threshold, which is the failure being fixed"
            )
        else:
            assert rate > 2.0, (
                f"the midpoint construction reconstructs to {rate} mm, "
                f"which a little leakage pushed under the threshold"
            )




def test_the_level_mode_is_recorded_and_checked():
    """An unknown mode must fail rather than be ignored."""
    with pytest.raises(ValueError):
        QuantumHierarchicalRainHead(
            in_channels=4, hidden_dim=32, num_bins=6, level_mode="mean",
        )


# ---------------------------------------------------------------------------
# the tier weights
# ---------------------------------------------------------------------------
def test_the_tier_weights_follow_the_threshold():
    """The weights belong to the threshold, not to a position.

    A run that scores the business set of thresholds -- (0.1, 0.5, 2.0,
    4.0) -- must give the 2.0 mm threshold the weights meant for 2.0 mm.
    With positional indexing it inherited the weights of 1.0 mm, and the
    heaviest threshold's weights were never used at all.
    """
    weights = HierarchicalPrecipLoss.weights_for((0.1, 0.5, 2.0, 4.0))
    assert weights[0] == (1.0, 3.0)   # 0.1 mm
    assert weights[1] == (1.5, 2.0)   # 0.5 mm
    assert weights[2] == (4.0, 0.5)   # 2.0 mm -- not the 1.0 mm weights
    assert weights[3] == (6.0, 0.2)   # 4.0 mm -- the heaviest ones


def test_the_canonical_set_keeps_its_weights():
    weights = HierarchicalPrecipLoss.weights_for((0.1, 0.5, 1.0, 2.0, 4.0))
    assert weights == ((1.0, 3.0), (1.5, 2.0), (2.5, 1.0), (4.0, 0.5),
                       (6.0, 0.2))


def test_an_unknown_threshold_gets_a_neutral_weight():
    weights = HierarchicalPrecipLoss.weights_for((0.1, 7.5))
    assert weights[1] == (2.5, 1.0)


def test_the_loss_picks_the_weights_up_from_its_thresholds():
    loss = HierarchicalPrecipLoss(thresholds=(0.1, 0.5, 2.0, 4.0))
    assert loss.tier_weights[2] == (4.0, 0.5)
    assert loss.tier_weights[3] == (6.0, 0.2)


# ---------------------------------------------------------------------------
# which channel answers which threshold
# ---------------------------------------------------------------------------
def test_the_channel_is_found_by_value_and_not_by_position():
    """The head's channels answer its own thresholds.

    The head carries the thresholds (0.1, 0.5, 1.0, 2.0, 4.0), so its
    channel 2 answers P(R >= 1.0) and its channel 3 answers P(R >= 2.0).
    A loss that asks about (0.1, 0.5, 2.0, 4.0) must therefore read
    channels 0, 1, 3 and 4. Taking them in order supervises P(R >= 1.0)
    against the truth's "at or above 2 mm" indicator, which is what made
    the model drive P(R >= 2 mm) to almost zero.
    """
    head_thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)
    assert HierarchicalPrecipLoss._channel_for(head_thresholds, 0.1) == 0
    assert HierarchicalPrecipLoss._channel_for(head_thresholds, 0.5) == 1
    assert HierarchicalPrecipLoss._channel_for(head_thresholds, 1.0) == 2
    assert HierarchicalPrecipLoss._channel_for(head_thresholds, 2.0) == 3
    assert HierarchicalPrecipLoss._channel_for(head_thresholds, 4.0) == 4


def test_a_threshold_the_model_does_not_carry_is_skipped():
    head_thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)
    assert HierarchicalPrecipLoss._channel_for(head_thresholds, 8.0) is None


def test_the_loss_reads_the_thresholds_off_the_model():
    """The thresholds come from the head, not from a guess about it."""
    import torch.nn as nn

    class _Head(nn.Module):
        thresholds = (0.1, 0.5, 1.0, 2.0, 4.0)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.quantum_povm_head = _Head()

    assert HierarchicalPrecipLoss._find_head_thresholds(_Model()) == (
        0.1, 0.5, 1.0, 2.0, 4.0
    )
    assert HierarchicalPrecipLoss._find_head_thresholds(None) == ()
    assert HierarchicalPrecipLoss._find_head_thresholds(nn.Linear(1, 1)) == ()


# ---------------------------------------------------------------------------
# the head settings apply only to a model that has the head
# ---------------------------------------------------------------------------
def test_the_registry_exposes_the_class_without_building_it():
    from radar_nwp_bench.models import model_class

    assert model_class("simvp_v2") is not None
    assert model_class("simvp_v2_hqma") is not None
    assert model_class("no_such_model") is None


def test_only_the_hqma_model_accepts_the_head_settings():
    """A baseline without the head must not be given its settings.

    The evaluation reads the head's settings from the training record so
    that a checkpoint is loaded into the architecture it was trained
    with. A baseline that has no head at all must not be given them: the
    constructor rejects the keyword and the evaluation fails before it
    reads a single sample.
    """
    import inspect

    from radar_nwp_bench.models import model_class

    head = inspect.signature(model_class("simvp_v2_hqma").__init__).parameters
    plain = inspect.signature(model_class("simvp_v2").__init__).parameters

    assert "head_bins" in head and "level_mode" in head
    assert "head_bins" not in plain
    assert "level_mode" not in plain


def test_the_evaluation_only_applies_the_settings_that_fit():
    """The settings are filtered by what the constructor accepts."""
    import inspect

    from radar_nwp_bench.models import model_class

    head_settings = {"head_bins": 6, "level_mode": "midpoint"}

    for name in ("simvp_v2", "simvp_v2_hqma"):
        accepted = inspect.signature(model_class(name).__init__).parameters
        applicable = {k: v for k, v in head_settings.items() if k in accepted}
        if name == "simvp_v2_hqma":
            assert applicable == head_settings
        else:
            assert applicable == {}
