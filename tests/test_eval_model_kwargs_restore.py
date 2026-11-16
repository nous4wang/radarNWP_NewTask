"""The evaluation side must rebuild the model the training run built.

A constructor argument that is not a weight -- an output clamp, a
threshold -- does not appear in the ``state_dict``.  A checkpoint
trained with one therefore loads cleanly into a model built without it,
and the evaluation then scores a model the training run never used.
This is the same failure mode the project has already paid for twice:
a runtime switch that does not change the weights, so the evaluation
silently measures the mechanism that was turned off.

These tests pin down the three facts that make the restore correct:

1.  the training run's ``config.json`` records the ``--model-kwargs`` it
    was given, verbatim;
2.  the evaluation reads that record back, and rebuilds the model with
    it;
3.  the clamp changes what the model outputs, and changes nothing about
    the weights -- which is precisely why the restore is needed and
    precisely why the checkpoint alone cannot carry it.

They also pin the negative case: a checkpoint that recorded no
constructor kwargs must be evaluated exactly as before.  Every one of the
21 E45 runs recorded none, so this is the path that all of them take.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent

from radar_nwp_bench.models import build_model


def _load_evaluate():
    """Load scripts/evaluate.py as a module (it has no import side effects)."""
    path = REPO / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("e45_eval_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_evaluate()


class TestTrainedModelKwargsRestore(unittest.TestCase):
    """The recorded constructor kwargs come back, and only when there are any."""

    def test_a_recorded_json_string_is_read_back(self):
        """The training run records the flag as the string it was given."""
        trained = {"model_kwargs": '{"clamp_mm": 20.0}'}
        self.assertEqual(EVAL._trained_model_kwargs(trained),
                         {"clamp_mm": 20.0})

    def test_a_recorded_mapping_is_accepted_too(self):
        """A config.json that stores the mapping itself is read back as is."""
        trained = {"model_kwargs": {"clamp_mm": 20.0}}
        self.assertEqual(EVAL._trained_model_kwargs(trained),
                         {"clamp_mm": 20.0})

    def test_nothing_recorded_means_nothing_restored(self):
        """The path every one of the 21 E45 runs takes."""
        for trained in (None, {}, {"model_kwargs": None},
                        {"model_kwargs": ""}, {"model_kwargs": {}}):
            with self.subTest(trained=trained):
                self.assertEqual(EVAL._trained_model_kwargs(trained), {})

    def test_unreadable_record_is_refused_rather_than_guessed(self):
        """A broken record must not become a silent half-restore."""
        trained = {"model_kwargs": "{not json"}
        self.assertEqual(EVAL._trained_model_kwargs(trained), {})

    def test_a_non_mapping_record_is_refused(self):
        """JSON that is not an object is not a set of constructor kwargs."""
        self.assertEqual(EVAL._trained_model_kwargs({"model_kwargs": "[1, 2]"}),
                         {})


class TestClampIsNotInTheStateDict(unittest.TestCase):
    """Why the restore is necessary at all."""

    def _model(self, clamp_mm: float) -> torch.nn.Module:
        torch.manual_seed(0)
        return build_model(
            "topo_simvp",
            in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10,
            clamp_mm=clamp_mm,
        )

    def test_the_clamp_is_not_a_weight(self):
        """The two models have byte-identical weights."""
        a = self._model(0.0)
        b = self._model(20.0)
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
        self.assertFalse(any("clamp" in k for k in a.state_dict()))

    def test_the_clamp_bounds_the_output_field(self):
        """With the clamp the field cannot exceed it; without, it can."""
        # The model's forward takes the radar history as
        # (B, T_in, in_radar_c, H, W) and the NWP as (B, T_nwp, in_nwp_c, H, W)
        # with T_nwp = 2, which the encoder flattens to in_nwp_c * 2.
        radar = torch.randn(1, 11, 10, 32, 32) * 60.0
        nwp = torch.randn(1, 2, 27, 32, 32) * 60.0

        # A freshly initialised network does not predict 20 mm anywhere, so
        # the bias of the last decoder convolution is raised until the
        # unclamped field does.  The value is chosen so that the
        # millimetre field lands near expm1(5) = 147 mm: above the clamp
        # and below the expm1(6) = 402.43 mm ceiling.
        torch.manual_seed(0)
        model = self._model(20.0).eval()
        with torch.no_grad():
            model.dec[-1].bias.fill_(5.0)

        with torch.no_grad():
            clamped = model(radar, nwp)
        self.assertLessEqual(float(clamped.max()), 20.0 + 1e-6)

        # The same weights, the same input, the clamp turned off.
        torch.manual_seed(0)
        unclamped = self._model(0.0).eval()
        with torch.no_grad():
            unclamped.dec[-1].bias.fill_(5.0)
            raw = unclamped(radar, nwp)
        self.assertGreater(float(raw.max()), 20.0)


class TestEvaluationScriptWiring(unittest.TestCase):
    """The restore is actually in the evaluation's model-construction path."""

    def _source(self) -> str:
        return (REPO / "scripts" / "evaluate.py").read_text(encoding="utf-8")

    def test_the_evaluation_reads_the_recorded_kwargs(self):
        """The function is called, and its result reaches the model config."""
        source = self._source()
        self.assertIn("_trained_model_kwargs(trained)", source)
        self.assertIn("restored_kwargs", source)

    def test_the_training_run_records_the_flag(self):
        """train.py writes its argparse namespace, so the flag is recorded."""
        source = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
        self.assertIn('"--model-kwargs"', source)
        self.assertIn("run_config = dict(vars(args))", source)

    def test_the_recorded_kwargs_reach_the_constructor(self):
        """The whole path, end to end, on the model_kwargs() helper."""
        from radar_nwp_bench.config import model_kwargs as mk

        kwargs = mk({"model": {}}, overrides='{"clamp_mm": 20.0}')
        self.assertEqual(kwargs["clamp_mm"], 20.0)
        model = build_model("topo_simvp", **kwargs)
        self.assertEqual(model.clamp_mm, 20.0)

    def test_the_shape_keys_are_not_overridable_by_the_record(self):
        """The data's shapes win over anything a record claims."""
        from radar_nwp_bench.config import model_kwargs as mk

        kwargs = mk(
            {"model": {"in_len": 999}},
            shapes={"in_len": 11},
            overrides='{"clamp_mm": 20.0}',
        )
        self.assertEqual(kwargs["in_len"], 11)


class TestArmTableStillCarriesNoModelKwargs(unittest.TestCase):
    """The E45 arm table is frozen; it is recorded, not changed.

    The wiring this round adds is the evaluation side.  The training
    side of the amplitude clamp belongs to the NEXT round's arm table,
    because the E45 arm table is the pre-registered definition of what
    E45 ran, and changing it would change what E45 means.
    """

    def test_the_e45_arm_table_is_untouched(self):
        source = (REPO / "scripts" / "_e45_arms.sh").read_text(encoding="utf-8")
        self.assertIn('"A3a|topo_simvp|topo|--topo-clamp-mm 20|topo_clamp_mm=20.0"',
                      source)
        self.assertNotIn("--model-kwargs", source)


if __name__ == "__main__":
    unittest.main()
