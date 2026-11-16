"""The contracts that the deterministic-training and R2 fixes must hold.

Three separate promises are checked here, and each of them is the kind of
thing that silently stops being true:

1. The training loop is reproducible: the same configuration twice gives
   the same epoch-0 loss. This is what the deterministic switches buy, and
   the project measured a 1.78x spread on CSI@2.0 before it.
2. The R2 model really receives the observed history during training.
   When it did not, four arms of the lag sweep trained identically and the
   sweep measured nothing.
3. The verifier can hand a bootstrap the per-window tables it needs.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO))

from radar_nwp_bench.metrics.precipitation import PrecipitationVerifier


def _source(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


class TestDeterministicTraining(unittest.TestCase):
    """The training run must be reproducible."""

    def test_cublas_workspace_is_set_before_torch_is_imported(self):
        """cuBLAS reads its workspace setting when the handle is created.

        If the environment variable is set after torch is imported, the
        first matmul may already have opened the handle, and
        ``use_deterministic_algorithms`` then raises on the CUDA build.
        """
        src = _source("scripts/train.py")
        tree = ast.parse(src)
        setup_line = None
        import_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" and import_line is None:
                        import_line = node.lineno
            if isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "setdefault"
                    and isinstance(call.func.value, ast.Attribute)
                    and call.func.value.attr == "environ"
                ):
                    setup_line = node.lineno
        self.assertIsNotNone(
            setup_line, "CUBLAS_WORKSPACE_CONFIG is never set in train.py"
        )
        self.assertIsNotNone(import_line, "torch is never imported")
        self.assertLess(
            setup_line,
            import_line,
            "CUBLAS_WORKSPACE_CONFIG must be set before torch is imported",
        )

    def test_the_three_switches_are_present(self):
        src = _source("scripts/train.py")
        self.assertIn("torch.use_deterministic_algorithms(True, warn_only=True)", src)
        self.assertIn("torch.backends.cudnn.deterministic = True", src)
        self.assertIn("torch.backends.cudnn.benchmark = False", src)

    def test_the_worker_seeds_torch_and_not_only_numpy(self):
        """A worker that only seeds numpy leaves torch's own draws free."""
        src = _source("radar_nwp_bench/data/dataset.py")
        self.assertIn("torch.manual_seed", src)


class TestR2ObservationReachesTraining(unittest.TestCase):
    """The assimilated model must see the observation while it trains."""

    def test_training_publishes_the_observation_on_the_model(self):
        src = _source("scripts/train.py")
        self.assertIn(
            "model.obs_y_hist = y_hist",
            src,
            "the training loop never publishes the observed history",
        )

    def test_the_publication_is_guarded_so_the_baselines_are_untouched(self):
        """Only the R2 model has the attribute, so the guard isolates it."""
        src = _source("scripts/train.py")
        self.assertIn('hasattr(model, "obs_y_hist")', src)

    def test_the_baseline_models_do_not_carry_the_attribute(self):
        """If the base class had it, every arm would change."""
        base = _source("radar_nwp_bench/models/base.py")
        self.assertNotIn(
            "obs_y_hist",
            base,
            "the base class must not define obs_y_hist, or every model "
            "would receive an observation channel",
        )

    def test_the_evaluation_publishes_it_the_same_way(self):
        """Training and evaluation must agree on how it is delivered."""
        src = _source("scripts/evaluate.py")
        self.assertIn("model.obs_y_hist", src)


class TestPerWindowCounts(unittest.TestCase):
    """A bootstrap needs the per-window tables, not only the totals."""

    def _verifier(self) -> PrecipitationVerifier:
        return PrecipitationVerifier(
            thresholds=(0.5, 2.0), per_window=True
        )

    def test_per_window_is_off_by_default(self):
        """The default path must not grow the report."""
        v = PrecipitationVerifier(thresholds=(0.5,))
        self.assertFalse(v.per_window)
        self.assertEqual(v.window_counts, [])

    def test_the_tables_add_up_to_the_totals(self):
        """The sum of the windows must equal the accumulated counts.

        This is the invariant that makes the bootstrap sound: if the
        per-window tables do not reproduce the score that was reported,
        the interval is being computed around a different quantity.
        """
        rng = np.random.default_rng(0)
        v = self._verifier()
        pred = rng.uniform(0, 5, size=(7, 10, 16, 16)).astype(np.float32)
        truth = rng.uniform(0, 5, size=(7, 10, 16, 16)).astype(np.float32)
        v.update(pred, truth)

        self.assertEqual(len(v.window_counts), 7)
        for thr in (0.5, 2.0):
            total = v.counts[thr]
            table = v.per_window_table()
            self.assertAlmostEqual(
                sum(w[str(thr)]["hits"] for w in table), total.hits, places=3
            )
            self.assertAlmostEqual(
                sum(w[str(thr)]["misses"] for w in table), total.misses, places=3
            )
            self.assertAlmostEqual(
                sum(w[str(thr)]["false_alarms"] for w in table),
                total.false_alarms,
                places=3,
            )

    def test_the_table_survives_json(self):
        import json

        v = self._verifier()
        v.update(
            np.zeros((2, 3, 8, 8), dtype=np.float32),
            np.ones((2, 3, 8, 8), dtype=np.float32),
        )
        text = json.dumps(v.per_window_table())
        self.assertEqual(len(json.loads(text)), 2)

    def test_evaluate_exposes_the_switch(self):
        src = _source("scripts/evaluate.py")
        self.assertIn("--per-window-out", src)
        self.assertIn("per_window=", src)
        # The tables are only collected when they are going to be written.
        self.assertIn(
            "args.per_window_out is not None",
            src,
            "the per-window tables must not be collected unless they are "
            "asked for",
        )


if __name__ == "__main__":
    unittest.main()
