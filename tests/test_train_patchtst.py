import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = "/Users/deanwang/Code/autoresearch_Time"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TRAIN_PATH = Path(ROOT) / "train.py"


class TrainEntrypointTests(unittest.TestCase):
    def test_train_py_exposes_patchtst_config(self):
        module = importlib.import_module("train")
        self.assertTrue(hasattr(module, "PatchTSTConfig"))
        self.assertTrue(hasattr(module, "PatchTSTTrainer"))
        self.assertTrue(hasattr(module, "PatchTSTResearcher"))

    def test_train_py_is_self_contained_for_patchtst_model(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertNotIn("from data_provider.data_factory import data_provider", source)
        self.assertNotIn("from models import PatchTST", source)

    def test_smoke_training_runs_end_to_end(self):
        train = importlib.import_module("train")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.PatchTSTConfig(
                output_root=tmpdir,
                num_workers=0,
                save_predictions=False,
            )
            config = train.make_smoke_config(config)
            trainer = train.PatchTSTTrainer(config)
            summary = trainer.run()

        self.assertIn("train", summary)
        self.assertIn("test", summary)
        self.assertGreater(len(summary["train"]["history"]), 0)
        self.assertIn("mse", summary["test"])
        self.assertTrue(summary["test"]["mse"] >= 0.0)

    def test_researcher_can_compare_trials(self):
        train = importlib.import_module("train")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.make_smoke_config(
                train.PatchTSTConfig(
                    output_root=tmpdir,
                    num_workers=0,
                    save_predictions=False,
                )
            )
            researcher = train.PatchTSTResearcher(config)
            baseline = researcher.run_trial("baseline")
            candidate = researcher.run_trial(
                "candidate",
                overrides={"dropout": 0.1, "fc_dropout": 0.1},
            )
            decision = researcher.compare_trials(baseline, candidate)

        self.assertIn("mse", baseline.metrics)
        self.assertIn("mse", candidate.metrics)
        self.assertIn(decision["status"], {"keep", "discard"})
        self.assertEqual(decision["baseline"], "baseline")
        self.assertEqual(decision["candidate"], "candidate")


if __name__ == "__main__":
    unittest.main()
