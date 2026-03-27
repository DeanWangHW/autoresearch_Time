import importlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = "/Users/deanwang/Code/autoresearch_Time"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TRAIN_PATH = Path(ROOT) / "train.py"


class TrainEntrypointTests(unittest.TestCase):
    def test_train_py_exposes_top_level_controls(self):
        module = importlib.import_module("train")
        for name in [
            "PRED_LEN",
            "TRAIN_TIME_BUDGET",
            "LEARNING_RATE",
            "BATCH_SIZE",
            "DEVICE",
            "build_config",
            "make_smoke_overrides",
            "run_experiment",
        ]:
            self.assertTrue(hasattr(module, name), name)
        self.assertEqual(module.PRED_LEN, 96)

    def test_train_py_removes_trainer_and_researcher_wrappers(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertNotIn("class PatchTSTTrainer", source)
        self.assertNotIn("class PatchTSTResearcher", source)
        self.assertNotIn("from data_provider.data_factory import data_provider", source)
        self.assertNotIn("from models import PatchTST", source)

    def test_smoke_training_runs_end_to_end_with_time_budget(self):
        train = importlib.import_module("train")
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = train.run_experiment(
                overrides=train.make_smoke_overrides(
                    output_root=tmpdir,
                    train_time_budget=0.01,
                )
            )

        self.assertIn("train", summary)
        self.assertIn("test", summary)
        self.assertGreaterEqual(summary["train"]["num_steps"], 1)
        self.assertEqual(summary["train"]["stop_reason"], "time_budget")
        self.assertEqual(summary["config"]["pred_len"], 96)
        self.assertIn("mse", summary["test"])
        self.assertTrue(summary["test"]["mse"] >= 0.0)

    def test_training_does_not_stop_before_time_budget_is_hit(self):
        import torch
        from unittest import mock

        train = importlib.import_module("train")
        overrides = train.make_smoke_overrides(train_time_budget=10.0)
        config = train.build_config(overrides)

        batch = (
            torch.zeros(config.batch_size, config.seq_len, config.enc_in),
            torch.zeros(config.batch_size, config.label_len + config.pred_len, config.c_out),
            torch.zeros(config.batch_size, config.seq_len, 1),
            torch.zeros(config.batch_size, config.label_len + config.pred_len, 1),
        )

        timeline = iter([
            0.0,
            0.1,
            1.0,
            1.1,
            10.1,
            10.2,
            10.3,
            10.4,
            10.5,
        ])

        def fake_data_provider(*_args, **_kwargs):
            return None, [batch]

        with tempfile.TemporaryDirectory() as tmpdir:
            overrides["output_root"] = tmpdir
            with mock.patch.object(train, "data_provider", side_effect=fake_data_provider), mock.patch.object(
                train.time, "perf_counter", side_effect=lambda: next(timeline)
            ):
                summary = train.run_experiment(overrides=overrides)

        self.assertEqual(summary["train"]["num_epochs"], 2)
        self.assertEqual(summary["train"]["stop_reason"], "time_budget")


if __name__ == "__main__":
    unittest.main()
