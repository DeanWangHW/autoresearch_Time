import importlib
import sys
import unittest
from pathlib import Path

ROOT = "/Users/deanwang/Code/autoresearch_Time"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class PrepareEntrypointTests(unittest.TestCase):
    def test_prepare_exposes_system_check(self):
        module = importlib.import_module("prepare")
        self.assertTrue(hasattr(module, "run_system_check"))
        self.assertTrue(hasattr(module, "PROJECT_ROOT"))
        self.assertTrue(hasattr(module, "ensure_etth1_split"))

    def test_prepare_creates_physical_train_and_blind_test_files(self):
        module = importlib.import_module("prepare")
        split = module.ensure_etth1_split()

        train_path = Path(split["train_path"])
        blind_test_path = Path(split["blind_test_path"])

        self.assertTrue(train_path.exists())
        self.assertTrue(blind_test_path.exists())
        self.assertGreater(split["train_rows"], 0)
        self.assertGreater(split["blind_test_rows"], 0)
        self.assertLess(split["train_last_date"], split["blind_test_first_date"])

    def test_quick_check_reports_core_dependencies(self):
        module = importlib.import_module("prepare")
        report = module.run_system_check(run_smoke=False)

        self.assertTrue(report["ok"])
        check_names = {item["name"] for item in report["checks"]}
        self.assertIn("python_environment", check_names)
        self.assertIn("dataset_etth1", check_names)
        self.assertIn("dataset_etth1_split", check_names)
        self.assertIn("train_module_import", check_names)
        self.assertNotIn("patchtst_horizon96_smoke", check_names)

    def test_default_check_runs_patchtst_horizon96_smoke(self):
        module = importlib.import_module("prepare")
        report = module.run_system_check()

        self.assertTrue(report["ok"])
        smoke = next(item for item in report["checks"] if item["name"] == "patchtst_horizon96_smoke")
        self.assertEqual(smoke["status"], "pass")
        self.assertEqual(smoke["details"]["pred_len"], 96)
        self.assertEqual(smoke["details"]["stop_reason"], "time_budget")
        self.assertIn("mse", smoke["details"]["test_metrics"])


if __name__ == "__main__":
    unittest.main()
