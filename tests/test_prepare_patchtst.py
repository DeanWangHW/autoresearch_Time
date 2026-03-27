import importlib
import sys
import unittest

ROOT = "/Users/deanwang/Code/autoresearch_Time"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class PrepareEntrypointTests(unittest.TestCase):
    def test_prepare_exposes_system_check(self):
        module = importlib.import_module("prepare")
        self.assertTrue(hasattr(module, "run_system_check"))
        self.assertTrue(hasattr(module, "PROJECT_ROOT"))

    def test_quick_check_reports_core_dependencies(self):
        module = importlib.import_module("prepare")
        report = module.run_system_check(run_smoke=False)

        self.assertTrue(report["ok"])
        check_names = {item["name"] for item in report["checks"]}
        self.assertIn("python_environment", check_names)
        self.assertIn("dataset_etth1", check_names)
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
