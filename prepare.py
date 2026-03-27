"""System self-check entrypoint for the local PatchTST autoresearch project.

Usage:
    python prepare.py          # full system check, includes horizon-96 smoke run
    python prepare.py --quick  # fast check, skip smoke training
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import os
import platform
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
PATCHTST_ROOT = PROJECT_ROOT / "patchtst"
TRAIN_PATH = PROJECT_ROOT / "train.py"
ETTH1_PATH = PATCHTST_ROOT / "dataset" / "ETTh1.csv"
ETTH2_PATH = PATCHTST_ROOT / "dataset" / "ETTh2.csv"


def _pass(name: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "pass", "details": details}


def _fail(name: str, exc: BaseException) -> dict[str, Any]:
    return {
        "name": name,
        "status": "fail",
        "details": {
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(limit=6).strip(),
        },
    }


def check_python_environment() -> dict[str, Any]:
    import torch

    version = sys.version_info
    if version < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, got {platform.python_version()}")

    return _pass(
        "python_environment",
        python=platform.python_version(),
        executable=sys.executable,
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        mps_available=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        cwd=str(PROJECT_ROOT),
    )


def check_project_files() -> dict[str, Any]:
    missing = []
    for path in [PROJECT_ROOT, PATCHTST_ROOT, TRAIN_PATH]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing required project paths: {missing}")

    return _pass(
        "project_files",
        project_root=str(PROJECT_ROOT),
        train_py=str(TRAIN_PATH),
        patchtst_root=str(PATCHTST_ROOT),
    )


def _read_csv_preview(path: Path, rows: int = 2) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        preview = []
        for _ in range(rows):
            try:
                preview.append(next(reader))
            except StopIteration:
                break
    return header, preview


def check_etth1_dataset() -> dict[str, Any]:
    if not ETTH1_PATH.exists():
        raise FileNotFoundError(f"ETTh1 dataset not found: {ETTH1_PATH}")

    header, preview = _read_csv_preview(ETTH1_PATH)
    if not header:
        raise RuntimeError("ETTh1.csv is empty")
    if "date" not in header or "OT" not in header:
        raise RuntimeError(f"Unexpected ETTh1 columns: {header}")

    return _pass(
        "dataset_etth1",
        path=str(ETTH1_PATH),
        size_bytes=ETTH1_PATH.stat().st_size,
        columns=header,
        preview_rows=preview,
        etth2_present=ETTH2_PATH.exists(),
    )


def check_train_module_import() -> tuple[dict[str, Any], Any]:
    train = importlib.import_module("train")
    required = [
        "PRED_LEN",
        "TRAIN_TIME_BUDGET",
        "build_config",
        "make_smoke_overrides",
        "run_experiment",
    ]
    missing = [name for name in required if not hasattr(train, name)]
    if missing:
        raise RuntimeError(f"train.py missing required symbols: {missing}")

    return (
        _pass(
            "train_module_import",
            module_path=getattr(train, "__file__", str(TRAIN_PATH)),
            required_symbols=required,
        ),
        train,
    )


def run_patchtst_horizon96_smoke(train_module: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="patchtst-smoke-") as tmpdir:
        overrides = train_module.make_smoke_overrides(
            output_root=tmpdir,
            train_time_budget=0.01,
        )
        config = train_module.build_config(overrides)
        if config.pred_len != 96:
            raise RuntimeError(f"Smoke config pred_len must be 96, got {config.pred_len}")

        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            summary = train_module.run_experiment(overrides=overrides)

    return _pass(
        "patchtst_horizon96_smoke",
        pred_len=config.pred_len,
        device=summary["device"],
        output_root=tmpdir,
        num_steps=summary["train"]["num_steps"],
        stop_reason=summary["train"]["stop_reason"],
        training_seconds=summary["train"]["training_seconds"],
        val_metrics=summary["val"],
        test_metrics=summary["test"],
        log_tail=log_buffer.getvalue().strip().splitlines()[-12:],
    )


def run_system_check(run_smoke: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    train_module = None

    for check_fn in [check_python_environment, check_project_files, check_etth1_dataset]:
        try:
            checks.append(check_fn())
        except Exception as exc:  # noqa: BLE001
            checks.append(_fail(check_fn.__name__.replace("check_", ""), exc))

    try:
        train_result, train_module = check_train_module_import()
        checks.append(train_result)
    except Exception as exc:  # noqa: BLE001
        checks.append(_fail("train_module_import", exc))

    if run_smoke:
        if train_module is None:
            checks.append(
                {
                    "name": "patchtst_horizon96_smoke",
                    "status": "fail",
                    "details": {"error": "Skipped because train.py import failed."},
                }
            )
        else:
            try:
                checks.append(run_patchtst_horizon96_smoke(train_module))
            except Exception as exc:  # noqa: BLE001
                checks.append(_fail("patchtst_horizon96_smoke", exc))

    ok = all(item["status"] == "pass" for item in checks)
    return {
        "ok": ok,
        "project_root": str(PROJECT_ROOT),
        "checks": checks,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Project root: {report['project_root']}",
        "System check:",
    ]
    for item in report["checks"]:
        label = "PASS" if item["status"] == "pass" else "FAIL"
        lines.append(f"- [{label}] {item['name']}")
        details = item.get("details", {})
        if item["status"] == "pass":
            if item["name"] == "python_environment":
                lines.append(
                    f"  python={details['python']} torch={details['torch_version']} "
                    f"cuda={details['cuda_available']} mps={details['mps_available']}"
                )
            elif item["name"] == "dataset_etth1":
                lines.append(f"  dataset={details['path']} size={details['size_bytes']} bytes")
            elif item["name"] == "patchtst_horizon96_smoke":
                mse = details["test_metrics"]["mse"]
                lines.append(
                    f"  pred_len={details['pred_len']} device={details['device']} test_mse={mse:.6f}"
                )
        else:
            lines.append(f"  error={details.get('error', 'unknown error')}")
    lines.append(f"Overall: {'PASS' if report['ok'] else 'FAIL'}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatchTST project system self-check")
    parser.add_argument("--quick", action="store_true", help="Skip the horizon-96 smoke run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_system_check(run_smoke=not args.quick)
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
