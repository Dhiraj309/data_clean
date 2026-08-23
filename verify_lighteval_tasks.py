#!/usr/bin/env python3
"""Verify a proposed LaughLM Stage-3 task list against installed Lighteval."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_contract import benchmark_task_errors
from config_loader import load_yaml


def task_identifier(task: Any) -> str:
    if isinstance(task, str):
        return task.strip()
    if isinstance(task, dict):
        return str(task.get("id") or task.get("name") or "").strip()
    return ""


def command_result(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lighteval-bin", default="lighteval")
    args = parser.parse_args()

    benchmark = load_yaml(args.benchmark_config)
    tasks = list(benchmark.get("lighteval_tasks") or [])
    errors = benchmark_task_errors(tasks)
    version = command_result([args.lighteval_bin, "--version"])
    checks: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        identifier = task_identifier(task)
        result = command_result([args.lighteval_bin, "tasks", "inspect", identifier])
        checks.append(
            {
                "index": index,
                "task": task,
                "identifier": identifier,
                "passed": result["returncode"] == 0,
                **result,
            }
        )

    report = {
        "format": "laughlm_lighteval_task_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_config": str(Path(args.benchmark_config)),
        "lighteval_bin": args.lighteval_bin,
        "version": version,
        "contract_errors": errors,
        "checks": checks,
        "status": "pass" if not errors and checks and all(item["passed"] for item in checks) else "fail",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[lighteval-task-audit] report written: {output}")
    if report["status"] != "pass":
        raise SystemExit("[lighteval-task-audit] FAIL")
    print("[lighteval-task-audit] PASS")


if __name__ == "__main__":
    main()
