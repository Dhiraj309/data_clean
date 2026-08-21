#!/usr/bin/env python3
"""Freeze the benchmark and sealed-evaluation contracts without mutating drafts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from benchmark_contract import (
    benchmark_manifest,
    require_frozen_benchmark,
    require_frozen_sealed_evaluation,
)
from config_loader import load_yaml, stable_hash


def write_yaml(path: Path, value: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument("--sealed-config", required=True)
    parser.add_argument("--training-repo-id", required=True)
    parser.add_argument("--benchmark-output", required=True)
    parser.add_argument("--sealed-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    args = parser.parse_args()

    benchmark = load_yaml(args.benchmark_config)
    sealed = load_yaml(args.sealed_config)
    benchmark["freeze_status"] = "frozen"
    sealed["freeze_status"] = "frozen"
    sealed["training_repo_id"] = args.training_repo_id

    benchmark_output = Path(args.benchmark_output).expanduser().resolve()
    sealed_output = Path(args.sealed_output).expanduser().resolve()
    manifest_output = Path(args.manifest_output).expanduser().resolve()
    if benchmark_output == sealed_output:
        raise ValueError("benchmark-output and sealed-output must be different files")
    benchmark["sealed_evaluation"] = dict(benchmark.get("sealed_evaluation") or {})
    benchmark["sealed_evaluation"]["config"] = os.path.relpath(
        sealed_output, benchmark_output.parent
    ).replace("\\", "/")

    # Validate the exact emitted values, not the draft inputs.
    benchmark_manifest_value = require_frozen_benchmark(benchmark)
    sealed_manifest_value = require_frozen_sealed_evaluation(sealed, args.training_repo_id)
    write_yaml(benchmark_output, benchmark)
    write_yaml(sealed_output, sealed)

    freeze_manifest = {
        "format": "laughlm_frozen_evaluation_contract_v1",
        "status": "frozen",
        "benchmark": benchmark_manifest_value,
        "sealed_evaluation": sealed_manifest_value,
        "benchmark_config": benchmark_output.name,
        "sealed_config": sealed_output.name,
        "training_repo_id": args.training_repo_id,
        "benchmark_config_hash": stable_hash(benchmark),
        "sealed_config_hash": stable_hash(sealed),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[benchmark-freeze] benchmark: {benchmark_output}")
    print(f"[benchmark-freeze] sealed evaluation: {sealed_output}")
    print(f"[benchmark-freeze] manifest: {manifest_output}")
    print("[benchmark-freeze] update Stage-3 configs to reference the frozen benchmark file before processing")


if __name__ == "__main__":
    main()
