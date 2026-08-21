#!/usr/bin/env python3
"""Compare two pipeline artifact snapshots for logical-output invariance.

Runtime-only fields such as timestamps and execution profiles are ignored.
Content hashes, counts, split assignments, lineage, shard metadata, and
semantic contracts remain part of the comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


VOLATILE_KEYS = {
    "committed_at",
    "created_at_utc",
    "updated_at",
    "failed_at",
    "execution_profile",
    "runtime_profile",
    "timings",
    "timing_seconds",
    "elapsed_seconds",
}
MANIFEST_NAMES = {"manifest.json", "corpus_manifest.json"}


def invariant_projection(value: Any, key: str | None = None) -> Any:
    """Remove only operational metadata while retaining logical output data."""
    if key in VOLATILE_KEYS:
        return None
    if isinstance(value, dict):
        return {
            name: invariant_projection(value[name], name)
            for name in sorted(value)
            if name not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [invariant_projection(item) for item in value]
    return value


def manifest_paths(root: Path) -> List[Path]:
    return sorted(
        path for path in root.rglob("*.json")
        if path.is_file() and path.name in MANIFEST_NAMES
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_roots(left_root: Path, right_root: Path) -> Dict[str, Any]:
    left_paths = {path.relative_to(left_root).as_posix(): path for path in manifest_paths(left_root)}
    right_paths = {path.relative_to(right_root).as_posix(): path for path in manifest_paths(right_root)}
    issues: List[Dict[str, Any]] = []
    compared = 0
    for relative in sorted(set(left_paths) | set(right_paths)):
        left = left_paths.get(relative)
        right = right_paths.get(relative)
        if left is None or right is None:
            issues.append({
                "path": relative,
                "error": "manifest missing from one snapshot",
                "left": left is not None,
                "right": right is not None,
            })
            continue
        try:
            left_projection = invariant_projection(load_json(left))
            right_projection = invariant_projection(load_json(right))
        except Exception as exc:  # noqa: BLE001
            issues.append({"path": relative, "error": f"manifest read failed: {type(exc).__name__}: {exc}"})
            continue
        compared += 1
        if left_projection != right_projection:
            issues.append({"path": relative, "error": "logical manifest contents differ"})

    return {
        "audit": "laughlm_logical_output_invariance",
        "left_root": str(left_root),
        "right_root": str(right_root),
        "ignored_fields": sorted(VOLATILE_KEYS),
        "manifests_left": len(left_paths),
        "manifests_right": len(right_paths),
        "manifests_compared": compared,
        "issues": issues,
        "status": "pass" if compared > 0 and not issues else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", required=True, help="First release/artifact snapshot")
    parser.add_argument("--right-root", required=True, help="Second release/artifact snapshot")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = audit_roots(Path(args.left_root).resolve(), Path(args.right_root).resolve())
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[invariance-audit] report written: {output}")
    print(f"[invariance-audit] {report['status'].upper()}: {report['manifests_compared']} manifests compared")
    if report["issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
