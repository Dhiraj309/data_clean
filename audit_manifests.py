#!/usr/bin/env python3
"""Audit Stage 1-3 remote manifests and their referenced output files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import HfApi

from config_loader import load_common
from manifest_contract import validate_committed_manifest
from pipeline_utils import (
    download_file,
    file_detail,
    hf_token,
    list_repo_files,
    write_json,
)


def audit_repository(
    *,
    repo_id: str,
    stage: str,
    revision: str,
    prefix: str,
    common: Dict[str, Any],
    token: str,
    verify_checksums: bool,
) -> Dict[str, Any]:
    api = HfApi(token=token)
    files = set(list_repo_files(api, repo_id, token, common))
    manifest_paths = sorted(
        path for path in files
        if path.endswith("/manifest.json") and (not prefix or path.startswith(prefix))
    )
    issues: List[Dict[str, Any]] = []
    valid = 0
    for manifest_path in manifest_paths:
        entry: Dict[str, Any] = {"manifest": manifest_path, "errors": []}
        try:
            local_manifest = download_file(repo_id, manifest_path, revision, token, common)
            manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            entry["errors"].append(f"manifest read failed: {type(exc).__name__}: {exc}")
            issues.append(entry)
            continue

        errors = validate_committed_manifest(
            manifest, expected_stage=stage, available_files=files
        )
        entry["source_key"] = manifest.get("source_key")
        entry["status"] = manifest.get("processing_status")
        entry["errors"].extend(errors)

        if verify_checksums and not errors:
            for detail in manifest["output_part_details"]:
                remote_path = detail["path"]
                try:
                    local_output = download_file(repo_id, remote_path, revision, token, common)
                    actual = file_detail(local_output, detail["algorithm"])
                    for field in ("bytes", "digest"):
                        if actual[field] != detail[field]:
                            entry["errors"].append(
                                f"checksum mismatch for {remote_path}: "
                                f"{field} expected={detail[field]!r} actual={actual[field]!r}"
                            )
                except Exception as exc:  # noqa: BLE001
                    entry["errors"].append(
                        f"output verification failed for {remote_path}: {type(exc).__name__}: {exc}"
                    )

        if entry["errors"]:
            issues.append(entry)
        else:
            valid += 1

    if not manifest_paths:
        issues.append({"manifest": None, "errors": ["no matching manifests found"]})

    return {
        "audit": "laughlm_stage_manifest",
        "repo_id": repo_id,
        "revision": revision,
        "stage": stage,
        "prefix": prefix,
        "verify_checksums": verify_checksums,
        "manifest_count": len(manifest_paths),
        "valid_manifest_count": valid,
        "invalid_manifest_count": len(issues),
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repository")
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2", "stage3"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="", help="Optional remote path prefix to audit")
    parser.add_argument("--common", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument("--verify-checksums", action="store_true", help="Download and verify every output part")
    args = parser.parse_args()

    common = load_common(args.common)
    token = hf_token(common)
    report = audit_repository(
        repo_id=args.repo_id,
        stage=args.stage,
        revision=args.revision,
        prefix=args.prefix,
        common=common,
        token=token,
        verify_checksums=args.verify_checksums,
    )
    if args.output:
        write_json(Path(args.output), report)
        print(f"[manifest-audit] report written: {args.output}")
    print(f"[manifest-audit] {report['status'].upper()}: {report['valid_manifest_count']}/{report['manifest_count']} valid")
    if report["issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
