#!/usr/bin/env python3
"""Audit Stage-3 split assignments and duplicate-family boundaries."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from config_loader import load_common
from manifest_contract import validate_committed_manifest
from pipeline_utils import download_file, hf_token, list_repo_files, write_json


def audit_splits(*, repo_id: str, revision: str, prefix: str, common: Dict[str, Any], token: str) -> Dict[str, Any]:
    api = HfApi(token=token)
    files = set(list_repo_files(api, repo_id, token, common))
    manifests = sorted(path for path in files if path.endswith("/manifest.json") and (not prefix or path.startswith(prefix)))
    groups: Dict[str, str] = {}
    split_counts: Dict[str, int] = defaultdict(int)
    issues: List[Dict[str, Any]] = []
    for manifest_path in manifests:
        local_manifest = download_file(repo_id, manifest_path, revision, token, common)
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        errors = validate_committed_manifest(manifest, expected_stage="stage3", available_files=files)
        if errors:
            issues.append({"manifest": manifest_path, "errors": errors})
            continue
        for remote_part in manifest["output_parts"]:
            local_part = download_file(repo_id, remote_part, revision, token, common)
            with pq.ParquetFile(local_part) as parquet:
                names = set(parquet.schema_arrow.names)
                if "split" not in names or "split_group" not in names:
                    issues.append({"manifest": manifest_path, "part": remote_part, "errors": ["missing split or split_group column"]})
                    continue
                for batch in parquet.iter_batches(batch_size=8192, columns=["split", "split_group"]):
                    splits = batch.column("split").to_pylist()
                    groups_in_batch = batch.column("split_group").to_pylist()
                    for split, group in zip(splits, groups_in_batch):
                        if not group:
                            issues.append({"manifest": manifest_path, "part": remote_part, "errors": ["row has empty split_group"]})
                            continue
                        split, group = str(split), str(group)
                        split_counts[split] += 1
                        previous = groups.get(group)
                        if previous is not None and previous != split:
                            issues.append({"manifest": manifest_path, "part": remote_part, "group": group, "errors": [f"duplicate family crosses splits: {previous} -> {split}"]})
                        groups[group] = split
    if not manifests:
        issues.append({"manifest": None, "errors": ["no matching Stage-3 manifests found"]})
    return {
        "audit": "laughlm_split_assignments",
        "repo_id": repo_id,
        "revision": revision,
        "prefix": prefix,
        "manifest_count": len(manifests),
        "unique_split_groups": len(groups),
        "split_counts": dict(sorted(split_counts.items())),
        "overlap_issue_count": len(issues),
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--common", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    common = load_common(args.common)
    report = audit_splits(repo_id=args.repo_id, revision=args.revision, prefix=args.prefix, common=common, token=hf_token(common))
    if args.output:
        write_json(Path(args.output), report)
        print(f"[split-audit] report written: {args.output}")
    print(f"[split-audit] {report['status'].upper()}: {report['unique_split_groups']} groups")
    if report["issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
