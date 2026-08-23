#!/usr/bin/env python3
"""Stage 1 for Smol data: stream one HF source and filter its rows."""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

from smol_pipeline import (
    accepts,
    ensure_dataset_repo,
    hf_token,
    load_config,
    normalize_row,
    stable_id,
    stream_hf_dataset,
    upload_file,
    write_json,
    ShardWriter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--limit-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if int(cfg.get("stage", -1)) != 1:
        raise ValueError("Stage-1 config must contain stage: 1")
    source_name = cfg["name"]
    source = cfg["source"]
    output = cfg["output"]
    run_id = str(output.get("run_id") or stable_id({"source": source, "filters": cfg.get("filters", {})}))
    token = hf_token()
    api = HfApi(token=token)

    stream = stream_hf_dataset(source, token)
    iterator = iter(stream)
    try:
        first = next(iterator)
    except StopIteration:
        raise RuntimeError(f"HF source is empty: {source['repo_id']}") from None
    available = sorted(first.keys()) if isinstance(first, dict) else []
    required = cfg.get("required_columns", [cfg.get("columns", {}).get("text", "text")])
    missing = [column for column in required if column not in available]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}; available columns: {available}")
    if args.dry_run:
        print({"source": source, "available_columns": available, "output_repo": output["repo_id"], "run_id": run_id})
        return

    ensure_dataset_repo(api, output["repo_id"], token, private=bool(output.get("private", True)))

    local_root = Path(output.get("local_dir", "work/smol_stage1")) / source_name / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(
        local_root / "parts",
        target_size_mb=int(output.get("target_size_mb", 2048)),
        max_documents=int(output.get("max_documents", 1_000_000)),
    )
    counters = Counter()
    rejected = Counter()
    started = time.perf_counter()
    total_estimated_tokens = 0
    rows_seen = 0

    for raw in chain([first], iterator):
        rows_seen += 1
        if args.limit_rows is not None and rows_seen > args.limit_rows:
            break
        row = normalize_row(raw, cfg, source_name)
        counters["seen"] += 1
        ok, reason = accepts(row, cfg.get("filters", {}))
        if not ok:
            rejected[reason or "rejected"] += 1
            continue
        counters["accepted"] += 1
        total_estimated_tokens += row["estimated_tokens"]
        if args.limit_tokens is not None and total_estimated_tokens > args.limit_tokens:
            break
        part = writer.add(row)
        if part is not None:
            remote = f"{output.get('path_prefix', 'data/' + run_id).rstrip('/')}/{part.name}"
            upload_file(api, output["repo_id"], part, remote, token)
            part.unlink(missing_ok=True)

    part = writer.flush()
    if part is not None:
        remote = f"{output.get('path_prefix', 'data/' + run_id).rstrip('/')}/{part.name}"
        upload_file(api, output["repo_id"], part, remote, token)
        part.unlink(missing_ok=True)

    manifest = {
        "stage": 1,
        "name": source_name,
        "run_id": run_id,
        "source": source,
        "filters": cfg.get("filters", {}),
        "output_repo": output["repo_id"],
        "path_prefix": output.get("path_prefix", "data/" + run_id),
        "rows_seen": counters["seen"],
        "rows_accepted": counters["accepted"],
        "rows_rejected": sum(rejected.values()),
        "estimated_tokens": total_estimated_tokens,
        "rejected_by_reason": dict(rejected),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = local_root / "manifest.json"
    write_json(manifest_path, manifest)
    upload_file(api, output["repo_id"], manifest_path, f"manifests/{run_id}.json", token)
    print(manifest)


if __name__ == "__main__":
    main()
