#!/usr/bin/env python3
"""Stage 2 for Smol data: stream Stage-1 repos and write a weighted mix."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from huggingface_hub import HfApi

from smol_pipeline import (
    ShardWriter,
    ensure_dataset_repo,
    hf_token,
    load_config,
    stream_parquet_prefix,
    upload_file,
    write_json,
)


def choose_source(rng: random.Random, sources: list[dict], exhausted: set[str]) -> dict:
    available = [source for source in sources if source["name"] not in exhausted]
    if not available:
        raise StopIteration
    total = sum(float(source["weight"]) for source in available)
    point = rng.random() * total
    for source in available:
        point -= float(source["weight"])
        if point <= 0:
            return source
    return available[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if int(cfg.get("stage", -1)) != 2:
        raise ValueError("Stage-2 config must contain stage: 2")
    sources = cfg.get("sources") or []
    if not sources:
        raise ValueError("Stage-2 config requires sources")
    total_weight = sum(float(source.get("weight", 0)) for source in sources)
    if total_weight <= 0:
        raise ValueError("Stage-2 source weights must sum to a positive value")
    token = hf_token()
    api = HfApi(token=token)
    output = cfg["output"]
    ensure_dataset_repo(api, output["repo_id"], token, private=bool(output.get("private", True)))

    if args.dry_run:
        for source in sources:
            print({"name": source["name"], "weight": source["weight"], "repo_id": source["repo_id"], "path_prefix": source.get("path_prefix")})
        return

    iterators = {}
    for source in sources:
        iterators[source["name"]] = iter(stream_parquet_prefix(api, source, token))

    run_id = str(output.get("run_id", "v1"))
    local_root = Path(output.get("local_dir", "work/smol_stage2")) / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(
        local_root / "parts",
        target_size_mb=int(output.get("target_size_mb", 2048)),
        max_documents=int(output.get("max_documents", 1_000_000)),
    )
    rng = random.Random(int(cfg.get("seed", 42)))
    exhausted: set[str] = set()
    counts = {source["name"]: 0 for source in sources}
    estimated_tokens = 0
    rows = 0
    started = time.perf_counter()
    target_tokens = args.target_tokens or cfg.get("target_tokens")

    while True:
        if args.limit_rows is not None and rows >= args.limit_rows:
            break
        if target_tokens is not None and estimated_tokens >= int(target_tokens):
            break
        try:
            source = choose_source(rng, sources, exhausted)
        except StopIteration:
            break
        name = source["name"]
        try:
            row = dict(next(iterators[name]))
        except StopIteration:
            exhausted.add(name)
            continue
        row["source_name"] = name
        row["mix_name"] = cfg.get("name", "smol_mix")
        row["mix_weight"] = float(source["weight"]) / total_weight
        row.setdefault("estimated_tokens", max(1, int(round(len(str(row.get("text", "")).split()) * 1.3))))
        estimated_tokens += int(row["estimated_tokens"] or 1)
        rows += 1
        counts[name] += 1
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
        "stage": 2,
        "name": cfg.get("name", "smol_mix"),
        "run_id": run_id,
        "sources": sources,
        "output_repo": output["repo_id"],
        "path_prefix": output.get("path_prefix", "data/" + run_id),
        "rows": rows,
        "estimated_tokens": estimated_tokens,
        "rows_by_source": counts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = local_root / "manifest.json"
    write_json(manifest_path, manifest)
    upload_file(api, output["repo_id"], manifest_path, f"manifests/{run_id}.json", token)
    print(manifest)


if __name__ == "__main__":
    main()
