#!/usr/bin/env python3
"""Stage 3: benchmark n-gram decontamination for one processed dataset."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table

from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontFilter, NGramsDecontIndexer
from datatrove.pipeline.readers import ParquetReader

from config_loader import load_stage3_bundle, stage2_semantic_hash, stage3_semantic_hash
from pipeline_utils import (
    download_file,
    ensure_repo,
    hf_token,
    list_repo_files,
    setup_logging,
    slug,
    upload_file,
    utc_now,
    write_json,
)
from stage2_process import LocalParquetSink, run_pipeline_inline


def decont_config(bundle: Dict[str, Any]) -> NGramsDecontConfig:
    b = bundle["benchmark"]
    return NGramsDecontConfig(
        n_grams=int(b.get("n_grams", 13)),
        find_query_ngrams=bool(b.get("find_query_ngrams", False)),
        find_overlap_ngrams=bool(b.get("find_overlap_ngrams", True)),
    )


def index_dir(bundle: Dict[str, Any]) -> Path:
    root = Path(bundle["benchmark"].get("index_root", "./work/decont_index"))
    return root / bundle["benchmark_hash"][:12]


def index_ready(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.index.hashes"))


def build_index(bundle: Dict[str, Any], force: bool = False) -> Path:
    b = bundle["benchmark"]
    tasks = b.get("lighteval_tasks") or []
    if not tasks:
        raise ValueError(
            f"{bundle['benchmark_path']} has no lighteval_tasks. Freeze the LaughLM evaluation suite first."
        )
    out = index_dir(bundle)
    if index_ready(out) and not force:
        return out
    if force:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    indexer = NGramsDecontIndexer(
        output_folder=str(out),
        lighteval_tasks=tasks,
        custom_lighteval_tasks=b.get("custom_lighteval_tasks"),
        config=decont_config(bundle),
        language=b.get("language", "en"),
    )
    result = indexer.run(data=None, rank=0, world_size=1)
    if result is not None:
        for _ in result:
            pass
    if not index_ready(out):
        raise RuntimeError(f"No decontamination index was produced in {out}")
    return out


def list_stage2_manifests(bundle: Dict[str, Any], api: HfApi, token: str) -> List[str]:
    common, dataset = bundle["common"], bundle["dataset"]
    s2_bundle = bundle["stage2_bundle"]
    s2_hash = stage2_semantic_hash(s2_bundle)
    namespace = slug(s2_bundle["stage"]["dedup_namespace"])
    prefix = f"{dataset['name']}/dedup/{namespace}/runs/{s2_hash[:12]}/sources/"
    return sorted(
        f for f in list_repo_files(api, dataset["repos"]["stage2"], token, common)
        if f.startswith(prefix) and f.endswith("/manifest.json")
    )


def process_source(bundle: Dict[str, Any], s2_manifest_path: str, api: HfApi, token: str, idx: Path) -> Dict[str, Any]:
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    s2_repo, s3_repo = dataset["repos"]["stage2"], dataset["repos"]["stage3"]
    s2_manifest_local = download_file(s2_repo, s2_manifest_path, "main", token, common)
    s2_manifest = json.loads(s2_manifest_local.read_text(encoding="utf-8"))
    source_key = s2_manifest["source_key"]
    run_hash = stage3_semantic_hash(bundle)
    prefix = f"{dataset['name']}/runs/{run_hash[:12]}/sources/{source_key}"
    manifest_remote = f"{prefix}/manifest.json"
    remote_files = set(list_repo_files(api, s3_repo, token, common))
    if manifest_remote in remote_files:
        return {"source_key": source_key, "status": "skipped"}

    work = Path(common["storage"]["local_work_dir"]) / dataset["name"] / "stage3" / source_key
    shutil.rmtree(work, ignore_errors=True)
    input_dir = work / "input"; input_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"seen": 0, "accepted": 0, "duplicates": 0, "custom_rejected": 0}
    metadata_fields = stage.get("output_metadata_fields") or bundle["stage2_bundle"]["stage"].get("output_metadata_fields", [])
    sink = LocalParquetSink(work / "parts", metadata_fields, stage["shard"], common["compression"], metrics)
    filt = NGramsDecontFilter(
        index_folder=str(idx),
        config=decont_config(bundle),
        language=bundle["benchmark"].get("language", "en"),
    )

    def count_seen(data, rank: int = 0, world_size: int = 1):
        del rank, world_size
        for doc in data:
            metrics["seen"] += 1
            yield doc

    for remote_part in s2_manifest["output_parts"]:
        local = download_file(s2_repo, remote_part, "main", token, common, input_dir)
        run_pipeline_inline([
            ParquetReader(
                data_folder=str(local.parent), glob_pattern=local.name,
                text_key="text", id_key="id", read_metadata=True,
                default_metadata={"source_dataset": dataset["name"], "stage2_source_key": source_key},
            ),
            count_seen,
            filt,
            sink,
        ])
        local.unlink(missing_ok=True)

    output_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in sink.paths)
    if output_rows != metrics["accepted"]:
        raise IOError("Stage-3 row verification failed")
    uploaded = []
    for part in sink.paths:
        remote = f"{prefix}/{part.name}"
        upload_file(api, s3_repo, part, remote, token, common)
        uploaded.append(remote)
    manifest = {
        "version": 1,
        "stage": 3,
        "dataset": dataset["name"],
        "source_key": source_key,
        "stage2_manifest": s2_manifest_path,
        "stage3_semantic_hash": run_hash,
        "stage3_config_hash": bundle["stage_hash"],
        "benchmark_name": bundle["benchmark"].get("name"),
        "benchmark_hash": bundle["benchmark_hash"],
        "n_grams": bundle["benchmark"].get("n_grams", 13),
        "documents_seen": metrics["seen"],
        "documents_accepted": metrics["accepted"],
        "contaminated_removed": metrics["seen"] - metrics["accepted"],
        "output_parts": uploaded,
        "committed_at": utc_now(),
    }
    local_manifest = work / "manifest.json"; write_json(local_manifest, manifest)
    upload_file(api, s3_repo, local_manifest, manifest_remote, token, common)
    shutil.rmtree(work, ignore_errors=True)
    return {"source_key": source_key, "status": "processed", "seen": metrics["seen"], "accepted": metrics["accepted"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="LaughLM Stage 3 benchmark decontamination")
    parser.add_argument("--config", required=True, help="configs/stage3/<dataset>.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle = load_stage3_bundle(args.config, args.common, args.registry)
    common, dataset = bundle["common"], bundle["dataset"]
    setup_logging(common, "stage3")
    token = hf_token(common); api = HfApi(token=token)
    ensure_repo(api, dataset["repos"]["stage2"], token, common, must_exist=True)
    ensure_repo(api, dataset["repos"]["stage3"], token, common)

    idx = index_dir(bundle)
    if args.build_index or args.force_rebuild_index:
        idx = build_index(bundle, force=args.force_rebuild_index)
    if args.index_only:
        return
    if not index_ready(idx) and not args.dry_run:
        raise RuntimeError(f"Decontamination index missing: {idx}. Run with --build-index first.")

    manifests = list_stage2_manifests(bundle, api, token)
    if args.limit_sources is not None:
        manifests = manifests[: args.limit_sources]
    table = Table(title=f"Stage 3 dry-run — {dataset['name']}")
    table.add_column("Field"); table.add_column("Value")
    table.add_row("Stage-2 sources", str(len(manifests)))
    table.add_row("Benchmark", str(bundle["benchmark"].get("name")))
    table.add_row("Benchmark hash", bundle["benchmark_hash"][:12])
    table.add_row("Stage-3 run", stage3_semantic_hash(bundle)[:12])
    Console().print(table)
    if args.dry_run:
        return
    for manifest in manifests:
        Console().print(process_source(bundle, manifest, api, token, idx))


if __name__ == "__main__":
    main()
