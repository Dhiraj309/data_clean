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
from benchmark_contract import (
    benchmark_manifest,
    require_frozen_benchmark,
    require_frozen_sealed_evaluation,
)
from manifest_contract import build_artifact_contract, validate_committed_manifest
from split_policy import assign_split
from pipeline_utils import (
    download_file,
    ensure_repo,
    file_detail,
    hf_token,
    list_repo_files,
    read_remote_json,
    setup_logging,
    slug,
    upload_file,
    local_work_root,
    utc_now,
    write_json,
    write_failure_manifest,
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


class AssignSplitStep:
    """Attach deterministic split and duplicate-family metadata."""
    name = "Deterministic split assignment"

    def __init__(self, dataset_name: str, policy: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self.dataset_name = dataset_name
        self.policy = policy
        self.metrics = metrics

    def run(self, data, rank: int = 0, world_size: int = 1):
        del rank, world_size
        for doc in data:
            metadata = doc.metadata or {}
            group_key = str(metadata.get("dedup_group") or doc.id or doc.text)
            split = assign_split(
                dataset_name=self.dataset_name,
                group_key=group_key,
                metadata=metadata,
                policy=self.policy,
            )
            metadata["split"] = split
            metadata["split_group"] = group_key
            doc.metadata = metadata
            self.metrics["split_counts"][split] = self.metrics["split_counts"].get(split, 0) + 1
            yield doc


def build_index(bundle: Dict[str, Any], force: bool = False) -> Path:
    b = bundle["benchmark"]
    require_frozen_benchmark(b)
    require_frozen_sealed_evaluation(bundle.get("sealed_evaluation"), dataset_training_repo(bundle))
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


def dataset_training_repo(bundle: Dict[str, Any]) -> str:
    return bundle["dataset"]["repos"]["stage3"]


def require_frozen_stage3_contract(bundle: Dict[str, Any]) -> None:
    require_frozen_benchmark(bundle["benchmark"])
    require_frozen_sealed_evaluation(bundle.get("sealed_evaluation"), dataset_training_repo(bundle))


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


def _process_source(bundle: Dict[str, Any], s2_manifest_path: str, api: HfApi, token: str, idx: Path) -> Dict[str, Any]:
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
        try:
            existing_manifest = read_remote_json(s3_repo, manifest_remote, token, common)
            errors = validate_committed_manifest(
                existing_manifest, expected_stage="stage3", available_files=remote_files
            )
            if not errors:
                return {"source_key": source_key, "status": "skipped"}
            logging.getLogger("stage3").warning(
                "Ignoring incomplete manifest %s: %s", manifest_remote, "; ".join(errors)
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("stage3").warning("Ignoring unreadable manifest %s: %s", manifest_remote, exc)

    work = local_work_root(common) / dataset["name"] / "stage3" / source_key
    shutil.rmtree(work, ignore_errors=True)
    input_dir = work / "input"; input_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "seen": 0,
        "accepted": 0,
        "duplicates": 0,
        "custom_rejected": 0,
        "split_counts": {},
    }
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

    input_part_details = []
    for remote_part in s2_manifest["output_parts"]:
        local = download_file(s2_repo, remote_part, "main", token, common, input_dir)
        input_part_details.append(file_detail(local, common["hashing"]["algorithm"], remote_path=remote_part))
        run_pipeline_inline([
            ParquetReader(
                data_folder=str(local.parent), glob_pattern=local.name,
                text_key="text", id_key="id", read_metadata=True,
                default_metadata={"source_dataset": dataset["name"], "stage2_source_key": source_key},
            ),
            count_seen,
            filt,
            AssignSplitStep(dataset["name"], common.get("splits") or {}, metrics),
            sink,
        ])
        local.unlink(missing_ok=True)

    output_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in sink.paths)
    if output_rows != metrics["accepted"]:
        raise IOError("Stage-3 row verification failed")
    contaminated_removed = metrics["seen"] - metrics["accepted"]
    rejected_by_reason = {"benchmark_contamination": contaminated_removed} if contaminated_removed else {}
    uploaded = []
    output_part_details = []
    for part in sink.paths:
        remote = f"{prefix}/{part.name}"
        output_part_details.append(file_detail(part, common["hashing"]["algorithm"], remote_path=remote))
        upload_file(api, s3_repo, part, remote, token, common)
        uploaded.append(remote)
    manifest = {
        "artifact_contract": build_artifact_contract(
            artifact_type="dataset_stage",
            stage="stage3",
            dataset_id=dataset["name"],
            run_id=run_hash,
            config_hash=bundle["stage_hash"],
            source_refs=[
                {
                    "repo_id": s2_repo,
                    "revision": "main",
                    "path": s2_manifest_path,
                    "stage": "stage2",
                }
            ],
            attributes={
                "benchmark_name": bundle["benchmark"].get("name"),
                "benchmark_hash": bundle["benchmark_hash"],
                "benchmark_manifest_id": benchmark_manifest(bundle["benchmark"])["config_hash"],
            },
        ),
        "version": 1,
        "stage": 3,
        "dataset": dataset["name"],
        "source_key": source_key,
        "stage2_manifest": s2_manifest_path,
        "stage3_semantic_hash": run_hash,
        "stage3_config_hash": bundle["stage_hash"],
        "processing_status": "committed",
        "benchmark_name": bundle["benchmark"].get("name"),
        "benchmark_hash": bundle["benchmark_hash"],
        "benchmark_manifest": benchmark_manifest(bundle["benchmark"]),
        "sealed_evaluation": bundle.get("sealed_evaluation"),
        "sealed_evaluation_hash": bundle.get("sealed_evaluation_hash"),
        "n_grams": bundle["benchmark"].get("n_grams", 13),
        "split_policy_version": (common.get("splits") or {}).get("version", 1),
        "split_counts": dict(sorted(metrics["split_counts"].items())),
        "input_part_details": input_part_details,
        "documents_seen": metrics["seen"],
        "documents_accepted": metrics["accepted"],
        "contaminated_removed": contaminated_removed,
        "documents_rejected": contaminated_removed,
        "rejected_by_reason": rejected_by_reason,
        "duplicate_count": 0,
        "duplicate_by_reason": {},
        "error_count": 0,
        "errors_by_reason": {},
        "counts": {
            "seen": metrics["seen"],
            "accepted": metrics["accepted"],
            "rejected": contaminated_removed,
            "duplicates": 0,
            "errors": 0,
            "rejected_by_reason": rejected_by_reason,
            "duplicate_by_reason": {},
            "errors_by_reason": {},
        },
        "output_parts": uploaded,
        "output_part_details": output_part_details,
        "committed_at": utc_now(),
    }
    local_manifest = work / "manifest.json"; write_json(local_manifest, manifest)
    upload_file(api, s3_repo, local_manifest, manifest_remote, token, common)
    shutil.rmtree(work, ignore_errors=True)
    return {"source_key": source_key, "status": "processed", "seen": metrics["seen"], "accepted": metrics["accepted"]}


def process_source(
    bundle: Dict[str, Any], s2_manifest_path: str, api: HfApi, token: str, idx: Path
) -> Dict[str, Any]:
    """Run Stage 3 and persist a retryable failure record on errors."""
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    run_hash = stage3_semantic_hash(bundle)
    key = s2_manifest_path.rstrip("/").split("/")[-2]
    prefix = f"{dataset['name']}/runs/{run_hash[:12]}/sources/{key}"
    try:
        return _process_source(bundle, s2_manifest_path, api, token, idx)
    except Exception as exc:
        contract = build_artifact_contract(
            artifact_type="dataset_stage",
            stage="stage3",
            dataset_id=dataset["name"],
            run_id=run_hash,
            config_hash=bundle["stage_hash"],
            source_refs=[
                {"repo_id": dataset["repos"]["stage2"], "revision": "main", "path": s2_manifest_path, "stage": "stage2"}
            ],
            attributes={
                "source_key": key,
                "benchmark_name": bundle["benchmark"].get("name"),
                "benchmark_hash": bundle["benchmark_hash"],
            },
        )
        write_failure_manifest(
            api=api,
            repo_id=dataset["repos"]["stage3"],
            token=token,
            common=common,
            local_root=local_work_root(common),
            manifest_remote=f"{prefix}/manifest.json",
            artifact_contract=contract,
            stage=3,
            source_key=key,
            exc=exc,
        )
        raise


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
    if not args.dry_run:
        require_frozen_stage3_contract(bundle)
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
