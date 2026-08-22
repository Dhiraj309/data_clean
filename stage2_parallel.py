#!/usr/bin/env python3
"""Parallel-map building blocks for canonical Stage-2 execution.

The expensive document-local work is safe to parallelize.  Global deduplication
is intentionally deferred to one ordered reducer, so worker completion order
can never select the winner of a duplicate family.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader

import filters as custom_filters
from config_loader import load_stage_bundle, stage2_semantic_hash
from document_identity import document_identity
from manifest_contract import build_artifact_contract, validate_committed_manifest
from pipeline_utils import download_file, ensure_repo, file_detail, hf_token, list_repo_files, local_work_root, read_remote_json, setup_logging, slug, upload_file, utc_now, write_json
from stage2_process import (
    CommittedHashStore,
    CrashSafeDedupStep,
    LocalParquetSink,
    RegistryAdapterStep,
    build_cleaning,
    build_post,
    build_quality,
    remote_prefix,
    run_pipeline_inline,
    list_stage1_manifests,
    source_key as make_source_key,
)


CANDIDATE_EXACT = "_stage2_exact_digest"
CANDIDATE_NORMALIZED = "_stage2_normalized_digest"
CANDIDATE_NEAR = "_stage2_near_fingerprint"


class CandidateIdentityStep(PipelineStep):
    """Attach identities once so the deterministic reducer never re-hashes."""

    name = "Stage-2 candidate identity"

    def __init__(self, algorithm: str, policy: Dict[str, Any], near_enabled: bool) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.policy = policy
        self.near_enabled = near_enabled

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        del rank, world_size
        for doc in data:
            identity = document_identity(
                doc.text,
                self.algorithm,
                self.policy,
                include_near_fingerprint=self.near_enabled,
            )
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata[CANDIDATE_EXACT] = identity.exact_digest
            doc.metadata[CANDIDATE_NORMALIZED] = identity.normalized_digest
            doc.metadata[CANDIDATE_NEAR] = identity.near_fingerprint
            yield doc


def _map_one_part(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run document-local Stage-2 work for one immutable Stage-1 part."""

    bundle = task["bundle"]
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    work = Path(task["work"])
    work.mkdir(parents=True, exist_ok=True)
    output = work / "candidates"
    token = task["token"]
    remote_part = task["remote_part"]
    local = download_file(
        dataset["repos"]["stage1"], remote_part, "main", token, common, work / "input"
    )
    metrics: Dict[str, Any] = {
        "seen": 0,
        "accepted": 0,
        "custom_rejected": 0,
        "rejected_by_reason": {},
    }

    def count_seen(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> Iterator[Document]:
        del rank, world_size
        for doc in data:
            metrics["seen"] += 1
            yield doc

    near_policy = (common.get("deduplication") or {}).get("near_duplicate", {})
    fields = [
        *stage.get("output_metadata_fields", []),
        CANDIDATE_EXACT,
        CANDIDATE_NORMALIZED,
        CANDIDATE_NEAR,
    ]
    # Candidate files are an internal map/reduce boundary, not final shards.
    # Keep their per-worker buffers deliberately small so a 24-core VM does
    # not reserve final-shard-sized Python row buffers in every child process.
    candidate_shard = dict(stage["shard"])
    candidate_shard["target_size_mb"] = min(int(candidate_shard["target_size_mb"]), 64)
    candidate_shard["max_documents"] = min(int(candidate_shard["max_documents"]), 25_000)
    sink = LocalParquetSink(output, fields, candidate_shard, common["compression"], metrics)
    pipeline: List[Any] = [
        ParquetReader(
            data_folder=str(local.parent),
            glob_pattern=local.name,
            text_key=stage.get("text_column", "text"),
            id_key=stage.get("id_column", "id"),
            read_metadata=True,
            default_metadata={
                "source_dataset": dataset["name"],
                "source_file": task["source_file"],
            },
        ),
        count_seen,
        *build_cleaning(stage),
        RegistryAdapterStep(stage.get("custom_filters") or [], metrics),
        *build_quality(stage),
        CandidateIdentityStep(
            common["hashing"]["algorithm"],
            (common.get("deduplication") or {}).get("normalized", {}),
            bool(near_policy.get("enabled", False)),
        ),
        sink,
    ]
    run_pipeline_inline(pipeline)
    result = {
        "ordinal": task["ordinal"],
        "remote_part": remote_part,
        "input_detail": file_detail(local, common["hashing"]["algorithm"], remote_path=remote_part),
        "candidate_parts": [str(path) for path in sink.paths],
        "candidate_rows": sum(pq.ParquetFile(path).metadata.num_rows for path in sink.paths),
        "metrics": metrics,
    }
    candidate_prefix = task["candidate_prefix"]
    api = HfApi(token=token)
    remote_candidates = []
    for candidate in sink.paths:
        remote = f"{candidate_prefix}/{candidate.name}"
        upload_file(api, dataset["repos"]["stage2"], candidate, remote, token, common)
        remote_candidates.append(remote)
    result["candidate_remote_parts"] = remote_candidates
    result["processing_status"] = "committed"
    manifest_path = work / "map_manifest.json"
    write_json(manifest_path, result)
    upload_file(api, dataset["repos"]["stage2"], manifest_path, f"{candidate_prefix}/manifest.json", token, common)
    local.unlink(missing_ok=True)
    return result


def map_source_parts(
    bundle: Dict[str, Any],
    stage1_manifest: Dict[str, Any],
    *,
    workers: int,
) -> List[Dict[str, Any]]:
    """Map source parts concurrently and retain ordered, local candidate paths."""

    common = bundle["common"]
    token = hf_token(common)
    source_key = stage1_manifest["source_key"]
    root = local_work_root(common) / "stage2_parallel" / bundle["dataset"]["name"] / source_key
    run_hash = stage2_semantic_hash(bundle)
    final_key = make_source_key(stage1_manifest, run_hash)
    prefix = remote_prefix(bundle["dataset"]["name"], bundle["stage"]["dedup_namespace"], run_hash, final_key)
    api = HfApi(token=token)
    remote_files = set(list_repo_files(api, bundle["dataset"]["repos"]["stage2"], token, common))
    tasks = [
        {
            "bundle": bundle,
            "token": token,
            "remote_part": remote_part,
            "ordinal": ordinal,
            "source_file": stage1_manifest["source_file"],
            "work": str(root / f"part-{ordinal:05d}"),
            "candidate_prefix": f"{prefix}/map/part-{ordinal:05d}",
        }
        for ordinal, remote_part in enumerate(stage1_manifest["output_parts"])
    ]
    results: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for task in tasks:
        manifest_remote = f"{task['candidate_prefix']}/manifest.json"
        if manifest_remote not in remote_files:
            pending.append(task)
            continue
        manifest = read_remote_json(bundle["dataset"]["repos"]["stage2"], manifest_remote, token, common)
        remote_parts = manifest.get("candidate_remote_parts") or []
        if manifest.get("processing_status") != "committed" or not remote_parts or any(path not in remote_files for path in remote_parts):
            pending.append(task)
            continue
        local_parts = [str(download_file(bundle["dataset"]["repos"]["stage2"], path, "main", token, common, root / "reused" / f"part-{task['ordinal']:05d}")) for path in remote_parts]
        manifest["candidate_parts"] = local_parts
        results.append(manifest)
    if pending:
        with ProcessPoolExecutor(max_workers=min(max(1, workers), len(pending))) as executor:
            futures = [executor.submit(_map_one_part, task) for task in pending]
            for future in as_completed(futures):
                results.append(future.result())
    return sorted(results, key=lambda item: item["ordinal"])


def _candidate_documents(parts: List[str]) -> Iterator[Document]:
    """Yield mapped candidates in source-part and row order."""

    for path in parts:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                text = row.pop("text")
                identifier = row.pop("id")
                yield Document(text=text, id=identifier, metadata=row)


def reduce_source_candidates(
    bundle: Dict[str, Any],
    mapped_parts: List[Dict[str, Any]],
    store: CommittedHashStore,
    *,
    source_key: str,
) -> Dict[str, Any]:
    """Apply one ordered global-dedup decision pass to mapped candidates.

    This function is deliberately single-process.  Source priority is enforced
    by the caller; source-part and row order are retained by ``mapped_parts``.
    """

    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    work = local_work_root(common) / "stage2_parallel" / dataset["name"] / source_key / "reduce"
    work.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = {
        "seen": sum(int(item["metrics"]["seen"]) for item in mapped_parts),
        "accepted": 0,
        "duplicates": 0,
        "custom_rejected": sum(int(item["metrics"]["custom_rejected"]) for item in mapped_parts),
        "near_duplicates": 0,
        "rejected_by_reason": {},
        "duplicate_by_reason": {},
    }
    for item in mapped_parts:
        for reason, count in item["metrics"]["rejected_by_reason"].items():
            metrics["rejected_by_reason"][reason] = metrics["rejected_by_reason"].get(reason, 0) + int(count)
    near_policy = (common.get("deduplication") or {}).get("near_duplicate", {})
    sidecar = work / "accepted.hashes"
    normalized_sidecar = work / "accepted.normalized.hashes"
    near_sidecar = work / "accepted.near.hashes" if near_policy.get("enabled", False) else None
    dedup = CrashSafeDedupStep(
        store,
        common["hashing"]["algorithm"],
        int(common["hashing"].get("dedup_batch_size", 500)),
        sidecar,
        normalized_sidecar,
        near_sidecar,
        metrics,
        (common.get("deduplication") or {}).get("normalized", {}),
        near_policy,
        int(dataset.get("source_priority", 0)),
    )
    sink = LocalParquetSink(
        work / "parts",
        stage.get("output_metadata_fields", []),
        stage["shard"],
        common["compression"],
        metrics,
    )
    candidate_paths = [path for item in mapped_parts for path in item["candidate_parts"]]
    pipeline: List[Any] = [
        lambda _unused: _candidate_documents(candidate_paths),
        dedup,
        *build_post(stage),
        sink,
    ]
    run_pipeline_inline(pipeline)
    return {
        "work": str(work),
        "parts": [str(path) for path in sink.paths],
        "sidecar": str(sidecar),
        "normalized_sidecar": str(normalized_sidecar),
        "near_sidecar": str(near_sidecar) if near_sidecar is not None else None,
        "metrics": metrics,
    }


def commit_reduced_source(
    bundle: Dict[str, Any],
    stage1_manifest_path: str,
    stage1_manifest: Dict[str, Any],
    mapped_parts: List[Dict[str, Any]],
    reduced: Dict[str, Any],
    api: HfApi,
    token: str,
    store: CommittedHashStore,
    timings: Dict[str, float],
) -> Dict[str, Any]:
    """Publish reducer output with the same manifest-last contract as Stage 2."""

    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    run_hash = stage2_semantic_hash(bundle)
    key = make_source_key(stage1_manifest, run_hash)
    prefix = remote_prefix(dataset["name"], stage["dedup_namespace"], run_hash, key)
    upload_started = time.perf_counter()
    output_parts: List[str] = []
    output_details: List[Dict[str, Any]] = []
    for local_path in map(Path, reduced["parts"]):
        remote = f"{prefix}/{local_path.name}"
        output_details.append(file_detail(local_path, common["hashing"]["algorithm"], remote_path=remote))
        upload_file(api, dataset["repos"]["stage2"], local_path, remote, token, common)
        output_parts.append(remote)
    sidecar = Path(reduced["sidecar"])
    normalized = Path(reduced["normalized_sidecar"])
    side_remote = f"{prefix}/accepted.hashes"
    normalized_remote = f"{prefix}/accepted.normalized.hashes"
    upload_file(api, dataset["repos"]["stage2"], sidecar, side_remote, token, common)
    upload_file(api, dataset["repos"]["stage2"], normalized, normalized_remote, token, common)
    near_remote = None
    if reduced["near_sidecar"]:
        near_remote = f"{prefix}/accepted.near.hashes"
        upload_file(api, dataset["repos"]["stage2"], Path(reduced["near_sidecar"]), near_remote, token, common)
    timings["upload_seconds"] = time.perf_counter() - upload_started
    timings["total_seconds"] = timings["map_seconds"] + timings["reduce_seconds"] + timings["upload_seconds"]
    metrics = reduced["metrics"]
    digest_size = 16 if common["hashing"]["algorithm"] == "xxh3_128" else 8
    manifest = {
        "artifact_contract": build_artifact_contract(
            artifact_type="dataset_stage", stage="stage2", dataset_id=dataset["name"],
            run_id=run_hash, config_hash=bundle["stage_hash"],
            source_refs=[{"repo_id": dataset["repos"]["stage1"], "revision": "main", "path": stage1_manifest_path, "stage": "stage1"}],
            attributes={"source_file": stage1_manifest["source_file"], "dedup_namespace": stage["dedup_namespace"], "execution": "parallel_map_ordered_reduce_v1"},
        ),
        "version": 1, "stage": 2, "dataset": dataset["name"], "dedup_namespace": stage["dedup_namespace"],
        "source_key": key, "stage1_manifest": stage1_manifest_path, "stage2_semantic_hash": run_hash,
        "stage2_config_hash": bundle["stage_hash"], "processing_status": "committed",
        "source_priority": int(dataset.get("source_priority", 0)),
        "source_priority_policy": (common.get("deduplication") or {}).get("source_priority", {}),
        "hash_algorithm": common["hashing"]["algorithm"], "hash_digest_size": digest_size,
        "hash_sidecar": side_remote, "normalized_hash_sidecar": normalized_remote,
        "near_duplicate_sidecar": near_remote, "near_duplicate_policy": (common.get("deduplication") or {}).get("near_duplicate", {}),
        "input_part_details": [item["input_detail"] for item in mapped_parts],
        "documents_seen": metrics["seen"], "accepted": metrics["accepted"], "duplicates": metrics["duplicates"],
        "near_duplicates": metrics["near_duplicates"], "custom_rejections": metrics["custom_rejected"],
        "rejected": metrics["custom_rejected"], "rejected_by_reason": dict(sorted(metrics["rejected_by_reason"].items())),
        "duplicate_by_reason": dict(sorted(metrics["duplicate_by_reason"].items())), "error_count": 0, "errors_by_reason": {},
        "counts": {"seen": metrics["seen"], "accepted": metrics["accepted"], "rejected": metrics["custom_rejected"], "duplicates": metrics["duplicates"], "near_duplicates": metrics["near_duplicates"], "errors": 0, "rejected_by_reason": dict(sorted(metrics["rejected_by_reason"].items())), "duplicate_by_reason": dict(sorted(metrics["duplicate_by_reason"].items())), "errors_by_reason": {}},
        "output_parts": output_parts, "output_part_details": output_details, "parallel_map": {"workers": len(mapped_parts), "parts": [{"ordinal": item["ordinal"], "candidate_rows": item["candidate_rows"]} for item in mapped_parts]}, "timings": {name: round(value, 3) for name, value in timings.items()}, "committed_at": utc_now(),
    }
    manifest_path = Path(reduced["work"]) / "manifest.json"
    write_json(manifest_path, manifest)
    upload_file(api, dataset["repos"]["stage2"], manifest_path, f"{prefix}/manifest.json", token, common)
    store.commit_sidecars(key, f"{prefix}/manifest.json", sidecar, normalized, Path(reduced["near_sidecar"]) if reduced["near_sidecar"] else None, digest_size)
    return {"source": stage1_manifest["source_file"], "status": "processed", **metrics}


def reuse_committed_source(
    bundle: Dict[str, Any], stage1_manifest: Dict[str, Any], api: HfApi, token: str, store: CommittedHashStore
) -> bool:
    """Validate a final commit and restore its hashes into this run's store."""
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    run_hash = stage2_semantic_hash(bundle)
    key = make_source_key(stage1_manifest, run_hash)
    prefix = remote_prefix(dataset["name"], stage["dedup_namespace"], run_hash, key)
    manifest_remote = f"{prefix}/manifest.json"
    files = set(list_repo_files(api, dataset["repos"]["stage2"], token, common))
    if manifest_remote not in files:
        return False
    manifest = read_remote_json(dataset["repos"]["stage2"], manifest_remote, token, common)
    if validate_committed_manifest(manifest, expected_stage="stage2", available_files=files):
        return False
    normalized = manifest.get("normalized_hash_sidecar")
    if manifest.get("hash_sidecar") not in files or normalized not in files:
        return False
    near = manifest.get("near_duplicate_sidecar")
    sidecar = download_file(dataset["repos"]["stage2"], manifest["hash_sidecar"], "main", token, common)
    normalized_sidecar = download_file(dataset["repos"]["stage2"], normalized, "main", token, common)
    near_sidecar = download_file(dataset["repos"]["stage2"], near, "main", token, common) if near in files else None
    store.commit_sidecars(key, manifest_remote, sidecar, normalized_sidecar, near_sidecar, int(manifest["hash_digest_size"]), int((manifest.get("near_duplicate_policy") or {}).get("bucket_bits", 8)))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/stage2_corpus.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    from stage2_corpus import load_bundles, load_plan

    plan = load_plan(args.plan)
    bundles = load_bundles(plan, args.common, args.registry)
    common = bundles[0]["common"]
    setup_logging(common, "stage2_parallel")
    if common.get("plugins", {}).get("filter_paths"):
        custom_filters.load_plugins(common["plugins"]["filter_paths"])
    if args.dry_run:
        for bundle in bundles:
            print(f"[stage2-parallel] priority={bundle['dataset'].get('source_priority', 0)} dataset={bundle['dataset']['name']}")
        return
    token = hf_token(common)
    api = HfApi(token=token)
    db = local_work_root(common) / "dedup" / f"{slug(plan['dedup_namespace'])}.sqlite3"
    store = CommittedHashStore(db, plan["dedup_namespace"], common["hashing"]["algorithm"])
    try:
        for bundle in bundles:
            dataset = bundle["dataset"]
            ensure_repo(api, dataset["repos"]["stage1"], token, common, must_exist=True)
            ensure_repo(api, dataset["repos"]["stage2"], token, common)
            for manifest_path in list_stage1_manifests(bundle, bundle["stage1_bundle"], api, token):
                stage1_manifest = read_remote_json(dataset["repos"]["stage1"], manifest_path, token, common)
                if reuse_committed_source(bundle, stage1_manifest, api, token, store):
                    print({"source": stage1_manifest["source_file"], "status": "skipped"})
                    continue
                started = time.perf_counter()
                mapped = map_source_parts(bundle, stage1_manifest, workers=args.workers)
                mapped_at = time.perf_counter()
                key = make_source_key(stage1_manifest, stage2_semantic_hash(bundle))
                reduced = reduce_source_candidates(bundle, mapped, store, source_key=key)
                reduced_at = time.perf_counter()
                result = commit_reduced_source(
                    bundle, manifest_path, stage1_manifest, mapped, reduced, api, token, store,
                    {"map_seconds": mapped_at - started, "reduce_seconds": reduced_at - mapped_at},
                )
                print(result)
    finally:
        store.close()


if __name__ == "__main__":
    main()
