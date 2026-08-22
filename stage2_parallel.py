#!/usr/bin/env python3
"""Parallel-map building blocks for canonical Stage-2 execution.

The expensive document-local work is safe to parallelize.  Global deduplication
is intentionally deferred to one ordered reducer, so worker completion order
can never select the winner of a duplicate family.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader

import filters as custom_filters
from config_loader import load_stage_bundle
from document_identity import document_identity
from pipeline_utils import download_file, file_detail, hf_token, local_work_root, setup_logging, write_json
from stage2_process import (
    LocalParquetSink,
    RegistryAdapterStep,
    build_cleaning,
    build_post,
    build_quality,
    run_pipeline_inline,
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
    sink = LocalParquetSink(output, fields, stage["shard"], common["compression"], metrics)
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
        *build_post(stage),
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
    write_json(work / "map_manifest.json", result)
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
    tasks = [
        {
            "bundle": bundle,
            "token": token,
            "remote_part": remote_part,
            "ordinal": ordinal,
            "source_file": stage1_manifest["source_file"],
            "work": str(root / f"part-{ordinal:05d}"),
        }
        for ordinal, remote_part in enumerate(stage1_manifest["output_parts"])
    ]
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(max(1, workers), len(tasks))) as executor:
        futures = [executor.submit(_map_one_part, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item["ordinal"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    bundle = load_stage_bundle(args.config, 2, args.common, args.registry)
    setup_logging(bundle["common"], "stage2_parallel")
    if bundle["common"].get("plugins", {}).get("filter_paths"):
        custom_filters.load_plugins(bundle["common"]["plugins"]["filter_paths"])
    # The reducer and final Stage-2 commit are intentionally not exposed until
    # their artifact contract is complete. This command is an internal module.
    raise SystemExit("Use stage2_corpus.py until the parallel reducer is enabled.")


if __name__ == "__main__":
    main()
