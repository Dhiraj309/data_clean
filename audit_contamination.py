#!/usr/bin/env python3
"""Compare training artifacts against a sealed evaluation corpus."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from contamination import build_eval_index, compare_training_texts
from config_loader import load_common, load_yaml
from manifest_contract import validate_committed_manifest
from pipeline_utils import download_file, hf_token, list_repo_files, write_json


def local_texts(path: Path, text_column: str = "text") -> Iterator[str]:
    paths = sorted(path.rglob("*") if path.is_dir() else [path])
    for item in paths:
        if not item.is_file():
            continue
        if item.suffix == ".parquet":
            with pq.ParquetFile(item) as parquet:
                for batch in parquet.iter_batches(batch_size=8192, columns=[text_column]):
                    for text in batch.column(text_column).to_pylist():
                        if text:
                            yield str(text)
        elif item.suffix in {".jsonl", ".json"} or item.name.endswith(".jsonl.gz"):
            opener = gzip.open if item.name.endswith(".gz") else open
            with opener(item, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        text = row.get(text_column)
                        if text:
                            yield str(text)


def remote_training_texts(repo_id: str, prefix: str, common: Dict[str, Any], token: str) -> Iterator[str]:
    api = HfApi(token=token)
    files = set(list_repo_files(api, repo_id, token, common))
    for manifest_path in sorted(path for path in files if path.endswith("/manifest.json") and (not prefix or path.startswith(prefix))):
        manifest_path_local = download_file(repo_id, manifest_path, "main", token, common)
        manifest = json.loads(manifest_path_local.read_text(encoding="utf-8"))
        if validate_committed_manifest(manifest, expected_stage="stage3", available_files=files):
            continue
        for remote_part in manifest["output_parts"]:
            local_part = download_file(repo_id, remote_part, "main", token, common)
            with pq.ParquetFile(local_part) as parquet:
                columns = ["text", "split"] if "split" in parquet.schema_arrow.names else ["text"]
                for batch in parquet.iter_batches(batch_size=8192, columns=columns):
                    texts = batch.column("text").to_pylist()
                    splits = batch.column("split").to_pylist() if len(columns) == 2 else ["train"] * len(texts)
                    for text, split in zip(texts, splits):
                        if text and split == "train":
                            yield str(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo-id", required=True)
    parser.add_argument("--training-prefix", default="")
    parser.add_argument("--sealed-config", required=True, help="Frozen sealed_evaluation.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-grams", type=int, default=13)
    parser.add_argument("--near-bucket-bits", type=int, default=8)
    parser.add_argument("--near-distance", type=int, default=3)
    args = parser.parse_args()

    common = load_common(args.common)
    sealed = load_yaml(args.sealed_config)
    sealed_repo = sealed.get("repo_id")
    if not sealed_repo or sealed_repo.startswith("REPLACE_"):
        raise ValueError("sealed evaluation config must specify a real repo_id")
    if sealed_repo == args.training_repo_id:
        raise ValueError("sealed evaluation repo must differ from training repo")
    token = hf_token(common)
    eval_api = HfApi(token=token)
    eval_files = set(list_repo_files(eval_api, sealed_repo, token, common))
    if sealed.get("freeze_status") != "frozen":
        raise ValueError("sealed evaluation config must set freeze_status='frozen'")
    configured_paths = set(sealed.get("paths") or [])

    def sealed_texts() -> Iterator[str]:
        candidates = sorted(
            p for p in eval_files
            if p.endswith((".parquet", ".jsonl", ".json", ".jsonl.gz"))
            and (not configured_paths or p in configured_paths)
        )
        for path in candidates:
            local = download_file(sealed_repo, path, sealed.get("revision", "main"), token, common)
            yield from local_texts(local, sealed.get("text_column", "text"))

    evaluation_index = build_eval_index(sealed_texts(), args.n_grams, args.near_bucket_bits)
    if evaluation_index["documents"] == 0:
        raise ValueError("sealed evaluation repository contains no configured text files")
    report = compare_training_texts(
        remote_training_texts(args.training_repo_id, args.training_prefix, common, token),
        evaluation_index,
        args.n_grams,
        args.near_bucket_bits,
        args.near_distance,
    )
    report.update({
        "audit": "laughlm_contamination",
        "training_repo_id": args.training_repo_id,
        "training_prefix": args.training_prefix,
        "sealed_evaluation_repo_id": sealed_repo,
        "sealed_evaluation_version": sealed.get("version"),
        "sealed_evaluation_name": sealed.get("name"),
        "contamination_detected": any(
            report[key] > 0
            for key in (
                "exact_match_documents",
                "normalized_match_documents",
                "ngram_match_documents",
                "near_duplicate_documents",
            )
        ),
        "status": "reported",
    })
    write_json(Path(args.output), report)
    print(f"[contamination-audit] report written: {args.output}")
    print(f"[contamination-audit] compared {report['training_documents']} training documents")


if __name__ == "__main__":
    main()
