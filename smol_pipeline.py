#!/usr/bin/env python3
"""Shared streaming utilities for the Smol-data Stage 1/Stage 2 pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url


OUTPUT_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("id", pa.string()),
        ("url", pa.string()),
        ("language", pa.string()),
        ("dataset", pa.string()),
        ("source_name", pa.string()),
        ("source_repo", pa.string()),
        ("source_config", pa.string()),
        ("source_split", pa.string()),
        ("language_score", pa.float64()),
        ("fasttext_score", pa.float64()),
        ("score", pa.float64()),
        ("int_score", pa.int64()),
        ("token_count", pa.int64()),
        ("word_count", pa.int64()),
        ("char_count", pa.int64()),
        ("estimated_tokens", pa.int64()),
        ("mix_name", pa.string()),
        ("mix_weight", pa.float64()),
    ]
)


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(os.path.expandvars(p.read_text(encoding="utf-8"))) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {p}")
    return data


def stable_id(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    return token


def get_path(row: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None


def as_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def stream_hf_dataset(source: dict[str, Any], token: str):
    repo_id = source["repo_id"]
    kwargs: dict[str, Any] = {
        "split": source.get("split", "train"),
        "streaming": True,
        "revision": source.get("revision", "main"),
        "token": token,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    return load_dataset(repo_id, **kwargs)


def stream_parquet_prefix(api: HfApi, source: dict[str, Any], token: str):
    repo_id = source["repo_id"]
    revision = source.get("revision", "main")
    prefix = source.get("path_prefix", "data/").rstrip("/") + "/"
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision, token=token)
    parquet_files = sorted(
        p for p in files if p.startswith(prefix) and p.lower().endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(f"No Parquet files found in {repo_id}:{prefix}")
    urls = [hf_hub_url(repo_id, filename=p, repo_type="dataset", revision=revision) for p in parquet_files]
    return load_dataset(
        "parquet",
        data_files={"train": urls},
        split="train",
        streaming=True,
        token=token,
    )


def ensure_dataset_repo(api: HfApi, repo_id: str, token: str, private: bool = True) -> None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)


def upload_file(api: HfApi, repo_id: str, path: Path, path_in_repo: str, token: str) -> None:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload {path_in_repo}",
    )


def normalize_row(row: dict[str, Any], cfg: dict[str, Any], source_name: str) -> dict[str, Any]:
    columns = cfg.get("columns", {})
    text = get_path(row, columns.get("text", "text"))
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    word_count = len(text.split())
    char_count = len(text)
    raw_token_count = as_int(get_path(row, columns.get("token_count", "token_count")))
    estimated_tokens = raw_token_count or max(1, int(round(word_count * 1.3)))
    return {
        "text": text,
        "id": as_string(get_path(row, columns.get("id", "id"))),
        "url": as_string(get_path(row, columns.get("url", "url"))),
        "language": as_string(get_path(row, columns.get("language", "language"))),
        "dataset": as_string(get_path(row, columns.get("dataset", "dataset"))) or source_name,
        "source_name": source_name,
        "source_repo": cfg["source"]["repo_id"],
        "source_config": cfg["source"].get("config"),
        "source_split": cfg["source"].get("split", "train"),
        "language_score": as_float(get_path(row, columns.get("language_score", "language_score"))),
        "fasttext_score": as_float(get_path(row, columns.get("fasttext_score", "fasttext_score"))),
        "score": as_float(get_path(row, columns.get("score", "score"))),
        "int_score": as_int(get_path(row, columns.get("int_score", "int_score"))),
        "token_count": raw_token_count,
        "word_count": word_count,
        "char_count": char_count,
        "estimated_tokens": estimated_tokens,
    }


def _check_min(value: Any, threshold: Any) -> bool:
    return threshold is None or (value is not None and value >= threshold)


def _check_max(value: Any, threshold: Any) -> bool:
    return threshold is None or (value is not None and value <= threshold)


def accepts(row: dict[str, Any], filters: dict[str, Any]) -> tuple[bool, str | None]:
    if not row["text"].strip():
        return False, "empty_text"
    if not _check_min(row["word_count"], filters.get("min_words")):
        return False, "too_few_words"
    if not _check_max(row["word_count"], filters.get("max_words")):
        return False, "too_many_words"
    if not _check_min(row["char_count"], filters.get("min_chars")):
        return False, "too_short_chars"
    if not _check_max(row["char_count"], filters.get("max_chars")):
        return False, "too_long_chars"
    allowed = filters.get("languages")
    if allowed and row["language"] not in allowed:
        return False, "language"
    numeric = (
        ("language_score", "min_language_score"),
        ("fasttext_score", "min_fasttext_score"),
        ("score", "min_score"),
        ("int_score", "min_int_score"),
        ("token_count", "min_token_count"),
    )
    missing_policy = filters.get("missing_policy", "ignore")
    for field, threshold_key in numeric:
        threshold = filters.get(threshold_key)
        if threshold is None:
            continue
        value = row[field]
        if value is None:
            if missing_policy == "reject":
                return False, f"missing_{field}"
            continue
        if value < threshold:
            return False, threshold_key
    return True, None


class ShardWriter:
    def __init__(self, out_dir: Path, target_size_mb: int, max_documents: int):
        self.out_dir = out_dir
        self.target_bytes = target_size_mb * 1024 * 1024
        self.max_documents = max_documents
        self.rows: list[dict[str, Any]] = []
        self.approx_bytes = 0
        self.index = 0
        self.total_rows = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any]) -> Path | None:
        self.rows.append(row)
        self.approx_bytes += len(row["text"].encode("utf-8")) + 256
        if self.approx_bytes >= self.target_bytes or len(self.rows) >= self.max_documents:
            return self.flush()
        return None

    def flush(self) -> Path | None:
        if not self.rows:
            return None
        path = self.out_dir / f"part-{self.index:05d}.parquet"
        table = pa.Table.from_pylist(self.rows, schema=OUTPUT_SCHEMA)
        pq.write_table(table, path, compression="zstd", compression_level=6)
        self.total_rows += len(self.rows)
        self.rows.clear()
        self.approx_bytes = 0
        self.index += 1
        return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
