#!/usr/bin/env python3
"""Stage 4: document-interleaved, exact-budget LaughLM token corpus builder.

Consumes ONE configs/stage4/<mixture>.yaml. Each source points to the exact
Stage-3 config it consumes, giving reproducible source lineage. Documents are
interleaved across datasets according to remaining token quotas before being
written into the final memory-mappable binary token stream.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import xxhash
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table
from tokenizers import Tokenizer

from config_loader import load_mixture_bundle, stage3_semantic_hash
from pipeline_utils import (
    download_file,
    ensure_repo,
    hf_token,
    list_repo_files,
    setup_logging,
    upload_file,
    utc_now,
    write_json,
)


@dataclass
class SourcePart:
    dataset: str
    repo_id: str
    path: str


def load_tokenizer(name_or_path: str) -> Tokenizer:
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return Tokenizer.from_file(str(path))
    if path.is_dir():
        file = path / "tokenizer.json"
        if not file.is_file():
            raise FileNotFoundError(f"No tokenizer.json in {path}")
        return Tokenizer.from_file(str(file))
    return Tokenizer.from_pretrained(name_or_path)


def resolve_eos_id(tokenizer: Tokenizer, mix: Dict[str, Any]) -> int:
    if mix.get("eos_token_id") is not None:
        return int(mix["eos_token_id"])
    token = mix.get("eos_token")
    if not token:
        raise ValueError("Stage-4 config must define eos_token or eos_token_id")
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"EOS token {token!r} is not in the tokenizer")
    return int(value)


def token_dtype(tokenizer: Tokenizer, configured: str) -> np.dtype:
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if configured == "uint16":
        if vocab > 65536:
            raise ValueError(f"Tokenizer vocab {vocab} does not fit uint16")
        return np.dtype("<u2")
    if configured == "uint32":
        return np.dtype("<u4")
    if configured != "auto":
        raise ValueError("token_dtype must be auto, uint16, or uint32")
    return np.dtype("<u2" if vocab <= 65536 else "<u4")


def list_stage3_parts(source_bundle: Dict[str, Any], api: HfApi, token: str, common: Dict[str, Any]) -> List[SourcePart]:
    dataset = source_bundle["dataset"]
    repo = dataset["repos"]["stage3"]
    run_hash = stage3_semantic_hash(source_bundle)
    prefix = f"{dataset['name']}/runs/{run_hash[:12]}/sources/"
    files = list_repo_files(api, repo, token, common)
    manifests = sorted(f for f in files if f.startswith(prefix) and f.endswith("/manifest.json"))
    parts: List[SourcePart] = []
    for manifest_path in manifests:
        local = download_file(repo, manifest_path, "main", token, common)
        manifest = json.loads(local.read_text(encoding="utf-8"))
        parts.extend(SourcePart(dataset["name"], repo, p) for p in manifest["output_parts"])
    return parts


def iter_parquet_text(path: Path, batch_rows: int = 512) -> Iterator[str]:
    with pq.ParquetFile(path) as pf:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if text:
                    yield text


class DatasetTextStream:
    """Keeps at most one downloaded Stage-3 part for this dataset."""
    def __init__(self, dataset: str, parts: List[SourcePart], common: Dict[str, Any], token: str, work_dir: Path, rng: random.Random) -> None:
        self.dataset = dataset
        self.parts = list(parts)
        rng.shuffle(self.parts)
        self.common = common
        self.token = token
        self.work_dir = work_dir
        self.index = 0
        self.current_local: Optional[Path] = None
        self.current_iter: Optional[Iterator[str]] = None

    def _open_next(self) -> None:
        if self.current_local is not None:
            self.current_local.unlink(missing_ok=True)
            self.current_local = None
        if self.index >= len(self.parts):
            raise StopIteration
        part = self.parts[self.index]
        self.index += 1
        local_dir = self.work_dir / self.dataset
        local_dir.mkdir(parents=True, exist_ok=True)
        self.current_local = download_file(part.repo_id, part.path, "main", self.token, self.common, local_dir)
        self.current_iter = iter_parquet_text(self.current_local)

    def next_text(self) -> str:
        while True:
            if self.current_iter is None:
                self._open_next()
            try:
                return next(self.current_iter)
            except StopIteration:
                self.current_iter = None
                self._open_next()

    def close(self) -> None:
        if self.current_local is not None:
            self.current_local.unlink(missing_ok=True)


class BinaryShardWriter:
    def __init__(self, repo: str, remote_prefix: str, local_dir: Path, tokens_per_shard: int, dtype: np.dtype, api: HfApi, token: str, common: Dict[str, Any]) -> None:
        self.repo = repo; self.remote_prefix = remote_prefix; self.local_dir = local_dir
        self.tokens_per_shard = tokens_per_shard; self.dtype = dtype; self.api = api; self.token = token; self.common = common
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0; self.current_tokens = 0; self.total_tokens = 0
        self.current_path: Optional[Path] = None; self.current_file = None
        self.shards: List[Dict[str, Any]] = []

    def _open(self) -> None:
        if self.current_file is None:
            self.current_path = self.local_dir / f"shard_{self.index:05d}.bin"
            self.current_file = self.current_path.open("wb")
            self.current_tokens = 0

    def _close_upload(self) -> None:
        if self.current_file is None or self.current_path is None:
            return
        self.current_file.flush(); os.fsync(self.current_file.fileno()); self.current_file.close()
        size = self.current_path.stat().st_size
        expected = self.current_tokens * self.dtype.itemsize
        if size != expected:
            raise IOError(f"Shard size mismatch {size} != {expected}")
        h = xxhash.xxh3_128()
        with self.current_path.open("rb") as f:
            while chunk := f.read(8 * 1024 * 1024): h.update(chunk)
        remote = f"{self.remote_prefix}/tokens/{self.current_path.name}"
        upload_file(self.api, self.repo, self.current_path, remote, self.token, self.common)
        self.shards.append({"path": remote, "tokens": self.current_tokens, "bytes": size, "xxh3_128": h.hexdigest()})
        self.current_path.unlink(missing_ok=True)
        self.current_file = None; self.current_path = None; self.current_tokens = 0; self.index += 1

    def write(self, ids: List[int]) -> None:
        offset = 0
        while offset < len(ids):
            self._open()
            room = self.tokens_per_shard - self.current_tokens
            take = min(room, len(ids) - offset)
            arr = np.asarray(ids[offset: offset + take], dtype=self.dtype)
            self.current_file.write(arr.tobytes(order="C"))
            self.current_tokens += take; self.total_tokens += take; offset += take
            if self.current_tokens >= self.tokens_per_shard:
                self._close_upload()

    def close(self) -> None:
        if self.current_file is not None and self.current_tokens:
            self._close_upload()


def choose_dataset(rng: random.Random, remaining: Dict[str, int]) -> str:
    names = [n for n, value in remaining.items() if value > 0]
    if not names:
        raise StopIteration
    weights = [remaining[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="LaughLM Stage 4 final corpus builder")
    parser.add_argument("--config", required=True, help="configs/stage4/<mixture>.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--limit-parts-per-dataset", type=int, default=None, help="Testing only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle = load_mixture_bundle(args.config, args.common, args.registry)
    common, mix = bundle["common"], bundle["mixture"]
    setup_logging(common, "stage4")
    token = hf_token(common); api = HfApi(token=token)
    ensure_repo(api, mix["output_repo"], token, common)

    quotas = {name: int(info["spec"]["tokens"]) for name, info in bundle["sources"].items()}
    target = int(mix["target_tokens"])
    if sum(quotas.values()) != target:
        raise ValueError(f"Source quotas sum to {sum(quotas.values())}, target_tokens={target}")

    parts_by_dataset: Dict[str, List[SourcePart]] = {}
    for name, info in bundle["sources"].items():
        parts = list_stage3_parts(info["stage3_bundle"], api, token, common)
        if args.limit_parts_per_dataset is not None:
            parts = parts[: args.limit_parts_per_dataset]
        parts_by_dataset[name] = parts

    table = Table(title=f"Stage 4 — {mix['name']}")
    table.add_column("Dataset"); table.add_column("Quota"); table.add_column("Stage-3 parts")
    for name in quotas:
        table.add_row(name, f"{quotas[name]/1e9:.3f}B", str(len(parts_by_dataset[name])))
    table.add_row("TOTAL", f"{target/1e9:.3f}B", "")
    Console().print(table)
    if args.dry_run:
        return
    if any(not parts for parts in parts_by_dataset.values()):
        missing = [n for n, p in parts_by_dataset.items() if not p]
        raise RuntimeError(f"No Stage-3 parts found for: {missing}")

    tokenizer = load_tokenizer(mix["tokenizer_name_or_path"])
    eos_id = resolve_eos_id(tokenizer, mix)
    dtype = token_dtype(tokenizer, mix.get("token_dtype", "auto"))
    rng = random.Random(int(mix.get("seed", 309)))
    work = Path(common["storage"]["local_work_dir"]) / "stage4" / bundle["mixture_hash"][:12]
    streams = {
        name: DatasetTextStream(name, parts_by_dataset[name], common, token, work / "input", random.Random(rng.randrange(2**63)))
        for name in quotas
    }
    remote_prefix = f"runs/{bundle['mixture_hash'][:12]}"
    writer = BinaryShardWriter(
        mix["output_repo"], remote_prefix, work / "tokens",
        int(mix.get("tokens_per_shard", 250_000_000)), dtype, api, token, common,
    )
    remaining = dict(quotas); consumed = {n: 0 for n in quotas}
    exact = bool(mix.get("exact_budget", True)); batch_size = int(mix.get("tokenizer_batch_size", 256))

    try:
        while any(v > 0 for v in remaining.values()):
            items: List[Tuple[str, str]] = []
            for _ in range(batch_size):
                if not any(v > 0 for v in remaining.values()):
                    break
                name = choose_dataset(rng, remaining)
                try:
                    text = streams[name].next_text()
                except StopIteration as exc:
                    raise RuntimeError(f"Ran out of Stage-3 data for {name} with {remaining[name]:,} tokens still required") from exc
                items.append((name, text))
            encodings = tokenizer.encode_batch([text for _, text in items], add_special_tokens=False)
            for (name, _), enc in zip(items, encodings):
                if remaining[name] <= 0:
                    continue
                ids = list(enc.ids) + [eos_id]
                room = remaining[name]
                if len(ids) > room:
                    if exact and room > 0:
                        writer.write(ids[:room])
                        consumed[name] += room
                    remaining[name] = 0
                    continue
                writer.write(ids)
                consumed[name] += len(ids); remaining[name] -= len(ids)
        writer.close()
    finally:
        for stream in streams.values(): stream.close()
        writer.close()

    if exact and writer.total_tokens != target:
        raise RuntimeError(f"Exact target mismatch: wrote {writer.total_tokens}, expected {target}")
    manifest = {
        "version": 1,
        "stage": 4,
        "name": mix["name"],
        "mixture_hash": bundle["mixture_hash"],
        "tokenizer": mix["tokenizer_name_or_path"],
        "eos_token_id": eos_id,
        "dtype": str(dtype),
        "target_tokens": target,
        "written_tokens": writer.total_tokens,
        "source_quotas": quotas,
        "source_tokens_written": consumed,
        "source_stage3_hashes": {n: stage3_semantic_hash(info["stage3_bundle"]) for n, info in bundle["sources"].items()},
        "tokens_per_shard": int(mix.get("tokens_per_shard", 250_000_000)),
        "shards": writer.shards,
        "committed_at": utc_now(),
    }
    manifest_path = work / "corpus_manifest.json"; write_json(manifest_path, manifest)
    upload_file(api, mix["output_repo"], manifest_path, f"{remote_prefix}/corpus_manifest.json", token, common)
    # Small pointer; a new mixture hash never overwrites old token shards.
    active = work / "ACTIVE.json"; write_json(active, {"mixture_hash": bundle["mixture_hash"], "manifest": f"{remote_prefix}/corpus_manifest.json"})
    upload_file(api, mix["output_repo"], active, "ACTIVE.json", token, common)
    Console().print(f"[green]Stage 4 complete[/green]: {writer.total_tokens:,} tokens, {len(writer.shards)} shards")


if __name__ == "__main__":
    main()
