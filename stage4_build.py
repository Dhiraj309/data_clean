#!/usr/bin/env python3
"""Stage 4: document-interleaved, exact-budget LaughLM token corpus builder.

Consumes ONE configs/stage4/<mixture>.yaml. Each source points to the exact
Stage-3 config it consumes, giving reproducible source lineage. Documents are
interleaved across datasets according to remaining token quotas before being
written into the final memory-mappable binary token stream.
"""
from __future__ import annotations

import argparse
import hashlib
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

from config_loader import load_mixture_bundle, stable_hash, stage3_semantic_hash
from manifest_contract import build_artifact_contract, validate_committed_manifest
from pipeline_utils import (
    download_file,
    ensure_repo,
    hf_token,
    list_repo_files,
    read_remote_json,
    setup_logging,
    upload_file,
    local_work_root,
    utc_now,
    write_json,
)


@dataclass
class SourcePart:
    dataset: str
    repo_id: str
    path: str


def tokenizer_contract(tokenizer: Tokenizer, mix: Dict[str, Any], dtype: np.dtype, eos_id: int) -> Dict[str, Any]:
    payload = tokenizer.to_str().encode("utf-8")
    return {
        "format": "huggingface_tokenizers_json",
        "tokenizer_name_or_path": mix["tokenizer_name_or_path"],
        "tokenizer_hash": hashlib.sha256(payload).hexdigest(),
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "eos_token_id": eos_id,
        "dtype": str(dtype),
    }


def packing_contract(mix: Dict[str, Any]) -> Dict[str, Any]:
    configured = dict(mix.get("packing") or {})
    return {
        "mode": str(configured.get("mode", "document_eos")),
        "sequence_length": int(configured.get("sequence_length", 2048)),
        "eos_between_documents": bool(configured.get("eos_between_documents", True)),
        "pad_to_multiple": configured.get("pad_to_multiple"),
    }


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
    if configured == "uint64":
        return np.dtype("<u8")
    if configured != "auto":
        raise ValueError("token_dtype must be auto, uint16, or uint64")
    return np.dtype("<u2" if vocab <= 65536 else "<u8")


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
        if validate_committed_manifest(manifest, expected_stage="stage3", available_files=files):
            continue
        parts.extend(SourcePart(dataset["name"], repo, p) for p in manifest["output_parts"])
    return parts


def iter_parquet_text(path: Path, batch_rows: int = 512, allowed_splits: List[str] | None = None) -> Iterator[str]:
    with pq.ParquetFile(path) as pf:
        names = set(pf.schema_arrow.names)
        has_split = "split" in names
        columns = ["text", "split"] if has_split and allowed_splits is not None else ["text"]
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            texts = batch.column("text").to_pylist()
            splits = batch.column("split").to_pylist() if has_split and len(columns) == 2 else None
            for index, text in enumerate(texts):
                if text and (splits is None or splits[index] in allowed_splits):
                    yield text


class DatasetTextStream:
    """Keeps at most one downloaded Stage-3 part for this dataset."""
    def __init__(self, dataset: str, parts: List[SourcePart], common: Dict[str, Any], token: str, work_dir: Path, rng: random.Random, allowed_splits: List[str]) -> None:
        self.dataset = dataset
        self.parts = list(parts)
        rng.shuffle(self.parts)
        self.common = common
        self.token = token
        self.work_dir = work_dir
        self.index = 0
        self.current_local: Optional[Path] = None
        self.current_iter: Optional[Iterator[str]] = None
        self.allowed_splits = allowed_splits

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
        self.current_iter = iter_parquet_text(self.current_local, allowed_splits=self.allowed_splits)

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
    def __init__(self, repo: str, remote_prefix: str, local_dir: Path, tokens_per_shard: int, dtype: np.dtype, api: HfApi, token: str, common: Dict[str, Any], start_index: int = 0, initial_total_tokens: int = 0, initial_shards: List[Dict[str, Any]] | None = None, on_shard_committed=None) -> None:
        self.repo = repo; self.remote_prefix = remote_prefix; self.local_dir = local_dir
        self.tokens_per_shard = tokens_per_shard; self.dtype = dtype; self.api = api; self.token = token; self.common = common
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.index = start_index; self.current_tokens = 0; self.total_tokens = initial_total_tokens
        self.current_path: Optional[Path] = None; self.current_file = None
        self.shards: List[Dict[str, Any]] = list(initial_shards or [])
        self.on_shard_committed = on_shard_committed

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
        self.shards.append({
            "path": remote,
            "tokens": self.current_tokens,
            "bytes": size,
            "algorithm": "xxh3_128",
            "digest": h.hexdigest(),
            "xxh3_128": h.hexdigest(),
        })
        if self.on_shard_committed is not None:
            self.on_shard_committed(self)
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


def quota_labels(info: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return the source-level labels used by optional mixture quotas."""
    spec = info["spec"]
    domain = spec.get("domain")
    time_bucket = spec.get("time_bucket")
    if time_bucket is None:
        time_range = spec.get("time_range")
        if isinstance(time_range, dict):
            time_bucket = time_range.get("label")
        elif time_range is not None:
            time_bucket = str(time_range)
    return (str(domain) if domain is not None else None,
            str(time_bucket) if time_bucket is not None else None)


def quota_map(name: str, values: Any) -> Dict[str, int]:
    """Normalize and validate an optional exact dimension quota map."""
    if not values:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a mapping of label to non-negative token count")
    normalized = {str(label): int(tokens) for label, tokens in values.items()}
    if any(tokens < 0 for tokens in normalized.values()):
        raise ValueError(f"{name} cannot contain negative token counts")
    return normalized


def validate_dimension_quotas(
    dimension: str,
    quotas: Dict[str, int],
    sources: Dict[str, Dict[str, Any]],
    source_quotas: Dict[str, int],
) -> None:
    """Fail before processing when a configured dimension cannot be satisfied."""
    if not quotas:
        return
    if sum(quotas.values()) != sum(source_quotas.values()):
        raise ValueError(
            f"{dimension}_quotas sum to {sum(quotas.values())}, "
            f"but source quotas require {sum(source_quotas.values())}"
        )
    capacity: Dict[str, int] = {}
    label_index = 0 if dimension == "domain" else 1
    for name, info in sources.items():
        labels = quota_labels(info)
        label = labels[label_index]
        if label is None and source_quotas[name] > 0:
            raise ValueError(
                f"Source {name!r} has no {dimension} label but its quota is positive"
            )
        if label is not None:
            capacity[label] = capacity.get(label, 0) + source_quotas[name]
    if capacity != quotas:
        raise ValueError(
            f"{dimension}_quotas do not match source metadata: "
            f"declared={quotas}, available={capacity}"
        )


def choose_dataset(
    rng: random.Random,
    remaining: Dict[str, int],
    sources: Dict[str, Dict[str, Any]],
    domain_remaining: Dict[str, int],
    time_remaining: Dict[str, int],
) -> str:
    names = []
    for name, value in remaining.items():
        if value <= 0:
            continue
        domain, time_bucket = quota_labels(sources[name])
        if domain_remaining and (domain is None or domain_remaining.get(domain, 0) <= 0):
            continue
        if time_remaining and (time_bucket is None or time_remaining.get(time_bucket, 0) <= 0):
            continue
        names.append(name)
    if not names:
        raise RuntimeError(
            "No source satisfies the remaining source/domain/time quotas; "
            "check Stage-4 source labels and quota maps"
        )
    weights = [remaining[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="LaughLM Stage 4 final corpus builder")
    parser.add_argument("--config", required=True, help="configs/stage4/<mixture>.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--limit-parts-per-dataset", type=int, default=None, help="Testing only")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing progress marker")
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
    domain_quotas = quota_map("domain_quotas", mix.get("domain_quotas"))
    time_quotas = quota_map("time_quotas", mix.get("time_quotas"))
    validate_dimension_quotas("domain", domain_quotas, bundle["sources"], quotas)
    validate_dimension_quotas("time", time_quotas, bundle["sources"], quotas)

    parts_by_dataset: Dict[str, List[SourcePart]] = {}
    allowed_splits_by_dataset: Dict[str, List[str]] = {}
    for name, info in bundle["sources"].items():
        parts = list_stage3_parts(info["stage3_bundle"], api, token, common)
        if args.limit_parts_per_dataset is not None:
            parts = parts[: args.limit_parts_per_dataset]
        parts_by_dataset[name] = parts
        allowed_splits_by_dataset[name] = list(info["spec"].get("allowed_splits", ["train"]))

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
    tokenizer_contract_data = tokenizer_contract(tokenizer, mix, dtype, eos_id)
    packing_contract_data = packing_contract(mix)
    contract = {
        "mixture_hash": bundle["mixture_hash"],
        "tokenizer": tokenizer_contract_data,
        "packing": packing_contract_data,
        "tokens_per_shard": int(mix.get("tokens_per_shard", 250_000_000)),
        "source_quotas": quotas,
        "domain_quotas": domain_quotas,
        "time_quotas": time_quotas,
        "allowed_splits_by_source": allowed_splits_by_dataset,
    }
    contract_hash = stable_hash(contract)
    rng = random.Random(int(mix.get("seed", 309)))
    work = local_work_root(common) / "stage4" / bundle["mixture_hash"][:12]
    remote_prefix = f"runs/{bundle['mixture_hash'][:12]}"
    progress_remote = f"{remote_prefix}/progress.json"
    corpus_remote = f"{remote_prefix}/corpus_manifest.json"
    existing_output_files = set(list_repo_files(api, mix["output_repo"], token, common))
    progress = None
    if not args.fresh and corpus_remote in existing_output_files:
        existing_manifest = read_remote_json(mix["output_repo"], corpus_remote, token, common)
        if existing_manifest.get("contract_hash") == contract_hash and existing_manifest.get("processing_status") == "committed":
            Console().print(f"[green]Stage 4 already committed[/green]: {corpus_remote}")
            return
        raise ValueError("Existing Stage-4 corpus manifest has a different tokenizer/packing contract")
    if not args.fresh and progress_remote in existing_output_files:
        progress = read_remote_json(mix["output_repo"], progress_remote, token, common)
        if progress.get("contract_hash") != contract_hash:
            raise ValueError("Existing Stage-4 progress marker has a different tokenizer/packing contract")
    streams = {
        name: DatasetTextStream(
            name,
            parts_by_dataset[name],
            common,
            token,
            work / "input",
            random.Random(rng.randrange(2**63)),
            allowed_splits_by_dataset[name],
        )
        for name in quotas
    }
    resume_tokens = int((progress or {}).get("written_tokens", 0))
    completed_shards = int((progress or {}).get("completed_shards", 0))
    initial_shards = list((progress or {}).get("shards", []))
    resume_complete = resume_tokens >= target
    remaining = {} if resume_complete else dict(quotas)
    consumed = dict((progress or {}).get("source_tokens_written", {})) if resume_complete else {n: 0 for n in quotas}
    domain_remaining = {} if resume_complete else dict(domain_quotas)
    time_remaining = {} if resume_complete else dict(time_quotas)
    consumed_domain = (
        dict((progress or {}).get("domain_tokens_written", {}))
        if resume_complete else {label: 0 for label in domain_quotas}
    )
    consumed_time = (
        dict((progress or {}).get("time_tokens_written", {}))
        if resume_complete else {label: 0 for label in time_quotas}
    )
    progress_local = work / "progress.json"

    def on_shard_committed(current_writer: BinaryShardWriter) -> None:
        write_json(progress_local, {
            "format": "laughlm_stage4_progress_v1",
            "processing_status": "in_progress",
            "contract_hash": contract_hash,
            "mixture_hash": bundle["mixture_hash"],
            "completed_shards": len(current_writer.shards),
            "written_tokens": current_writer.total_tokens,
            "source_tokens_written": consumed,
            "domain_tokens_written": consumed_domain,
            "time_tokens_written": consumed_time,
            "shards": current_writer.shards,
            "updated_at": utc_now(),
        })
        upload_file(api, mix["output_repo"], progress_local, progress_remote, token, common)

    writer = BinaryShardWriter(
        mix["output_repo"], remote_prefix, work / "tokens",
        int(mix.get("tokens_per_shard", 250_000_000)), dtype, api, token, common,
        start_index=completed_shards,
        initial_total_tokens=resume_tokens,
        initial_shards=initial_shards,
        on_shard_committed=on_shard_committed,
    )
    exact = bool(mix.get("exact_budget", True)); batch_size = int(mix.get("tokenizer_batch_size", 256))
    skip_tokens = resume_tokens

    def emit(ids: List[int]) -> None:
        nonlocal skip_tokens
        if skip_tokens:
            skipped = min(skip_tokens, len(ids))
            skip_tokens -= skipped
            ids = ids[skipped:]
        if ids:
            writer.write(ids)

    try:
        while remaining and any(v > 0 for v in remaining.values()):
            items: List[Tuple[str, str]] = []
            for _ in range(batch_size):
                if not any(v > 0 for v in remaining.values()):
                    break
                name = choose_dataset(
                    rng, remaining, bundle["sources"], domain_remaining, time_remaining
                )
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
                domain, time_bucket = quota_labels(bundle["sources"][name])
                rooms = [remaining[name]]
                if domain_remaining:
                    rooms.append(domain_remaining[domain])
                if time_remaining:
                    rooms.append(time_remaining[time_bucket])
                room = min(rooms)
                if room <= 0:
                    continue
                take = min(len(ids), room)
                consumed[name] += take
                remaining[name] -= take
                if domain_remaining:
                    domain_remaining[domain] -= take
                    consumed_domain[domain] += take
                if time_remaining:
                    time_remaining[time_bucket] -= take
                    consumed_time[time_bucket] += take
                emit(ids[:take])
        writer.close()
    finally:
        for stream in streams.values(): stream.close()
        writer.close()

    if exact and writer.total_tokens != target:
        raise RuntimeError(f"Exact target mismatch: wrote {writer.total_tokens}, expected {target}")
    source_stage3_hashes = {
        name: stage3_semantic_hash(info["stage3_bundle"])
        for name, info in bundle["sources"].items()
    }
    source_exposure = {
        name: {
            "domain": quota_labels(info)[0],
            "time_range": info["spec"].get("time_range"),
            "time_bucket": quota_labels(info)[1],
            "quota_tokens": quotas[name],
            "tokens_written": consumed[name],
            "exposure_fraction": consumed[name] / target if target else 0.0,
            "allowed_splits": allowed_splits_by_dataset[name],
        }
        for name, info in bundle["sources"].items()
    }
    domain_exposure = {
        label: {
            "quota_tokens": domain_quotas[label],
            "tokens_written": consumed_domain.get(label, 0),
            "exposure_fraction": consumed_domain.get(label, 0) / target if target else 0.0,
        }
        for label in domain_quotas
    }
    time_exposure = {
        label: {
            "quota_tokens": time_quotas[label],
            "tokens_written": consumed_time.get(label, 0),
            "exposure_fraction": consumed_time.get(label, 0) / target if target else 0.0,
        }
        for label in time_quotas
    }
    manifest = {
        "artifact_contract": build_artifact_contract(
            artifact_type="dataset_stage",
            stage="stage4",
            dataset_id=mix["name"],
            run_id=bundle["mixture_hash"],
            config_hash=bundle["mixture_hash"],
            source_refs=[
                {"dataset_id": name, "stage": "stage3", "run_id": run_hash}
                for name, run_hash in source_stage3_hashes.items()
            ],
            attributes={
                "output_repo": mix["output_repo"],
                "tokenizer": mix["tokenizer_name_or_path"],
            },
        ),
        "version": 1,
        "stage": 4,
        "name": mix["name"],
        "mixture_hash": bundle["mixture_hash"],
        "tokenizer": mix["tokenizer_name_or_path"],
        "tokenizer_contract": tokenizer_contract_data,
        "packing_contract": packing_contract_data,
        "contract_hash": contract_hash,
        "eos_token_id": eos_id,
        "dtype": str(dtype),
        "target_tokens": target,
        "written_tokens": writer.total_tokens,
        "source_quotas": quotas,
        "source_tokens_written": consumed,
        "domain_quotas": domain_quotas,
        "domain_tokens_written": consumed_domain,
        "time_quotas": time_quotas,
        "time_tokens_written": consumed_time,
        "source_exposure": source_exposure,
        "domain_exposure": domain_exposure,
        "time_exposure": time_exposure,
        "source_stage3_hashes": source_stage3_hashes,
        "allowed_splits_by_source": allowed_splits_by_dataset,
        "split_policy_version": (common.get("splits") or {}).get("version", 1),
        "processing_status": "committed",
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
