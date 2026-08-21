#!/usr/bin/env python3
"""Stage 2: dataset-specific content safeguards + crash-safe global exact dedup.

Consumes ONE Stage-2 YAML and the exact Stage-1 run referenced by stage1_config.
All Stage-2 configs participating in one corpus must share dedup_namespace.
The namespace is an explicit corpus-version boundary: bump it if you materially
change Stage-2 recipes and want a fresh cross-dataset dedup universe.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
import datatrove.pipeline.extractors as dt_extractors
import datatrove.pipeline.filters as dt_filters
import datatrove.pipeline.formatters as dt_formatters

import filters as custom_filters
from document_identity import DocumentIdentity, document_identity, hamming_distance
from config_loader import (
    load_stage_bundle,
    resolve_relative,
    stage1_semantic_hash,
    stage2_semantic_hash,
)
from manifest_contract import build_artifact_contract, validate_committed_manifest
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

SQLITE_QUERY_CHUNK = 500
INTERNAL_METADATA_FIELDS = ("dedup_group", "near_duplicate_fingerprint", "source_priority", "split", "split_group")


def source_key(stage1_manifest: Dict[str, Any], run_hash: str) -> str:
    return xxhash.xxh3_128_hexdigest(
        f"{stage1_manifest['source_key']}\0{run_hash}".encode("utf-8")
    )


def remote_prefix(dataset_name: str, namespace: str, run_hash: str, key: str) -> str:
    return f"{dataset_name}/dedup/{slug(namespace)}/runs/{run_hash[:12]}/sources/{key}"


class CommittedHashStore:
    """Local accelerator reconstructed from durable Stage-2 hash sidecars."""
    def __init__(self, path: Path, namespace: str, algorithm: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")
        self.conn.execute("CREATE TABLE IF NOT EXISTS hashes (h BLOB PRIMARY KEY)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS normalized_hashes (h BLOB PRIMARY KEY)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS near_hashes (fingerprint TEXT PRIMARY KEY, bucket INTEGER NOT NULL)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sources (source_key TEXT PRIMARY KEY, manifest_path TEXT NOT NULL, committed_at TEXT NOT NULL)"
        )
        self.conn.execute("CREATE TABLE IF NOT EXISTS metadata (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        prev = self.conn.execute("SELECT v FROM metadata WHERE k='namespace'").fetchone()
        prev_alg = self.conn.execute("SELECT v FROM metadata WHERE k='algorithm'").fetchone()
        if (prev and prev[0] != namespace) or (prev_alg and prev_alg[0] != algorithm):
            self.conn.execute("DELETE FROM hashes")
            self.conn.execute("DELETE FROM normalized_hashes")
            self.conn.execute("DELETE FROM near_hashes")
            self.conn.execute("DELETE FROM sources")
        self.conn.execute("INSERT OR REPLACE INTO metadata(k,v) VALUES('namespace',?)", (namespace,))
        self.conn.execute("INSERT OR REPLACE INTO metadata(k,v) VALUES('algorithm',?)", (algorithm,))

    def source_is_committed(self, key: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sources WHERE source_key=?", (key,)).fetchone() is not None

    def existing(self, digests: Sequence[bytes]) -> set[bytes]:
        out: set[bytes] = set()
        for i in range(0, len(digests), SQLITE_QUERY_CHUNK):
            chunk = list(digests[i : i + SQLITE_QUERY_CHUNK])
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(f"SELECT h FROM hashes WHERE h IN ({placeholders})", chunk).fetchall()
            out.update(r[0] for r in rows)
        return out

    def existing_normalized(self, digests: Sequence[bytes]) -> set[bytes]:
        out: set[bytes] = set()
        for i in range(0, len(digests), SQLITE_QUERY_CHUNK):
            chunk = list(digests[i : i + SQLITE_QUERY_CHUNK])
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT h FROM normalized_hashes WHERE h IN ({placeholders})", chunk
            ).fetchall()
            out.update(r[0] for r in rows)
        return out

    def near_duplicate(self, fingerprint: int, bucket_bits: int, max_distance: int) -> bool:
        if not 1 <= bucket_bits <= 64:
            raise ValueError("near-duplicate bucket_bits must be between 1 and 64")
        bucket = fingerprint >> (64 - bucket_bits)
        rows = self.conn.execute(
            "SELECT fingerprint FROM near_hashes WHERE bucket=?", (bucket,)
        ).fetchall()
        return any(
            hamming_distance(fingerprint, int(value, 16)) <= max_distance
            for (value,) in rows
        )

    def commit_sidecar(self, key: str, manifest_path: str, sidecar: Path, digest_size: int) -> int:
        return self.commit_sidecars(key, manifest_path, sidecar, None, None, digest_size)

    def commit_sidecars(
        self,
        key: str,
        manifest_path: str,
        sidecar: Path,
        normalized_sidecar: Path | None,
        near_sidecar: Path | None,
        digest_size: int,
        near_bucket_bits: int = 8,
    ) -> int:
        if self.source_is_committed(key):
            return 0
        if sidecar.stat().st_size % digest_size:
            raise IOError(f"Corrupt hash sidecar: {sidecar}")
        if normalized_sidecar is not None and normalized_sidecar.stat().st_size % digest_size:
            raise IOError(f"Corrupt normalized hash sidecar: {normalized_sidecar}")
        if near_sidecar is not None and near_sidecar.stat().st_size % 8:
            raise IOError(f"Corrupt near-duplicate sidecar: {near_sidecar}")
        inserted = 0
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            with sidecar.open("rb") as f:
                batch: List[tuple[bytes]] = []
                while True:
                    d = f.read(digest_size)
                    if not d:
                        break
                    batch.append((d,))
                    if len(batch) >= 10000:
                        before = self.conn.total_changes
                        self.conn.executemany("INSERT OR IGNORE INTO hashes(h) VALUES(?)", batch)
                        inserted += self.conn.total_changes - before
                        batch.clear()
                if batch:
                    before = self.conn.total_changes
                    self.conn.executemany("INSERT OR IGNORE INTO hashes(h) VALUES(?)", batch)
                    inserted += self.conn.total_changes - before
            if normalized_sidecar is not None:
                with normalized_sidecar.open("rb") as f:
                    batch = []
                    while True:
                        d = f.read(digest_size)
                        if not d:
                            break
                        batch.append((d,))
                        if len(batch) >= 10000:
                            self.conn.executemany("INSERT OR IGNORE INTO normalized_hashes(h) VALUES(?)", batch)
                            batch.clear()
                    if batch:
                        self.conn.executemany("INSERT OR IGNORE INTO normalized_hashes(h) VALUES(?)", batch)
            if near_sidecar is not None:
                with near_sidecar.open("rb") as f:
                    while True:
                        raw = f.read(8)
                        if not raw:
                            break
                        fingerprint = int.from_bytes(raw, "big")
                        bucket = fingerprint >> (64 - near_bucket_bits)
                        self.conn.execute(
                            "INSERT OR IGNORE INTO near_hashes(fingerprint,bucket) VALUES(?,?)",
                            (f"{fingerprint:016x}", bucket),
                        )
            self.conn.execute(
                "INSERT INTO sources(source_key,manifest_path,committed_at) VALUES(?,?,?)",
                (key, manifest_path, utc_now()),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return inserted

    def close(self) -> None:
        self.conn.close()


class RegistryAdapterStep(PipelineStep):
    name = "LaughLM custom filters"
    def __init__(self, steps: List[Dict[str, Any]], metrics: Dict[str, int]) -> None:
        super().__init__()
        self.steps = steps
        self.metrics = metrics
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        del rank, world_size
        for doc in data:
            cleaned, reason = custom_filters.run_pipeline(doc.text, self.steps)
            if cleaned is None:
                self.metrics["custom_rejected"] += 1
                reason_key = f"custom_filter:{reason or 'rejected'}"
                self.metrics["rejected_by_reason"][reason_key] = (
                    self.metrics["rejected_by_reason"].get(reason_key, 0) + 1
                )
                continue
            doc.text = cleaned
            yield doc


class CrashSafeDedupStep(PipelineStep):
    name = "Crash-safe exact and normalized dedup"
    def __init__(
        self,
        store: CommittedHashStore,
        algorithm: str,
        batch_size: int,
        sidecar: Path,
        normalized_sidecar: Path,
        near_sidecar: Path | None,
        metrics: Dict[str, Any],
        identity_policy: Dict[str, Any],
        near_policy: Dict[str, Any],
        source_priority: int,
    ) -> None:
        super().__init__()
        self.store = store
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.sidecar = sidecar
        self.normalized_sidecar = normalized_sidecar
        self.near_sidecar = near_sidecar
        self.metrics = metrics
        self.identity_policy = identity_policy
        self.near_policy = near_policy
        self.near_enabled = bool(near_policy.get("enabled", False)) and near_sidecar is not None
        if not 1 <= int(near_policy.get("bucket_bits", 8)) <= 64:
            raise ValueError("near-duplicate bucket_bits must be between 1 and 64")
        self.source_priority = source_priority
        self.seen_current: set[bytes] = set()
        self.seen_normalized_current: set[bytes] = set()
        self.seen_near_current: Dict[int, List[int]] = {}
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        normalized_sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(b"")
        normalized_sidecar.write_bytes(b"")
        if near_sidecar is not None:
            near_sidecar.parent.mkdir(parents=True, exist_ok=True)
            near_sidecar.write_bytes(b"")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        del rank, world_size
        docs: List[Document] = []
        identities: List[DocumentIdentity] = []
        with self.sidecar.open("ab") as side, self.normalized_sidecar.open("ab") as normalized_side:
            near_handle = self.near_sidecar.open("ab") if self.near_sidecar is not None else None
            def flush() -> Iterator[Document]:
                if not docs:
                    return iter(())
                exact = [identity.exact_digest for identity in identities]
                normalized = [identity.normalized_digest for identity in identities]
                existing = self.store.existing(exact)
                existing_normalized = self.store.existing_normalized(normalized)
                kept: List[Document] = []
                for doc, identity in zip(docs, identities):
                    duplicate_reason = None
                    if identity.exact_digest in existing or identity.exact_digest in self.seen_current:
                        duplicate_reason = "exact_text_hash"
                    elif identity.normalized_digest in existing_normalized or identity.normalized_digest in self.seen_normalized_current:
                        duplicate_reason = "normalized_text_hash"
                    elif self.near_enabled:
                        bucket_bits = int(self.near_policy.get("bucket_bits", 8))
                        max_distance = int(self.near_policy.get("max_hamming_distance", 3))
                        bucket = identity.near_fingerprint >> (64 - bucket_bits)
                        current_near = self.seen_near_current.get(bucket, [])
                        if self.store.near_duplicate(identity.near_fingerprint, bucket_bits, max_distance) or any(
                            hamming_distance(identity.near_fingerprint, value) <= max_distance
                            for value in current_near
                        ):
                            duplicate_reason = "near_duplicate_simhash"
                    if duplicate_reason is not None:
                        self.metrics["duplicates"] += 1
                        if duplicate_reason == "near_duplicate_simhash":
                            self.metrics["near_duplicates"] += 1
                        self.metrics["duplicate_by_reason"][duplicate_reason] = (
                            self.metrics["duplicate_by_reason"].get(duplicate_reason, 0) + 1
                        )
                        continue
                    self.seen_current.add(identity.exact_digest)
                    self.seen_normalized_current.add(identity.normalized_digest)
                    side.write(identity.exact_digest)
                    normalized_side.write(identity.normalized_digest)
                    if self.near_enabled and near_handle is not None:
                        near_handle.write(identity.near_fingerprint.to_bytes(8, "big"))
                        bucket_bits = int(self.near_policy.get("bucket_bits", 8))
                        bucket = identity.near_fingerprint >> (64 - bucket_bits)
                        self.seen_near_current.setdefault(bucket, []).append(identity.near_fingerprint)
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["dedup_group"] = identity.normalized_digest.hex()
                    doc.metadata["near_duplicate_fingerprint"] = f"{identity.near_fingerprint:016x}"
                    doc.metadata["source_priority"] = self.source_priority
                    kept.append(doc)
                docs.clear(); identities.clear()
                return iter(kept)
            for doc in data:
                docs.append(doc)
                identities.append(document_identity(doc.text, self.algorithm, self.identity_policy))
                if len(docs) >= self.batch_size:
                    yield from flush()
            yield from flush()
            if near_handle is not None:
                near_handle.close()


CrashSafeExactDedupStep = CrashSafeDedupStep


class LocalParquetSink(PipelineStep):
    name = "Local parquet sink"
    def __init__(self, output_dir: Path, metadata_fields: List[str], shard_cfg: Dict[str, Any], compression: Dict[str, Any], metrics: Dict[str, int]) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.metadata_fields = metadata_fields
        self.target_bytes = int(shard_cfg["target_size_mb"] * 1024 * 1024)
        self.max_docs = int(shard_cfg["max_documents"])
        self.compression = compression
        self.metrics = metrics
        self.rows: List[Dict[str, Any]] = []
        self.bytes = 0
        self.index = 0
        self.paths: List[Path] = []
        output_dir.mkdir(parents=True, exist_ok=True)

    def _flush(self) -> None:
        if not self.rows:
            return
        path = self.output_dir / f"part-{self.index:05d}.parquet"
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, path, compression=self.compression["codec"], compression_level=self.compression.get("level"))
        self.paths.append(path)
        self.index += 1
        self.rows.clear(); self.bytes = 0

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        del rank, world_size
        for doc in data:
            row = {"id": doc.id, "text": doc.text}
            for field in dict.fromkeys([*self.metadata_fields, *INTERNAL_METADATA_FIELDS]):
                row[field] = doc.metadata.get(field)
            self.rows.append(row)
            self.bytes += len(doc.text.encode("utf-8"))
            self.metrics["accepted"] += 1
            if self.bytes >= self.target_bytes or len(self.rows) >= self.max_docs:
                self._flush()
            yield doc
        self._flush()


def run_pipeline_inline(steps: List[Any]) -> None:
    data = None
    for step in steps:
        if hasattr(step, "run"):
            data = step.run(data)
        elif callable(step):
            data = step(data)
        else:
            raise TypeError(f"Unsupported pipeline step: {step!r}")
    if data is not None:
        for _ in data:
            pass


def build_cleaning(stage: Dict[str, Any]) -> List[Any]:
    cfg = stage.get("cleaning") or {}
    out: List[Any] = []
    if cfg.get("use_trafilatura"):
        out.append(dt_extractors.Trafilatura(**cfg.get("trafilatura_kwargs", {})))
    if cfg.get("use_ftfy"):
        out.append(dt_formatters.FTFYFormatter(**cfg.get("ftfy_kwargs", {})))
    symbols = cfg.get("symbol_lines_to_remove") or []
    if symbols:
        out.append(dt_formatters.SymbolLinesFormatter(symbols_to_remove=symbols))
    return out


def build_quality(stage: Dict[str, Any]) -> List[Any]:
    out = []
    for spec in stage.get("datatrove_quality_filters") or []:
        cls = getattr(dt_filters, spec["type"])
        out.append(cls(**spec.get("kwargs", {})))
    return out


def build_post(stage: Dict[str, Any]) -> List[Any]:
    pii = stage.get("pii") or {}
    return [dt_formatters.PIIFormatter(**pii.get("kwargs", {}))] if pii.get("enabled") else []


def load_stage1_bundle_for_stage2(bundle: Dict[str, Any], common_path=None, registry_path=None) -> Dict[str, Any]:
    path = resolve_relative(bundle["stage"]["stage1_config"], bundle["stage_path"])
    return load_stage_bundle(path, 1, common_path, registry_path)


def list_stage1_manifests(bundle: Dict[str, Any], s1_bundle: Dict[str, Any], api: HfApi, token: str) -> List[str]:
    common, dataset = bundle["common"], bundle["dataset"]
    repo = dataset["repos"]["stage1"]
    run_hash = stage1_semantic_hash(s1_bundle)
    prefix = f"{dataset['name']}/runs/{run_hash[:12]}/sources/"
    return sorted(f for f in list_repo_files(api, repo, token, common) if f.startswith(prefix) and f.endswith("/manifest.json"))


def reconcile_namespace(bundle: Dict[str, Any], api: HfApi, token: str, store: CommittedHashStore) -> None:
    common, registry, stage = bundle["common"], bundle["registry"], bundle["stage"]
    namespace = slug(stage["dedup_namespace"])
    # Rebuild the namespace in source-priority order so the configured winner
    # policy is independent of HF listing order and the order of stage runs.
    registry_items = sorted(
        registry.items(),
        key=lambda item: (-int(item[1].get("source_priority", 0)), item[0]),
    )
    for name, entry in registry_items:
        repo = entry.get("repos", {}).get("stage2")
        if not repo:
            continue
        try:
            files = list_repo_files(api, repo, token, common)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("stage2.reconcile").warning("Could not inspect %s: %s", repo, exc)
            continue
        prefix = f"{name}/dedup/{namespace}/"
        manifest_paths = sorted(
            f for f in files if f.startswith(prefix) and f.endswith("/manifest.json")
        )
        for manifest_path in manifest_paths:
            manifest = read_remote_json(repo, manifest_path, token, common)
            key = manifest["source_key"]
            if store.source_is_committed(key):
                continue
            normalized_path = manifest.get("normalized_hash_sidecar")
            if not normalized_path or normalized_path not in files:
                logging.getLogger("stage2.reconcile").warning(
                    "Skipping legacy manifest without normalized hash sidecar: %s", manifest_path
                )
                continue
            side = download_file(repo, manifest["hash_sidecar"], "main", token, common)
            normalized_side = download_file(repo, normalized_path, "main", token, common)
            near_side = None
            if manifest.get("near_duplicate_sidecar") in files:
                near_side = download_file(repo, manifest["near_duplicate_sidecar"], "main", token, common)
            store.commit_sidecars(
                key,
                manifest_path,
                side,
                normalized_side,
                near_side,
                int(manifest["hash_digest_size"]),
                int((manifest.get("near_duplicate_policy") or {}).get("bucket_bits", 8)),
            )


def _process_source(bundle: Dict[str, Any], s1_manifest_path: str, api: HfApi, token: str, store: CommittedHashStore) -> Dict[str, Any]:
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    s1_repo, s2_repo = dataset["repos"]["stage1"], dataset["repos"]["stage2"]
    s1_manifest = read_remote_json(s1_repo, s1_manifest_path, token, common)
    run_hash = stage2_semantic_hash(bundle)
    key = source_key(s1_manifest, run_hash)
    prefix = remote_prefix(dataset["name"], stage["dedup_namespace"], run_hash, key)
    manifest_remote = f"{prefix}/manifest.json"

    remote_files = set(list_repo_files(api, s2_repo, token, common))
    if manifest_remote in remote_files:
        try:
            manifest = read_remote_json(s2_repo, manifest_remote, token, common)
            errors = validate_committed_manifest(
                manifest, expected_stage="stage2", available_files=remote_files
            )
            normalized_path = manifest.get("normalized_hash_sidecar")
            near_path = manifest.get("near_duplicate_sidecar")
            near_required = bool((common.get("deduplication") or {}).get("near_duplicate", {}).get("enabled", False))
            if (
                not errors
                and manifest.get("hash_sidecar") in remote_files
                and normalized_path in remote_files
                and (not near_required or near_path in remote_files)
            ):
                if not store.source_is_committed(key):
                    side = download_file(s2_repo, manifest["hash_sidecar"], "main", token, common)
                    normalized_side = download_file(s2_repo, normalized_path, "main", token, common)
                    near_side = download_file(s2_repo, near_path, "main", token, common) if near_required else None
                    store.commit_sidecars(
                        key, manifest_remote, side, normalized_side, near_side,
                        int(manifest["hash_digest_size"]),
                        int((manifest.get("near_duplicate_policy") or {}).get("bucket_bits", 8)),
                    )
                return {"source": s1_manifest["source_file"], "status": "skipped"}
            if manifest.get("hash_sidecar") not in remote_files:
                errors.append("hash_sidecar is missing")
            if normalized_path not in remote_files:
                errors.append("normalized_hash_sidecar is missing")
            if near_required and near_path not in remote_files:
                errors.append("near_duplicate_sidecar is missing")
            logging.getLogger("stage2").warning(
                "Ignoring incomplete manifest %s: %s", manifest_remote, "; ".join(errors)
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("stage2").warning("Ignoring unreadable manifest %s: %s", manifest_remote, exc)

    work = local_work_root(common) / dataset["name"] / "stage2" / key
    shutil.rmtree(work, ignore_errors=True)
    inputs = work / "input"; inputs.mkdir(parents=True, exist_ok=True)
    sidecar = work / "accepted.hashes"
    normalized_sidecar = work / "accepted.normalized.hashes"
    near_policy = (common.get("deduplication") or {}).get("near_duplicate", {})
    near_sidecar = work / "accepted.near.hashes" if near_policy.get("enabled", False) else None
    identity_policy = (common.get("deduplication") or {}).get("normalized", {})
    source_priority = int(dataset.get("source_priority", 0))
    metrics = {
        "seen": 0,
        "accepted": 0,
        "duplicates": 0,
        "custom_rejected": 0,
        "near_duplicates": 0,
        "rejected_by_reason": {},
        "duplicate_by_reason": {},
    }
    sink = LocalParquetSink(work / "parts", stage.get("output_metadata_fields", []), stage["shard"], common["compression"], metrics)
    dedup = CrashSafeDedupStep(
        store,
        common["hashing"]["algorithm"],
        int(common["hashing"].get("dedup_batch_size", 500)),
        sidecar,
        normalized_sidecar,
        near_sidecar,
        metrics,
        identity_policy,
        near_policy,
        source_priority,
    )

    def count_seen(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        del rank, world_size
        for doc in data:
            metrics["seen"] += 1
            yield doc

    input_part_details = []
    for remote_part in s1_manifest["output_parts"]:
        local = download_file(s1_repo, remote_part, "main", token, common, inputs)
        input_part_details.append(file_detail(local, common["hashing"]["algorithm"], remote_path=remote_part))
        pipeline: List[Any] = [
            ParquetReader(
                data_folder=str(local.parent),
                glob_pattern=local.name,
                text_key=stage.get("text_column", "text"),
                id_key=stage.get("id_column", "id"),
                read_metadata=True,
                default_metadata={"source_dataset": dataset["name"], "source_file": s1_manifest["source_file"]},
            ),
            count_seen,
            *build_cleaning(stage),
            RegistryAdapterStep(stage.get("custom_filters") or [], metrics),
            *build_quality(stage),
            dedup,
            *build_post(stage),
            sink,
        ]
        run_pipeline_inline(pipeline)
        local.unlink(missing_ok=True)

    digest_size = 16 if common["hashing"]["algorithm"] == "xxh3_128" else 8
    side_count = sidecar.stat().st_size // digest_size if sidecar.exists() else 0
    normalized_side_count = normalized_sidecar.stat().st_size // digest_size if normalized_sidecar.exists() else 0
    near_side_count = near_sidecar.stat().st_size // 8 if near_sidecar is not None and near_sidecar.exists() else 0
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in sink.paths)
    if side_count != rows or normalized_side_count != rows or (near_sidecar is not None and near_side_count != rows) or rows != metrics["accepted"]:
        raise IOError(
            "Stage-2 atomic verification failed: "
            f"exact={side_count} normalized={normalized_side_count} near={near_side_count} "
            f"rows={rows} accepted={metrics['accepted']}"
        )

    uploaded = []
    output_part_details = []
    for part in sink.paths:
        remote = f"{prefix}/{part.name}"
        output_part_details.append(file_detail(part, common["hashing"]["algorithm"], remote_path=remote))
        upload_file(api, s2_repo, part, remote, token, common)
        uploaded.append(remote)
    side_remote = f"{prefix}/accepted.hashes"
    upload_file(api, s2_repo, sidecar, side_remote, token, common)
    normalized_side_remote = f"{prefix}/accepted.normalized.hashes"
    upload_file(api, s2_repo, normalized_sidecar, normalized_side_remote, token, common)
    near_side_remote = None
    if near_sidecar is not None:
        near_side_remote = f"{prefix}/accepted.near.hashes"
        upload_file(api, s2_repo, near_sidecar, near_side_remote, token, common)
    manifest = {
        "artifact_contract": build_artifact_contract(
            artifact_type="dataset_stage",
            stage="stage2",
            dataset_id=dataset["name"],
            run_id=run_hash,
            config_hash=bundle["stage_hash"],
            source_refs=[
                {
                    "repo_id": s1_repo,
                    "revision": "main",
                    "path": s1_manifest_path,
                    "stage": "stage1",
                }
            ],
            attributes={
                "source_file": s1_manifest["source_file"],
                "dedup_namespace": stage["dedup_namespace"],
            },
        ),
        "version": 1,
        "stage": 2,
        "dataset": dataset["name"],
        "dedup_namespace": stage["dedup_namespace"],
        "source_key": key,
        "stage1_manifest": s1_manifest_path,
        "stage2_semantic_hash": run_hash,
        "stage2_config_hash": bundle["stage_hash"],
        "processing_status": "committed",
        "source_priority": source_priority,
        "source_priority_policy": (common.get("deduplication") or {}).get("source_priority", {}),
        "hash_algorithm": common["hashing"]["algorithm"],
        "hash_digest_size": digest_size,
        "hash_sidecar": side_remote,
        "normalized_hash_sidecar": normalized_side_remote,
        "near_duplicate_sidecar": near_side_remote,
        "near_duplicate_policy": near_policy,
        "input_part_details": input_part_details,
        "documents_seen": metrics["seen"],
        "accepted": metrics["accepted"],
        "duplicates": metrics["duplicates"],
        "near_duplicates": metrics["near_duplicates"],
        "custom_rejections": metrics["custom_rejected"],
        "rejected": metrics["custom_rejected"],
        "rejected_by_reason": dict(sorted(metrics["rejected_by_reason"].items())),
        "duplicate_by_reason": dict(sorted(metrics["duplicate_by_reason"].items())),
        "error_count": 0,
        "errors_by_reason": {},
        "counts": {
            "seen": metrics["seen"],
            "accepted": metrics["accepted"],
            "rejected": metrics["custom_rejected"],
            "duplicates": metrics["duplicates"],
            "near_duplicates": metrics["near_duplicates"],
            "errors": 0,
            "rejected_by_reason": dict(sorted(metrics["rejected_by_reason"].items())),
            "duplicate_by_reason": dict(sorted(metrics["duplicate_by_reason"].items())),
            "errors_by_reason": {},
        },
        "output_parts": uploaded,
        "output_part_details": output_part_details,
        "committed_at": utc_now(),
    }
    manifest_local = work / "manifest.json"; write_json(manifest_local, manifest)
    upload_file(api, s2_repo, manifest_local, manifest_remote, token, common)  # commit marker last
    store.commit_sidecars(
        key,
        manifest_remote,
        sidecar,
        normalized_sidecar,
        near_sidecar,
        digest_size,
        int(near_policy.get("bucket_bits", 8)),
    )
    shutil.rmtree(work, ignore_errors=True)
    return {"source": s1_manifest["source_file"], "status": "processed", **metrics}


def process_source(
    bundle: Dict[str, Any], s1_manifest_path: str, api: HfApi, token: str, store: CommittedHashStore
) -> Dict[str, Any]:
    """Run Stage 2 and persist a retryable failure record on errors."""
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    run_hash = stage2_semantic_hash(bundle)
    key = s1_manifest_path.rstrip("/").split("/")[-2]
    prefix = remote_prefix(dataset["name"], stage["dedup_namespace"], run_hash, key)
    try:
        return _process_source(bundle, s1_manifest_path, api, token, store)
    except Exception as exc:
        contract = build_artifact_contract(
            artifact_type="dataset_stage",
            stage="stage2",
            dataset_id=dataset["name"],
            run_id=run_hash,
            config_hash=bundle["stage_hash"],
            source_refs=[
                {"repo_id": dataset["repos"]["stage1"], "revision": "main", "path": s1_manifest_path, "stage": "stage1"}
            ],
            attributes={"source_key": key, "dedup_namespace": stage["dedup_namespace"]},
        )
        write_failure_manifest(
            api=api,
            repo_id=dataset["repos"]["stage2"],
            token=token,
            common=common,
            local_root=local_work_root(common),
            manifest_remote=f"{prefix}/manifest.json",
            artifact_contract=contract,
            stage=2,
            source_key=key,
            exc=exc,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="LaughLM Stage 2 content processing")
    parser.add_argument("--config", required=True, help="configs/stage2/<dataset>.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--skip-dedup-reconcile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle = load_stage_bundle(args.config, 2, args.common, args.registry)
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    setup_logging(common, "stage2")
    token = hf_token(common); api = HfApi(token=token)
    ensure_repo(api, dataset["repos"]["stage1"], token, common, must_exist=True)
    ensure_repo(api, dataset["repos"]["stage2"], token, common)
    if common.get("plugins", {}).get("filter_paths"):
        custom_filters.load_plugins(common["plugins"]["filter_paths"])

    s1_bundle = load_stage1_bundle_for_stage2(bundle, args.common, args.registry)
    manifests = list_stage1_manifests(bundle, s1_bundle, api, token)
    if args.limit_sources is not None:
        manifests = manifests[: args.limit_sources]
    run_hash = stage2_semantic_hash(bundle)

    table = Table(title=f"Stage 2 dry-run — {dataset['name']}")
    table.add_column("Field"); table.add_column("Value")
    table.add_row("Stage-1 sources", str(len(manifests)))
    table.add_row("Dedup namespace", stage["dedup_namespace"])
    table.add_row("Run hash", run_hash[:12])
    table.add_row("Output", dataset["repos"]["stage2"])
    Console().print(table)
    if args.dry_run:
        return

    db = local_work_root(common) / "dedup" / f"{slug(stage['dedup_namespace'])}.sqlite3"
    store = CommittedHashStore(db, stage["dedup_namespace"], common["hashing"]["algorithm"])
    try:
        if not args.skip_dedup_reconcile:
            reconcile_namespace(bundle, api, token, store)
        for m in manifests:
            Console().print(process_source(bundle, m, api, token, store))
    finally:
        store.close()


if __name__ == "__main__":
    main()
