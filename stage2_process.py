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
from config_loader import (
    load_stage_bundle,
    resolve_relative,
    stage1_semantic_hash,
    stage2_semantic_hash,
)
from pipeline_utils import (
    download_file,
    ensure_repo,
    hf_token,
    list_repo_files,
    read_remote_json,
    setup_logging,
    slug,
    upload_file,
    utc_now,
    write_json,
)

SQLITE_QUERY_CHUNK = 500


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
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sources (source_key TEXT PRIMARY KEY, manifest_path TEXT NOT NULL, committed_at TEXT NOT NULL)"
        )
        self.conn.execute("CREATE TABLE IF NOT EXISTS metadata (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        prev = self.conn.execute("SELECT v FROM metadata WHERE k='namespace'").fetchone()
        prev_alg = self.conn.execute("SELECT v FROM metadata WHERE k='algorithm'").fetchone()
        if (prev and prev[0] != namespace) or (prev_alg and prev_alg[0] != algorithm):
            self.conn.execute("DELETE FROM hashes")
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

    def commit_sidecar(self, key: str, manifest_path: str, sidecar: Path, digest_size: int) -> int:
        if self.source_is_committed(key):
            return 0
        if sidecar.stat().st_size % digest_size:
            raise IOError(f"Corrupt hash sidecar: {sidecar}")
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
                continue
            doc.text = cleaned
            yield doc


class CrashSafeExactDedupStep(PipelineStep):
    name = "Crash-safe exact dedup"
    def __init__(self, store: CommittedHashStore, algorithm: str, batch_size: int, sidecar: Path, metrics: Dict[str, int]) -> None:
        super().__init__()
        self.store = store
        self.hasher = xxhash.xxh3_128 if algorithm == "xxh3_128" else xxhash.xxh64
        self.batch_size = batch_size
        self.sidecar = sidecar
        self.metrics = metrics
        self.seen_current: set[bytes] = set()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(b"")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        del rank, world_size
        docs: List[Document] = []
        hashes: List[bytes] = []
        with self.sidecar.open("ab") as side:
            def flush() -> Iterator[Document]:
                if not docs:
                    return iter(())
                existing = self.store.existing(hashes)
                kept: List[Document] = []
                for doc, digest in zip(docs, hashes):
                    if digest in existing or digest in self.seen_current:
                        self.metrics["duplicates"] += 1
                        continue
                    self.seen_current.add(digest)
                    side.write(digest)
                    kept.append(doc)
                docs.clear(); hashes.clear()
                return iter(kept)
            for doc in data:
                docs.append(doc)
                hashes.append(self.hasher(doc.text.encode("utf-8")).digest())
                if len(docs) >= self.batch_size:
                    yield from flush()
            yield from flush()


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
            for field in self.metadata_fields:
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
    for name, entry in registry.items():
        repo = entry.get("repos", {}).get("stage2")
        if not repo:
            continue
        try:
            files = list_repo_files(api, repo, token, common)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("stage2.reconcile").warning("Could not inspect %s: %s", repo, exc)
            continue
        prefix = f"{name}/dedup/{namespace}/"
        for manifest_path in (f for f in files if f.startswith(prefix) and f.endswith("/manifest.json")):
            manifest = read_remote_json(repo, manifest_path, token, common)
            key = manifest["source_key"]
            if store.source_is_committed(key):
                continue
            side = download_file(repo, manifest["hash_sidecar"], "main", token, common)
            store.commit_sidecar(key, manifest_path, side, int(manifest["hash_digest_size"]))


def process_source(bundle: Dict[str, Any], s1_manifest_path: str, api: HfApi, token: str, store: CommittedHashStore) -> Dict[str, Any]:
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    s1_repo, s2_repo = dataset["repos"]["stage1"], dataset["repos"]["stage2"]
    s1_manifest = read_remote_json(s1_repo, s1_manifest_path, token, common)
    run_hash = stage2_semantic_hash(bundle)
    key = source_key(s1_manifest, run_hash)
    prefix = remote_prefix(dataset["name"], stage["dedup_namespace"], run_hash, key)
    manifest_remote = f"{prefix}/manifest.json"

    remote_files = set(list_repo_files(api, s2_repo, token, common))
    if manifest_remote in remote_files:
        if not store.source_is_committed(key):
            manifest = read_remote_json(s2_repo, manifest_remote, token, common)
            side = download_file(s2_repo, manifest["hash_sidecar"], "main", token, common)
            store.commit_sidecar(key, manifest_remote, side, int(manifest["hash_digest_size"]))
        return {"source": s1_manifest["source_file"], "status": "skipped"}

    work = Path(common["storage"]["local_work_dir"]) / dataset["name"] / "stage2" / key
    shutil.rmtree(work, ignore_errors=True)
    inputs = work / "input"; inputs.mkdir(parents=True, exist_ok=True)
    sidecar = work / "accepted.hashes"
    metrics = {"seen": 0, "accepted": 0, "duplicates": 0, "custom_rejected": 0}
    sink = LocalParquetSink(work / "parts", stage.get("output_metadata_fields", []), stage["shard"], common["compression"], metrics)
    dedup = CrashSafeExactDedupStep(store, common["hashing"]["algorithm"], int(common["hashing"].get("dedup_batch_size", 500)), sidecar, metrics)

    def count_seen(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        del rank, world_size
        for doc in data:
            metrics["seen"] += 1
            yield doc

    for remote_part in s1_manifest["output_parts"]:
        local = download_file(s1_repo, remote_part, "main", token, common, inputs)
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
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in sink.paths)
    if side_count != rows or rows != metrics["accepted"]:
        raise IOError(f"Stage-2 atomic verification failed: sidecar={side_count} rows={rows} accepted={metrics['accepted']}")

    uploaded = []
    for part in sink.paths:
        remote = f"{prefix}/{part.name}"
        upload_file(api, s2_repo, part, remote, token, common)
        uploaded.append(remote)
    side_remote = f"{prefix}/accepted.hashes"
    upload_file(api, s2_repo, sidecar, side_remote, token, common)
    manifest = {
        "version": 1,
        "stage": 2,
        "dataset": dataset["name"],
        "dedup_namespace": stage["dedup_namespace"],
        "source_key": key,
        "stage1_manifest": s1_manifest_path,
        "stage2_semantic_hash": run_hash,
        "stage2_config_hash": bundle["stage_hash"],
        "hash_algorithm": common["hashing"]["algorithm"],
        "hash_digest_size": digest_size,
        "hash_sidecar": side_remote,
        "documents_seen": metrics["seen"],
        "accepted": metrics["accepted"],
        "duplicates": metrics["duplicates"],
        "custom_rejections": metrics["custom_rejected"],
        "output_parts": uploaded,
        "committed_at": utc_now(),
    }
    manifest_local = work / "manifest.json"; write_json(manifest_local, manifest)
    upload_file(api, s2_repo, manifest_local, manifest_remote, token, common)  # commit marker last
    store.commit_sidecar(key, manifest_remote, sidecar, digest_size)
    shutil.rmtree(work, ignore_errors=True)
    return {"source": s1_manifest["source_file"], "status": "processed", **metrics}


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

    db = Path(common["storage"]["local_work_dir"]) / "dedup" / f"{slug(stage['dedup_namespace'])}.sqlite3"
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
