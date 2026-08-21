#!/usr/bin/env python3
"""Stage 1: source-specific metadata selection -> durable Parquet on HF.

Each run consumes ONE configs/stage1/<dataset>.yaml file. Source identity and
output repository are resolved from configs/datasets.yaml. Every upstream file
is an atomic unit: output parts are uploaded first and manifest.json is uploaded
last. Resume simply skips sources with an existing manifest.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xxhash
import zstandard as zstd
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table

from config_loader import load_stage_bundle, stage1_semantic_hash
from manifest_contract import build_artifact_contract, validate_committed_manifest
from pipeline_utils import (
    download_file,
    ensure_disk_space,
    ensure_repo,
    file_detail,
    hf_token,
    list_repo_files,
    matches_any,
    read_remote_json,
    setup_logging,
    slug,
    upload_file,
    local_work_root,
    runtime_settings,
    utc_now,
    write_json,
    write_failure_manifest,
)

SUPPORTED_OPS = {"eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "contains"}


def source_key(repo_id: str, revision: str, path: str, run_hash: str) -> str:
    return xxhash.xxh3_128_hexdigest(f"{repo_id}\0{revision}\0{path}\0{run_hash}".encode())


def remote_prefix(dataset: str, run_hash: str, key: str) -> str:
    return f"{dataset}/runs/{run_hash[:12]}/sources/{key}"


def _arrow_expression(rule: Dict[str, Any]):
    field = ds.field(rule["column"])
    op, value = rule["op"], rule["value"]
    if op == "eq": return field == value
    if op == "neq": return field != value
    if op == "gt": return field > value
    if op == "gte": return field >= value
    if op == "lt": return field < value
    if op == "lte": return field <= value
    if op == "in": return field.isin(value)
    if op == "not_in": return ~field.isin(value)
    raise ValueError(f"Operator {op} is not Arrow-pushdown compatible")


def _json_accept(row: Dict[str, Any], rules: List[Dict[str, Any]]) -> bool:
    for r in rules:
        col, op, value = r["column"], r["op"], r["value"]
        actual = row.get(col)
        if op == "eq" and actual != value: return False
        if op == "neq" and actual == value: return False
        if op == "gt" and not (actual is not None and actual > value): return False
        if op == "gte" and not (actual is not None and actual >= value): return False
        if op == "lt" and not (actual is not None and actual < value): return False
        if op == "lte" and not (actual is not None and actual <= value): return False
        if op == "in" and actual not in value: return False
        if op == "not_in" and actual in value: return False
        if op == "contains" and (actual is None or str(value) not in str(actual)): return False
    return True


def _select_json_columns(row: Dict[str, Any], columns: Any) -> Dict[str, Any]:
    if columns == "*":
        return row
    return {c: row.get(c) for c in columns}


class BufferedParquetWriter:
    def __init__(self, out_dir: Path, target_mb: int, max_documents: int, compression: Dict[str, Any]) -> None:
        self.out_dir = out_dir
        self.target_bytes = int(target_mb * 1024 * 1024)
        self.max_documents = int(max_documents)
        self.compression = compression
        self.tables: List[pa.Table] = []
        self.nbytes = 0
        self.rows = 0
        self.index = 0
        self.paths: List[Path] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self.tables.append(table)
        self.nbytes += int(table.nbytes)
        self.rows += table.num_rows
        if self.nbytes >= self.target_bytes or self.rows >= self.max_documents:
            self.flush()

    def flush(self) -> None:
        if not self.tables:
            return
        table = pa.concat_tables(self.tables, promote_options="default")
        path = self.out_dir / f"part-{self.index:05d}.parquet"
        pq.write_table(
            table,
            path,
            compression=self.compression["codec"],
            compression_level=self.compression.get("level"),
        )
        self.paths.append(path)
        self.index += 1
        self.tables.clear()
        self.nbytes = 0
        self.rows = 0

    def close(self) -> List[Path]:
        self.flush()
        return self.paths


def iter_parquet_tables(local: Path, stage: Dict[str, Any], batch_size: int) -> tuple[Iterator[pa.Table], int]:
    dataset = ds.dataset(str(local), format="parquet")
    rules = stage.get("metadata_filters") or []
    schema_names = set(dataset.schema.names)
    for rule in rules:
        if rule["op"] not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported Stage-1 op {rule['op']!r}")
        if rule["column"] not in schema_names:
            raise ValueError(f"Missing filter column {rule['column']!r}; schema={sorted(schema_names)}")
    output_columns = stage.get("output_columns", "*")
    if output_columns != "*":
        missing = [c for c in output_columns if c not in schema_names]
        if missing:
            raise ValueError(f"Missing output columns {missing}; schema={sorted(schema_names)}")
    push = [r for r in rules if r["op"] != "contains"]
    post = [r for r in rules if r["op"] == "contains"]
    expression = None
    for rule in push:
        e = _arrow_expression(rule)
        expression = e if expression is None else expression & e
    columns = None if output_columns == "*" else list(output_columns)
    # contains filters need their source column even when it is not an output column.
    scan_columns = None if columns is None else sorted(set(columns) | {r["column"] for r in post})
    scanner = dataset.scanner(filter=expression, columns=scan_columns, batch_size=batch_size, use_threads=True)
    total = dataset.count_rows()

    def _gen():
        for rb in scanner.to_batches():
            table = pa.Table.from_batches([rb])
            if post:
                mask = None
                import pyarrow.compute as pc
                for rule in post:
                    m = pc.match_substring(table[rule["column"]], pattern=str(rule["value"]))
                    mask = m if mask is None else pc.and_(mask, m)
                table = table.filter(mask)
            if columns is not None:
                table = table.select(columns)
            if table.num_rows:
                yield table
    return _gen(), total


def _open_json_text(local: Path):
    name = local.name.lower()
    if name.endswith(".zst"):
        raw = local.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        import io
        return io.TextIOWrapper(reader, encoding="utf-8"), raw
    if name.endswith(".gz"):
        return gzip.open(local, "rt", encoding="utf-8"), None
    return local.open("r", encoding="utf-8"), None


def iter_json_tables(local: Path, stage: Dict[str, Any], batch_size: int) -> tuple[Iterator[pa.Table], int | None]:
    rules = stage.get("metadata_filters") or []
    for r in rules:
        if r["op"] not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported Stage-1 op {r['op']!r}")
    output_columns = stage.get("output_columns", "*")

    def _gen():
        handle, extra = _open_json_text(local)
        rows: List[Dict[str, Any]] = []
        try:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if _json_accept(row, rules):
                    rows.append(_select_json_columns(row, output_columns))
                if len(rows) >= batch_size:
                    yield pa.Table.from_pylist(rows)
                    rows.clear()
            if rows:
                yield pa.Table.from_pylist(rows)
        finally:
            handle.close()
            if extra is not None:
                extra.close()
    return _gen(), None


def _process_source(
    filename: str,
    bundle: Dict[str, Any],
    api: HfApi,
    token: str,
    existing_remote: set[str],
    batch_size: int,
) -> Dict[str, Any]:
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    source, out_repo = dataset["source"], dataset["repos"]["stage1"]
    started_at = time.perf_counter()
    run_hash = stage1_semantic_hash(bundle)
    key = source_key(source["repo_id"], source.get("revision", "main"), filename, run_hash)
    prefix = remote_prefix(dataset["name"], run_hash, key)
    manifest_remote = f"{prefix}/manifest.json"
    if manifest_remote in existing_remote:
        try:
            existing_manifest = read_remote_json(out_repo, manifest_remote, token, common)
            errors = validate_committed_manifest(
                existing_manifest, expected_stage="stage1", available_files=existing_remote
            )
            if not errors:
                return {"source": filename, "status": "skipped"}
            logging.getLogger("stage1").warning(
                "Ignoring uncommitted or incomplete manifest %s: %s", manifest_remote, "; ".join(errors)
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("stage1").warning("Ignoring unreadable manifest %s: %s", manifest_remote, exc)

    work = local_work_root(common) / dataset["name"] / "stage1" / key
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    download_started = time.perf_counter()
    local = download_file(
        source["repo_id"], filename, source.get("revision", "main"), token, common, work / "input"
    )
    source_file_detail = file_detail(local, common["hashing"]["algorithm"], remote_path=filename)
    download_seconds = time.perf_counter() - download_started
    processing_started = time.perf_counter()
    writer = BufferedParquetWriter(work / "parts", stage["shard"]["target_size_mb"], stage["shard"]["max_documents"], common["compression"])
    fmt = source.get("format", "parquet")
    if fmt == "parquet":
        tables, seen = iter_parquet_tables(local, stage, batch_size)
    elif fmt in {"jsonl", "jsonl_zst", "jsonl_gz"}:
        tables, seen = iter_json_tables(local, stage, batch_size)
    else:
        raise ValueError(f"Unsupported source format {fmt!r} for {dataset['name']}")

    accepted = 0
    documents_seen = seen if seen is not None else 0
    for table in tables:
        if seen is None:
            documents_seen += table.num_rows
        accepted += table.num_rows
        writer.add(table)
    parts = writer.close()
    processing_seconds = time.perf_counter() - processing_started
    rejected = max(0, documents_seen - accepted)
    rejected_by_reason = {"metadata_filter": rejected} if rejected else {}

    uploaded: List[str] = []
    output_part_details: List[Dict[str, Any]] = []
    upload_started = time.perf_counter()
    for part in parts:
        remote = f"{prefix}/{part.name}"
        output_part_details.append(file_detail(part, common["hashing"]["algorithm"], remote_path=remote))
        upload_file(api, out_repo, part, remote, token, common)
        uploaded.append(remote)
    upload_seconds = time.perf_counter() - upload_started
    manifest = {
        "artifact_contract": build_artifact_contract(
            artifact_type="dataset_stage",
            stage="stage1",
            dataset_id=dataset["name"],
            run_id=run_hash,
            config_hash=bundle["stage_hash"],
            source_refs=[
                {
                    "repo_id": source["repo_id"],
                    "revision": source.get("revision", "main"),
                    "path": filename,
                }
            ],
            attributes={"source_file": filename},
        ),
        "version": 1,
        "stage": 1,
        "dataset": dataset["name"],
        "source_repo": source["repo_id"],
        "source_revision": source.get("revision", "main"),
        "source_file": filename,
        "source_key": key,
        "stage1_semantic_hash": run_hash,
        "stage1_config_hash": bundle["stage_hash"],
        "processing_status": "committed",
        "source_file_detail": source_file_detail,
        "documents_seen": documents_seen,
        "documents_accepted": accepted,
        "documents_rejected": rejected,
        "rejected_by_reason": rejected_by_reason,
        "duplicate_count": 0,
        "duplicate_by_reason": {},
        "error_count": 0,
        "errors_by_reason": {},
        "counts": {
            "seen": documents_seen,
            "accepted": accepted,
            "rejected": rejected,
            "duplicates": 0,
            "errors": 0,
            "rejected_by_reason": rejected_by_reason,
            "duplicate_by_reason": {},
            "errors_by_reason": {},
        },
        "output_parts": uploaded,
        "output_part_details": output_part_details,
        "timings": {
            "download_seconds": round(download_seconds, 6),
            "processing_seconds": round(processing_seconds, 6),
            "upload_seconds": round(upload_seconds, 6),
            "total_seconds": round(time.perf_counter() - started_at, 6),
        },
        "committed_at": utc_now(),
    }
    manifest_local = work / "manifest.json"
    write_json(manifest_local, manifest)
    upload_file(api, out_repo, manifest_local, manifest_remote, token, common)  # commit marker last
    shutil.rmtree(work, ignore_errors=True)
    return {"source": filename, "status": "processed", "accepted": accepted, "parts": len(parts)}


def process_source(
    filename: str,
    bundle: Dict[str, Any],
    api: HfApi,
    token: str,
    existing_remote: set[str],
    batch_size: int,
) -> Dict[str, Any]:
    """Run Stage 1 and persist a retryable failure record on errors."""
    common, dataset, source = bundle["common"], bundle["dataset"], bundle["dataset"]["source"]
    run_hash = stage1_semantic_hash(bundle)
    key = source_key(source["repo_id"], source.get("revision", "main"), filename, run_hash)
    prefix = remote_prefix(dataset["name"], run_hash, key)
    try:
        return _process_source(filename, bundle, api, token, existing_remote, batch_size)
    except Exception as exc:
        contract = build_artifact_contract(
            artifact_type="dataset_stage",
            stage="stage1",
            dataset_id=dataset["name"],
            run_id=run_hash,
            config_hash=bundle["stage_hash"],
            source_refs=[
                {
                    "repo_id": source["repo_id"],
                    "revision": source.get("revision", "main"),
                    "path": filename,
                }
            ],
            attributes={"source_file": filename},
        )
        write_failure_manifest(
            api=api,
            repo_id=dataset["repos"]["stage1"],
            token=token,
            common=common,
            local_root=local_work_root(common),
            manifest_remote=f"{prefix}/manifest.json",
            artifact_contract=contract,
            stage=1,
            source_key=key,
            exc=exc,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="LaughLM Stage 1 metadata/source filtering")
    parser.add_argument("--config", required=True, help="configs/stage1/<dataset>.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-inflight-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle = load_stage_bundle(args.config, 1, args.common, args.registry)
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    runtime = runtime_settings(common)
    file_workers = args.workers if args.workers is not None else runtime["file_workers"]
    batch_size = args.batch_size if args.batch_size is not None else runtime["batch_rows"]
    max_inflight = (
        args.max_inflight_files
        if args.max_inflight_files is not None
        else runtime["max_inflight_files"]
    )
    if file_workers <= 0 or batch_size <= 0 or max_inflight <= 0:
        raise ValueError("workers, batch-size, and max-inflight-files must be > 0")
    effective_workers = min(file_workers, max_inflight)
    disk = ensure_disk_space(common, local_work_root(common))
    setup_logging(common, "stage1")
    token = hf_token(common)
    api = HfApi(token=token)
    ensure_repo(api, dataset["source"]["repo_id"], token, common, must_exist=True)
    ensure_repo(api, dataset["repos"]["stage1"], token, common)

    all_files = list_repo_files(api, dataset["source"]["repo_id"], token, common)
    patterns = stage.get("source_patterns") or dataset["source"].get("glob_patterns") or ["*.parquet"]
    excludes = stage.get("source_exclude_patterns") or []
    files = [f for f in all_files if matches_any(f, patterns) and not matches_any(f, excludes)]
    files.sort()
    if args.limit_files is not None:
        files = files[: args.limit_files]
    run_hash = stage1_semantic_hash(bundle)
    existing = set(list_repo_files(api, dataset["repos"]["stage1"], token, common))

    table = Table(title=f"Stage 1 dry-run — {dataset['name']}")
    table.add_column("Field"); table.add_column("Value")
    table.add_row("Source", dataset["source"]["repo_id"])
    table.add_row("Format", dataset["source"].get("format", "parquet"))
    table.add_row("Matched files", str(len(files)))
    table.add_row("Run hash", run_hash[:12])
    table.add_row("Output", dataset["repos"]["stage1"])
    table.add_row("File workers", str(effective_workers))
    table.add_row("Max in-flight files", str(max_inflight))
    table.add_row("Batch rows", str(batch_size))
    table.add_row("Free disk GiB", str(disk["free_gib"]))
    Console().print(table)
    if args.dry_run:
        return

    results = []
    file_iter = iter(files)
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        pending = set()
        for _ in range(min(max_inflight, len(files))):
            try:
                filename = next(file_iter)
            except StopIteration:
                break
            pending.add(
                pool.submit(process_source, filename, bundle, api, token, existing, batch_size)
            )

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                logging.getLogger("stage1").info("%s", result)
                Console().print(result)
                try:
                    filename = next(file_iter)
                except StopIteration:
                    continue
                pending.add(
                    pool.submit(process_source, filename, bundle, api, token, existing, batch_size)
                )
    processed = sum(r["status"] == "processed" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    Console().print(f"[green]Stage 1 complete[/green]: processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
