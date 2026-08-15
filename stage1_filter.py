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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from pipeline_utils import (
    download_file,
    ensure_repo,
    hf_token,
    list_repo_files,
    matches_any,
    setup_logging,
    slug,
    upload_file,
    utc_now,
    write_json,
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


def process_source(
    filename: str,
    bundle: Dict[str, Any],
    api: HfApi,
    token: str,
    existing_remote: set[str],
    batch_size: int,
) -> Dict[str, Any]:
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
    source, out_repo = dataset["source"], dataset["repos"]["stage1"]
    run_hash = stage1_semantic_hash(bundle)
    key = source_key(source["repo_id"], source.get("revision", "main"), filename, run_hash)
    prefix = remote_prefix(dataset["name"], run_hash, key)
    manifest_remote = f"{prefix}/manifest.json"
    if manifest_remote in existing_remote:
        return {"source": filename, "status": "skipped"}

    work = Path(common["storage"]["local_work_dir"]) / dataset["name"] / "stage1" / key
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    local = download_file(
        source["repo_id"], filename, source.get("revision", "main"), token, common, work / "input"
    )
    writer = BufferedParquetWriter(work / "parts", stage["shard"]["target_size_mb"], stage["shard"]["max_documents"], common["compression"])
    fmt = source.get("format", "parquet")
    if fmt == "parquet":
        tables, seen = iter_parquet_tables(local, stage, batch_size)
    elif fmt in {"jsonl", "jsonl_zst", "jsonl_gz"}:
        tables, seen = iter_json_tables(local, stage, batch_size)
    else:
        raise ValueError(f"Unsupported source format {fmt!r} for {dataset['name']}")

    accepted = 0
    for table in tables:
        accepted += table.num_rows
        writer.add(table)
    parts = writer.close()

    uploaded: List[str] = []
    for part in parts:
        remote = f"{prefix}/{part.name}"
        upload_file(api, out_repo, part, remote, token, common)
        uploaded.append(remote)
    manifest = {
        "version": 1,
        "stage": 1,
        "dataset": dataset["name"],
        "source_repo": source["repo_id"],
        "source_revision": source.get("revision", "main"),
        "source_file": filename,
        "source_key": key,
        "stage1_semantic_hash": run_hash,
        "stage1_config_hash": bundle["stage_hash"],
        "documents_seen": seen,
        "documents_accepted": accepted,
        "output_parts": uploaded,
        "committed_at": utc_now(),
    }
    manifest_local = work / "manifest.json"
    write_json(manifest_local, manifest)
    upload_file(api, out_repo, manifest_local, manifest_remote, token, common)  # commit marker last
    shutil.rmtree(work, ignore_errors=True)
    return {"source": filename, "status": "processed", "accepted": accepted, "parts": len(parts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="LaughLM Stage 1 metadata/source filtering")
    parser.add_argument("--config", required=True, help="configs/stage1/<dataset>.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle = load_stage_bundle(args.config, 1, args.common, args.registry)
    common, dataset, stage = bundle["common"], bundle["dataset"], bundle["stage"]
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
    Console().print(table)
    if args.dry_run:
        return

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process_source, f, bundle, api, token, existing, args.batch_size) for f in files]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            logging.getLogger("stage1").info("%s", result)
            Console().print(result)
    processed = sum(r["status"] == "processed" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    Console().print(f"[green]Stage 1 complete[/green]: processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
