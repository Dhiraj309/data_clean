#!/usr/bin/env python3
"""Resumable Stage 2 token-level mixing of completed Stage-1 Parquet shards.

Stage 2 downloads at most one current Parquet shard per source, reads it in
batches, and writes immutable output shards. The remote checkpoint is
committed atomically with each completed output shard, so a restart does not
replay the previous mixture.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from smol_pipeline import (
    OUTPUT_SCHEMA,
    download_json_if_present,
    ensure_dataset_repo,
    hf_token,
    load_config,
    stable_id,
    write_json,
)


MIXER_VERSION = 2


@dataclass(frozen=True)
class ResourcePlan:
    """Safe defaults that scale with the host without oversubscribing it."""

    cpu_workers: int
    download_workers: int
    upload_workers: int


def make_resource_plan(
    cpu_count: int | None = None,
    cpu_workers: int | None = None,
    download_workers: int | None = None,
    upload_workers: int | None = None,
) -> ResourcePlan:
    cpus = max(1, int(cpu_count or (os.cpu_count() or 1)))
    return ResourcePlan(
        cpu_workers=max(1, min(int(cpu_workers or max(1, cpus - 2)), 64)),
        download_workers=max(1, min(int(download_workers or max(1, cpus // 8)), 24)),
        # Hub commits are ordered and should remain single-flight. This is
        # intentionally capped at one even on very large machines.
        upload_workers=1 if upload_workers is None else max(1, min(int(upload_workers), 1)),
    )


def empty_state(
    sources: list[dict[str, Any]],
    config_hash: str = "",
    source_revisions: dict[str, str] | None = None,
    source_files: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "stage": 2,
        "mixer_version": MIXER_VERSION,
        "config_hash": config_hash,
        "source_revisions": source_revisions or {},
        "source_files": source_files or {},
        "rows": 0,
        "estimated_tokens": 0,
        "rows_by_source": {source["name"]: 0 for source in sources},
        "tokens_by_source": {source["name"]: 0 for source in sources},
        "source_cursors": {
            source["name"]: {
                "file_index": 0,
                "row_offset": 0,
                "rows": 0,
                "tokens": 0,
                "exhausted": False,
            }
            for source in sources
        },
        "next_part": 0,
        "finished": False,
    }


def _tie_rank(seed: int, name: str) -> int:
    digest = hashlib.blake2b(
        f"{int(seed)}:{name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def choose_source_by_token_debt(
    sources: list[dict[str, Any]],
    tokens_by_source: dict[str, int],
    exhausted: set[str],
    seed: int,
    target_tokens: int | None = None,
) -> dict[str, Any]:
    """Choose the available source furthest below its token target."""
    total_weight = sum(float(source["weight"]) for source in sources)
    candidates = []
    for source in sources:
        name = source["name"]
        if name in exhausted:
            continue
        weight = float(source["weight"]) / total_weight
        emitted = int(tokens_by_source.get(name, 0))
        if target_tokens is not None and emitted >= int(target_tokens * weight):
            continue
        candidates.append((emitted / weight, _tie_rank(seed, name), source))
    if not candidates:
        raise StopIteration
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _source_manifest(
    api: HfApi, source: dict[str, Any], token: str
) -> tuple[str, list[str]]:
    """Resolve one completed Stage-1 source to an immutable revision/files."""
    repo_id = source["repo_id"]
    requested_revision = source.get("revision", "main")
    info = api.repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=requested_revision,
        token=token,
    )
    revision = str(getattr(info, "sha", None) or requested_revision)
    prefix = source.get("path_prefix", "").strip("/")
    prefix_path = f"{prefix}/" if prefix else ""
    progress_name = str(source.get("progress_file", "progress.json")).lstrip("/")
    progress_path = prefix_path + progress_name
    progress = download_json_if_present(repo_id, progress_path, revision, token)
    if progress is None:
        raise RuntimeError(f"Missing Stage-1 progress manifest: {repo_id}:{progress_path}")
    if not bool(progress.get("finished", False)):
        raise RuntimeError(f"Stage-1 source is not complete: {repo_id}:{prefix_path}")

    shard_count = int(progress.get("next_shard", 0))
    filename_prefix = str(source.get("filename_prefix", source["name"]))
    files = [
        f"{prefix_path}{filename_prefix}_shard_{index:05d}.parquet"
        for index in range(shard_count)
    ]
    repo_files = set(
        api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    )
    missing = [filename for filename in files if filename not in repo_files]
    if missing:
        raise RuntimeError(
            f"Stage-1 progress references missing files in {repo_id}: {missing[:3]}"
        )
    if not files:
        raise RuntimeError(f"Stage-1 source has no completed shards: {repo_id}:{prefix_path}")
    return revision, files


class ShardPrefetcher:
    """Bounded background downloader for current/next source shards."""

    def __init__(self, workers: int) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.futures: dict[tuple[str, int], Future[str]] = {}

    def submit(
        self,
        source: dict[str, Any],
        revision: str,
        filename: str,
        file_index: int,
        token: str,
        local_dir: Path,
    ) -> None:
        key = (str(source["name"]), int(file_index))
        if key in self.futures:
            return
        local_dir.mkdir(parents=True, exist_ok=True)
        self.futures[key] = self.executor.submit(
            hf_hub_download,
            repo_id=source["repo_id"],
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=token,
            local_dir=str(local_dir),
        )

    def take(
        self,
        source: dict[str, Any],
        revision: str,
        filename: str,
        file_index: int,
        token: str,
        local_dir: Path,
    ) -> Path:
        key = (str(source["name"]), int(file_index))
        self.submit(source, revision, filename, file_index, token, local_dir)
        return Path(self.futures.pop(key).result())

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)


class SourceCursor:
    """One lazy local Parquet shard and a resumable row cursor."""

    def __init__(
        self,
        source: dict[str, Any],
        revision: str,
        files: list[str],
        token: str,
        local_root: Path,
        state: dict[str, Any],
        batch_rows: int,
        prefetcher: ShardPrefetcher,
    ) -> None:
        self.source = source
        self.revision = revision
        self.files = files
        self.token = token
        self.prefetcher = prefetcher
        self.local_dir = local_root / source["name"]
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.file_index = int(state.get("file_index", 0))
        self.row_offset = int(state.get("row_offset", 0))
        self.rows = int(state.get("rows", 0))
        self.tokens = int(state.get("tokens", 0))
        self.exhausted = bool(state.get("exhausted", False))
        self.batch_rows = max(1, int(batch_rows))
        self._local_path: Path | None = None
        self._parquet: pq.ParquetFile | None = None
        self._batches = None
        self._arrow_batch: pa.RecordBatch | None = None
        self._row_buffer: deque[dict[str, Any]] = deque()
        self._open_position()

    @property
    def name(self) -> str:
        return str(self.source["name"])

    def _open_position(self) -> None:
        if self.exhausted or self.file_index >= len(self.files):
            self.exhausted = True
            return
        remote_name = self.files[self.file_index]
        downloaded = self.prefetcher.take(
            self.source,
            self.revision,
            remote_name,
            self.file_index,
            self.token,
            self.local_dir,
        )
        self._local_path = downloaded
        self._parquet = pq.ParquetFile(self._local_path)
        self._batches = iter(self._parquet.iter_batches(batch_size=self.batch_rows))
        self._arrow_batch = None
        self._row_buffer.clear()

        # Resume at a row offset without loading the whole shard. The offset
        # is applied batch-by-batch and is persisted only at output boundaries.
        remaining = self.row_offset
        while remaining > 0:
            try:
                batch = next(self._batches)
            except StopIteration:
                self._advance_file()
                return
            if remaining >= batch.num_rows:
                remaining -= batch.num_rows
                continue
            self._arrow_batch = batch.slice(remaining)
            remaining = 0

    def _advance_file(self) -> None:
        old_path = self._local_path
        self._parquet = None
        self._batches = None
        self._arrow_batch = None
        self._row_buffer.clear()
        self._local_path = None
        if old_path is not None:
            old_path.unlink(missing_ok=True)
        self.file_index += 1
        self.row_offset = 0
        if self.file_index >= len(self.files):
            self.exhausted = True
            return
        self._open_position()

    def _next_batch(self) -> bool:
        while not self._row_buffer and not self.exhausted:
            if self._batches is None:
                self.exhausted = True
                break
            try:
                self._row_buffer.extend(next(self._batches).to_pylist())
            except StopIteration:
                self._advance_file()
        return bool(self._row_buffer)

    def next_batch(self, max_rows: int | None = None) -> pa.RecordBatch | None:
        """Return one Arrow batch and advance the resumable row cursor."""
        while not self.exhausted:
            if self._arrow_batch is not None:
                batch = self._arrow_batch
                self._arrow_batch = None
            elif self._batches is not None:
                try:
                    batch = next(self._batches)
                except StopIteration:
                    self._advance_file()
                    continue
            else:
                self.exhausted = True
                break
            if batch.num_rows == 0:
                continue
            if max_rows is not None and batch.num_rows > max_rows:
                self._arrow_batch = batch.slice(max_rows)
                batch = batch.slice(0, max_rows)
            self.row_offset += batch.num_rows
            self.rows += batch.num_rows
            self._maybe_prefetch_next()
            return batch
        return None

    def _maybe_prefetch_next(self) -> None:
        if self.exhausted or self._parquet is None:
            return
        if self.file_index + 1 >= len(self.files):
            return
        if self.row_offset < max(0, self._parquet.metadata.num_rows - 2 * self.batch_rows):
            return
        self.prefetcher.submit(
            self.source,
            self.revision,
            self.files[self.file_index + 1],
            self.file_index + 1,
            self.token,
            self.local_dir,
        )

    def __next__(self) -> dict[str, Any]:
        if not self._next_batch():
            raise StopIteration
        row = self._row_buffer.popleft()
        self.row_offset += 1
        self.rows += 1
        self._maybe_prefetch_next()
        return row

    def add_tokens(self, tokens: int) -> None:
        self.tokens += int(tokens)

    def state(self) -> dict[str, Any]:
        return {
            "file_index": self.file_index,
            "row_offset": self.row_offset,
            "rows": self.rows,
            "tokens": self.tokens,
            "exhausted": self.exhausted,
            "current_file": self.files[self.file_index]
            if not self.exhausted and self.file_index < len(self.files)
            else None,
        }

    def close(self, remove_current: bool = False) -> None:
        path = self._local_path
        self._parquet = None
        self._batches = None
        self._arrow_batch = None
        self._row_buffer.clear()
        self._local_path = None
        if remove_current and path is not None:
            path.unlink(missing_ok=True)


class OutputShardWriter:
    """Append Arrow batches directly to one compressed output shard."""

    def __init__(
        self,
        output_dir: Path,
        next_part: int,
        target_bytes: int,
        batch_rows: int,
        max_documents: int,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index = int(next_part)
        self.target_bytes = int(target_bytes)
        self.batch_rows = max(1, int(batch_rows))
        self.max_documents = max(1, int(max_documents))
        self.path = self.output_dir / f"part-{self.index:05d}.partial.parquet"
        self.path.unlink(missing_ok=True)
        self._writer: pq.ParquetWriter | None = None
        self._rows: list[dict[str, Any]] = []
        self.documents = 0

    def _flush_batch(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=OUTPUT_SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path,
                OUTPUT_SCHEMA,
                compression="zstd",
                compression_level=3,
            )
        self._writer.write_table(table)
        self.documents += len(self._rows)
        self._rows.clear()

    def add_table(self, table: pa.Table) -> Path | None:
        """Write a complete Arrow table without converting rows to Python."""
        if table.num_rows == 0:
            return None
        self._flush_batch()
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path,
                OUTPUT_SCHEMA,
                compression="zstd",
                compression_level=3,
            )
        self._writer.write_table(table)
        self.documents += table.num_rows
        if self.documents >= self.max_documents or self.path.stat().st_size >= self.target_bytes:
            return self.close()
        return None

    def add(self, row: dict[str, Any]) -> Path | None:
        self._rows.append(row)
        if len(self._rows) >= self.batch_rows:
            self._flush_batch()
        if self.documents >= self.max_documents or (
            self._writer is not None and self.path.stat().st_size >= self.target_bytes
        ):
            return self.close()
        return None

    def close(self) -> Path | None:
        self._flush_batch()
        if self._writer is None:
            return None
        self._writer.close()
        self._writer = None
        if not self.path.is_file() or self.path.stat().st_size == 0:
            self.path.unlink(missing_ok=True)
            return None
        completed = self.output_dir / f"part-{self.index:05d}.parquet"
        self.path.rename(completed)
        self.index += 1
        self.path = self.output_dir / f"part-{self.index:05d}.partial.parquet"
        self.documents = 0
        return completed


def mix_status_panel(
    sources: list[dict[str, Any]],
    rows_by_source: dict[str, int],
    tokens_by_source: dict[str, int],
    rows: int,
    estimated_tokens: int,
    target_tokens: int | None,
    current_source: str,
    status: str,
) -> Panel:
    summary = Table.grid(expand=True)
    summary.add_column(style="cyan")
    summary.add_column(justify="right")
    summary.add_row("Status", status)
    summary.add_row("Current source", current_source or "—")
    summary.add_row("Rows", f"{rows:,}")
    token_value = f"{estimated_tokens:,}"
    if target_tokens:
        token_value += f" / {int(target_tokens):,} ({100.0 * estimated_tokens / int(target_tokens):.2f}%)"
    summary.add_row("Estimated tokens", token_value)

    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("Source", ratio=3)
    table.add_column("Rows", justify="right", width=14)
    table.add_column("Tokens", justify="right", width=16)
    table.add_column("Actual %", justify="right", width=12)
    table.add_column("Target %", justify="right", width=12)
    total_weight = sum(float(source["weight"]) for source in sources)
    for source in sources:
        name = source["name"]
        source_tokens = int(tokens_by_source.get(name, 0))
        actual = (100.0 * source_tokens / estimated_tokens) if estimated_tokens else 0.0
        target = 100.0 * float(source["weight"]) / total_weight
        table.add_row(
            name,
            f"{int(rows_by_source.get(name, 0)):,}",
            f"{source_tokens:,}",
            f"{actual:.2f}%",
            f"{target:.2f}%",
        )
    return Panel(Group(summary, table), title="Stage 2 · token mixture", border_style="magenta")


def load_state(
    local_state: Path,
    repo_id: str,
    remote_state: str,
    token: str,
    sources: list[dict[str, Any]],
    config: dict[str, Any],
    config_hash: str,
    source_revisions: dict[str, str],
    source_files: dict[str, list[str]],
) -> dict[str, Any]:
    remote = download_json_if_present(repo_id, remote_state, "main", token)
    candidate = remote
    if candidate is None and local_state.is_file():
        candidate = json.loads(local_state.read_text(encoding="utf-8"))
    if candidate is None:
        return empty_state(sources, config_hash, source_revisions, source_files)
    if int(candidate.get("mixer_version", 0)) != MIXER_VERSION:
        raise RuntimeError(
            "Existing Stage-2 state uses an older mixer. Use a new run_id or "
            "remove the old Stage-2 output before restarting."
        )

    saved_revisions = candidate.get("source_revisions") or {}
    saved_files = candidate.get("source_files") or {}
    saved_hash = stable_id({
        "mixer_version": MIXER_VERSION,
        "config": config,
        "source_revisions": saved_revisions,
        "source_files": saved_files,
    })
    if candidate.get("config_hash") != saved_hash:
        raise RuntimeError(
            "Stage-2 configuration differs from the saved checkpoint. "
            "Use the original configuration or a new run_id."
        )

    if candidate.get("config_hash") == config_hash:
        return candidate

    # Stage-1 is append-only: completed shards are immutable and new shards
    # are added at the end. This lets an unfinished Stage-2 run resume after
    # a source has been extended, without replaying any mixed output.
    extended = False
    for source in sources:
        name = str(source["name"])
        previous = list(saved_files.get(name, []))
        current = list(source_files.get(name, []))
        if name not in saved_files or name not in source_files:
            raise RuntimeError("Stage-1 source manifest differs from the saved checkpoint")
        if current[: len(previous)] != previous:
            raise RuntimeError(
                f"Stage-1 source {name!r} was removed, reordered, or changed; "
                "use a new run_id."
            )
        if len(current) == len(previous):
            if saved_revisions.get(name) != source_revisions.get(name):
                raise RuntimeError(
                    f"Stage-1 source revision changed for {name!r}; use a new run_id."
                )
        else:
            extended = True

    if not extended:
        raise RuntimeError("Stage-1 source manifest differs from the saved checkpoint")

    resumed = copy.deepcopy(candidate)
    resumed["source_revisions"] = source_revisions
    resumed["source_files"] = source_files
    resumed["config_hash"] = config_hash
    resumed["manifest_extended"] = True
    for source in sources:
        name = str(source["name"])
        cursor = resumed.get("source_cursors", {}).get(name)
        if not isinstance(cursor, dict):
            continue
        if bool(cursor.get("exhausted", False)) and int(cursor.get("file_index", 0)) < len(source_files[name]):
            cursor["exhausted"] = False
    return resumed


def _tokens_for_row(row: dict[str, Any]) -> int:
    for key in ("estimated_tokens", "token_count"):
        value = row.get(key)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    return max(1, int(round(len(str(row.get("text", "")).split()) * 1.3)))


def _prepare_output_batch(
    batch: pa.RecordBatch,
    source: dict[str, Any],
    mix_name: str,
    total_weight: float,
) -> pa.Table:
    """Project one Stage-1 Arrow batch to the stable Stage-2 schema."""
    table = pa.Table.from_batches([batch])
    arrays = []
    for field in OUTPUT_SCHEMA:
        if field.name in table.column_names:
            array = table[field.name].combine_chunks()
            if array.type != field.type:
                array = pc.cast(array, field.type, safe=False)
        else:
            array = pa.nulls(table.num_rows, type=field.type)
        arrays.append(array)
    output = pa.Table.from_arrays(arrays, schema=OUTPUT_SCHEMA)
    output = output.set_column(
        output.schema.get_field_index("source_name"),
        "source_name",
        pa.array([source["name"]] * output.num_rows, type=pa.string()),
    )
    output = output.set_column(
        output.schema.get_field_index("mix_name"),
        "mix_name",
        pa.array([mix_name] * output.num_rows, type=pa.string()),
    )
    output = output.set_column(
        output.schema.get_field_index("mix_weight"),
        "mix_weight",
        pa.array(
            [float(source["weight"]) / total_weight] * output.num_rows,
            type=pa.float64(),
        ),
    )
    return output


def _batch_token_count(batch: pa.RecordBatch) -> int:
    """Sum Stage-1 token metadata in Arrow; fall back only for malformed rows."""
    names = set(batch.schema.names)
    for column_name in ("estimated_tokens", "token_count"):
        if column_name not in names:
            continue
        values = batch.column(batch.schema.get_field_index(column_name))
        try:
            values = pc.cast(values, pa.int64(), safe=False)
            total = pc.sum(values).as_py()
            if total is not None:
                return max(1, int(total))
        except (TypeError, ValueError, pa.ArrowInvalid):
            pass
    return sum(_tokens_for_row(row) for row in batch.to_pylist())


def _commit_output_shard(
    api: HfApi,
    repo_id: str,
    token: str,
    output_path: Path,
    remote_part: str,
    checkpoint_path: Path,
    remote_state: str,
    output_progress: str,
    state_after: dict[str, Any],
) -> None:
    write_json(checkpoint_path, state_after)
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=remote_part, path_or_fileobj=str(output_path)),
            CommitOperationAdd(path_in_repo=remote_state, path_or_fileobj=str(checkpoint_path)),
            CommitOperationAdd(path_in_repo=output_progress, path_or_fileobj=str(checkpoint_path)),
        ],
        commit_message=f"Add {remote_part} and Stage-2 checkpoint",
        token=token,
    )


def _existing_output_parts(
    api: HfApi,
    repo_id: str,
    revision: str,
    prefix: str,
    filename_prefix: str,
    token: str,
) -> list[str]:
    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    root = prefix.rstrip("/") + "/"
    return sorted(
        filename for filename in files
        if filename.startswith(root) and filename.lower().endswith(".parquet")
        and filename.rsplit("/", 1)[-1].startswith(f"{filename_prefix}_shard_")
    )


class UploadPipeline:
    """Overlap processing with one ordered Hub commit at a time."""

    def __init__(self, workers: int) -> None:
        # Multiple concurrent create_commit calls can race on main. Keep the
        # commit stream ordered while allowing the caller to continue writing
        # the next output shard during the upload.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: deque[tuple[Future[None], Path, Path, str, int]] = deque()
        self.workers = max(1, int(workers))

    def submit(
        self,
        api: HfApi,
        repo_id: str,
        token: str,
        output_path: Path,
        remote_part: str,
        checkpoint_path: Path,
        remote_state: str,
        output_progress: str,
        state_after: dict[str, Any],
    ) -> None:
        if self.pending:
            self.wait_one()
        future = self.executor.submit(
            _commit_output_shard,
            api,
            repo_id,
            token,
            output_path,
            remote_part,
            checkpoint_path,
            remote_state,
            output_progress,
            state_after,
        )
        self.pending.append(
            (future, output_path, checkpoint_path, remote_part, int(state_after["last_part_bytes"]))
        )

    def wait_one(self) -> None:
        future, output_path, checkpoint_path, remote_part, size_bytes = self.pending.popleft()
        future.result()
        output_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        print(f"Pushed {remote_part} ({size_bytes / 1024**2:.1f} MiB)", flush=True)

    def wait_all(self) -> None:
        while self.pending:
            self.wait_one()
        self.executor.shutdown(wait=True, cancel_futures=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Override config run_id, useful for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--auto-workers", action="store_true", help="Use hardware-adaptive worker defaults")
    parser.add_argument("--compute-workers", type=int, default=None)
    parser.add_argument("--download-workers", type=int, default=None)
    parser.add_argument("--upload-workers", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if int(cfg.get("stage", -1)) != 2:
        raise ValueError("Stage-2 config must contain stage: 2")
    sources = cfg.get("sources") or []
    if not sources:
        raise ValueError("Stage-2 config requires sources")
    total_weight = sum(float(source.get("weight", 0)) for source in sources)
    if total_weight <= 0:
        raise ValueError("Stage-2 source weights must sum to a positive value")

    if args.run_id:
        cfg = copy.deepcopy(cfg)
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["run_id"] = args.run_id
        cfg["output"]["path_prefix"] = f"_smoke/{args.run_id}/train"

    token = hf_token()
    api = HfApi(token=token)
    output = cfg["output"]
    console = Console()
    target_tokens = args.target_tokens or cfg.get("target_tokens")
    resource_plan = make_resource_plan(
        cpu_workers=args.compute_workers,
        download_workers=args.download_workers,
        upload_workers=args.upload_workers,
    )

    if args.dry_run:
        plan = Table(expand=True, box=None, padding=(0, 1))
        plan.add_column("Source")
        plan.add_column("Repository", ratio=3)
        plan.add_column("Folder")
        plan.add_column("Weight", justify="right")
        for source in sources:
            plan.add_row(
                source["name"], source["repo_id"], source.get("path_prefix", ""),
                f"{100.0 * float(source['weight']) / total_weight:.2f}%",
            )
        plan.caption = (
            f"Output: {output['repo_id']}/{output.get('path_prefix', '')} · "
            f"Target tokens: {int(target_tokens or 0):,}"
        )
        console.print(Panel(plan, title="Stage 2 dry-run", border_style="magenta"))
        return

    ensure_dataset_repo(api, output["repo_id"], token, private=bool(output.get("private", True)))
    run_id = str(output.get("run_id", "v1"))
    local_root = Path(output.get("local_dir", "work/smol_stage2")) / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    local_state = local_root / "state.json"
    remote_state = f"_checkpoints/stage2/{run_id}.json"
    output_prefix = str(output.get("path_prefix", "data/" + run_id)).rstrip("/")
    filename_prefix = str(output.get("filename_prefix", cfg.get("name", "training")))

    manifests = {
        source["name"]: _source_manifest(api, source, token) for source in sources
    }
    source_revisions = {name: value[0] for name, value in manifests.items()}
    source_files = {name: value[1] for name, value in manifests.items()}
    config_hash = stable_id({
        "mixer_version": MIXER_VERSION,
        "config": cfg,
        "source_revisions": source_revisions,
        "source_files": source_files,
    })

    remote_checkpoint = download_json_if_present(output["repo_id"], remote_state, "main", token)
    if args.no_resume:
        if remote_checkpoint is not None or _existing_output_parts(
            api, output["repo_id"], "main", output_prefix, filename_prefix, token
        ):
            raise RuntimeError("--no-resume cannot overwrite an existing Stage-2 checkpoint; use a new --run-id")
        state = empty_state(sources, config_hash, source_revisions, source_files)
    else:
        if remote_checkpoint is None:
            existing = _existing_output_parts(
                api, output["repo_id"], "main", output_prefix, filename_prefix, token
            )
            if existing:
                raise RuntimeError(
                    "Stage-2 output shards exist without a checkpoint; refusing to overwrite them. "
                    "Inspect the Hub repo or use a new run_id."
                )
        state = load_state(
            local_state, output["repo_id"], remote_state, token, sources,
            cfg, config_hash, source_revisions, source_files,
        )
    if state.get("finished"):
        console.print("[dim]Stage 2 is already complete; nothing to process.[/dim]")
        return

    rows = int(state.get("rows", 0))
    estimated_tokens = int(state.get("estimated_tokens", 0))
    rows_by_source = dict(state.get("rows_by_source", {}))
    tokens_by_source = dict(state.get("tokens_by_source", {}))
    if hasattr(pa, "set_cpu_count"):
        pa.set_cpu_count(resource_plan.cpu_workers)
    if hasattr(pa, "set_io_thread_count"):
        pa.set_io_thread_count(max(1, min(resource_plan.cpu_workers, 32)))
    prefetcher = ShardPrefetcher(resource_plan.download_workers)
    for source in sources:
        name = source["name"]
        cursor_state = state.get("source_cursors", {}).get(name, {})
        file_index = int(cursor_state.get("file_index", 0))
        if not bool(cursor_state.get("exhausted", False)) and file_index < len(source_files[name]):
            prefetcher.submit(
                source,
                source_revisions[name],
                source_files[name][file_index],
                file_index,
                token,
                local_root / "sources" / name,
            )
    cursors = {
        source["name"]: SourceCursor(
            source,
            source_revisions[source["name"]],
            source_files[source["name"]],
            token,
            local_root / "sources",
            state.get("source_cursors", {}).get(source["name"], {}),
            int(output.get("buffer_batch_rows", 4096)),
            prefetcher,
        )
        for source in sources
    }
    exhausted = {name for name, cursor in cursors.items() if cursor.exhausted}
    output_writer = OutputShardWriter(
        local_root / "output",
        int(state.get("next_part", 0)),
        int(output.get("target_size_mb", 1024)) * 1024 * 1024,
        int(output.get("buffer_batch_rows", 4096)),
        int(output.get("max_documents", 1_000_000)),
    )
    checkpoint_dir = local_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    upload_pipeline = UploadPipeline(resource_plan.upload_workers)
    started = time.perf_counter()
    live = Live(
        mix_status_panel(sources, rows_by_source, tokens_by_source, rows, estimated_tokens,
                         target_tokens, "", "initializing"),
        console=console,
        refresh_per_second=2,
        auto_refresh=console.is_terminal or console.is_jupyter,
    )
    live.start()
    completed = False

    def commit_output(path: Path) -> None:
        nonlocal state
        next_part = int(state.get("next_part", 0))
        state_after = {
            "stage": 2,
            "mixer_version": MIXER_VERSION,
            "config_hash": config_hash,
            "source_revisions": source_revisions,
            "source_files": source_files,
            "name": cfg.get("name", "smol_mix"),
            "run_id": run_id,
            "rows": rows,
            "estimated_tokens": estimated_tokens,
            "rows_by_source": dict(rows_by_source),
            "tokens_by_source": dict(tokens_by_source),
            "source_cursors": {name: cursor.state() for name, cursor in cursors.items()},
            "next_part": next_part + 1,
            "last_part": next_part,
            "last_part_bytes": path.stat().st_size,
            "finished": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        remote_part = f"{output_prefix}/{filename_prefix}_shard_{next_part:05d}.parquet"
        checkpoint_path = checkpoint_dir / f"state-{next_part:05d}.json"
        upload_pipeline.submit(
            api,
            output["repo_id"],
            token,
            path,
            remote_part,
            checkpoint_path,
            remote_state,
            f"{output_prefix}/progress.json",
            state_after,
        )
        write_json(local_state, state_after)
        state = state_after

    try:
        last_status = started
        while True:
            if args.limit_rows is not None and rows >= int(args.limit_rows):
                break
            if target_tokens is not None and estimated_tokens >= int(target_tokens):
                break
            try:
                source = choose_source_by_token_debt(
                    sources,
                    tokens_by_source,
                    exhausted,
                    int(cfg.get("seed", 42)),
                    int(target_tokens) if target_tokens is not None else None,
                )
            except StopIteration:
                if target_tokens is not None and estimated_tokens < int(target_tokens):
                    missing = [name for name in cursors if name in exhausted]
                    raise RuntimeError(
                        f"Stage-1 sources exhausted before target token budget: {missing}"
                    )
                break

            name = source["name"]
            try:
                remaining_rows = None
                if args.limit_rows is not None:
                    remaining_rows = max(1, int(args.limit_rows) - rows)
                batch = cursors[name].next_batch(remaining_rows)
            except StopIteration:
                exhausted.add(name)
                continue
            if batch is None:
                exhausted.add(name)
                continue
            tokens = _batch_token_count(batch)
            batch_table = _prepare_output_batch(
                batch, source, cfg.get("name", "smol_mix"), total_weight
            )
            estimated_tokens += tokens
            batch_rows = int(batch.num_rows)
            rows += batch_rows
            rows_by_source[name] = int(rows_by_source.get(name, 0)) + batch_rows
            tokens_by_source[name] = int(tokens_by_source.get(name, 0)) + tokens
            cursors[name].add_tokens(tokens)
            part = output_writer.add_table(batch_table)

            now = time.perf_counter()
            if rows % max(2048, int(output.get("buffer_batch_rows", 4096))) == 0 or now - last_status >= 1.0:
                live.update(
                    mix_status_panel(sources, rows_by_source, tokens_by_source, rows,
                                     estimated_tokens, target_tokens, name, "mixing")
                )
                last_status = now
            if part is not None:
                commit_output(part)

        part = output_writer.close()
        if part is not None:
            commit_output(part)
        upload_pipeline.wait_all()

        completed = (
            args.limit_rows is None
            and (target_tokens is None or estimated_tokens >= int(target_tokens))
        )
        state.update({
            "rows": rows,
            "estimated_tokens": estimated_tokens,
            "rows_by_source": rows_by_source,
            "tokens_by_source": tokens_by_source,
            "source_cursors": {name: cursor.state() for name, cursor in cursors.items()},
            "finished": completed,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        write_json(local_state, state)

        if completed:
            # No output shard may exist when a tiny run has zero rows. Publish
            # the terminal state separately in that case.
            checkpoint_path = local_root / "checkpoint-final.json"
            write_json(checkpoint_path, state)
            api.create_commit(
                repo_id=output["repo_id"],
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=remote_state, path_or_fileobj=str(checkpoint_path)),
                    CommitOperationAdd(path_in_repo=f"{output_prefix}/progress.json", path_or_fileobj=str(checkpoint_path)),
                ],
                commit_message="Mark Stage-2 mix complete",
                token=token,
            )
            checkpoint_path.unlink(missing_ok=True)
            for cursor in cursors.values():
                cursor.close(remove_current=True)

        live.update(
            mix_status_panel(
                sources, rows_by_source, tokens_by_source, rows, estimated_tokens,
                target_tokens, "", "complete" if completed else "paused",
            )
        )
    finally:
        for cursor in cursors.values():
            cursor.close(remove_current=False)
        if upload_pipeline.pending:
            upload_pipeline.wait_all()
        prefetcher.close()
        live.stop()

    console.print(
        f"[bold green]Stage 2 {'complete' if completed else 'paused'}[/bold green] · "
        f"rows {rows:,} · estimated tokens {estimated_tokens:,}"
    )


if __name__ == "__main__":
    main()
