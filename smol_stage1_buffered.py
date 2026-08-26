#!/usr/bin/env python3
"""Parallel, resumable Stage 1 filtering with one rolling buffer per domain."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import ExitStack
from itertools import chain, islice
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from smol_pipeline import (
    OUTPUT_SCHEMA,
    ShardWriter,
    accepts,
    download_json_if_present,
    ensure_dataset_repo,
    hf_token,
    list_hf_parquet_files,
    load_config,
    normalize_row,
    stream_hf_parquet_file,
    upload_file,
    write_json,
)


def file_key(filename: str) -> str:
    return filename.replace("/", "__").replace("\\", "__")


def output_prefix(path_prefix: str, domain: str) -> str:
    return "/".join(part for part in (path_prefix.strip("/"), domain.strip("/")) if part)


def domain_checkpoint(run_id: str, domain: str) -> str:
    return f"_checkpoints/stage1/{run_id}/{domain}/progress.json"


def empty_domain_state() -> dict[str, Any]:
    return {
        "completed_sources": [], "next_shard": 0, "rows_seen": 0, "accepted": 0,
        "rejected": 0, "estimated_tokens": 0, "rejected_by_reason": {}, "finished": False,
    }


def publish_status(
    job: dict[str, Any], rows_seen: int, accepted: int, rejected: int, state: str,
    detail: str = "",
) -> None:
    shared = job.get("status")
    if shared is not None:
        shared[job["source_file"]] = {
            "file": job["source_file"], "source_index": int(job["source_index"]),
            "rows_seen": int(rows_seen), "accepted": int(accepted),
            "rejected": int(rejected),
            "acceptance": (100.0 * accepted / rows_seen) if rows_seen else 0.0,
            "state": state, "detail": detail,
        }


def status_panel(domain: str, statuses: Any, completed: int, total: int) -> Panel:
    active = sorted(
        dict(statuses).values(), key=lambda item: int(item.get("source_index", 0)),
    )
    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("File", ratio=5, overflow="ellipsis")
    table.add_column("Status", width=12)
    table.add_column("Seen", justify="right", width=12)
    table.add_column("Accepted", justify="right", width=12)
    table.add_column("Rejected", justify="right", width=12)
    table.add_column("Accept %", justify="right", width=10)
    table.add_column("Detail", ratio=2, overflow="ellipsis", no_wrap=True)
    if active:
        for item in active:
            table.add_row(
                str(item.get("file", "")), str(item.get("state", "processing")),
                f"{int(item.get('rows_seen', 0)):,}", f"{int(item.get('accepted', 0)):,}",
                f"{int(item.get('rejected', 0)):,}",
                f"{float(item.get('acceptance', 0.0)):.2f}%",
                str(item.get("detail", "")),
            )
    else:
        table.add_row("No files selected", "idle", "0", "0", "0", "0.00%", "")
    return Panel(
        table, title=f"Stage 1 · {domain}", subtitle=f"Completed files: {completed}/{total}",
        border_style="cyan",
    )


def local_work_root(cfg: dict[str, Any]) -> Path:
    """Return the resumable local root, optionally on persistent storage."""
    output = cfg["output"]
    persistent_root = os.environ.get("SMOL_PERSIST_ROOT")
    if persistent_root:
        return Path(persistent_root).expanduser() / cfg["name"] / str(output.get("run_id", "v1"))
    return Path(output.get("local_dir", "work/smol_stage1")) / cfg["name"] / str(output.get("run_id", "v1"))


def source_state_path(cfg: dict[str, Any], source_file: str) -> Path:
    return local_work_root(cfg) / "staging" / file_key(source_file) / "state.json"


def save_source_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, state)


def filter_source_file_streaming(job: dict[str, Any]) -> dict[str, Any]:
    """Filter one source file to local staging Parquet pieces; workers never upload."""
    cfg = job["cfg"]
    source = cfg["source"]
    source_file = job["source_file"]
    state_path = source_state_path(cfg, source_file)
    parts_dir = state_path.parent / "parts"
    state: dict[str, Any] = {}
    if job["resume"] and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        parts = [Path(value) for value in state.get("staging_parts", [])]
        if state.get("finished") and all(path.is_file() for path in parts):
            publish_status(
                job, int(state.get("rows_seen", 0)), int(state.get("accepted", 0)),
                int(state.get("rejected", 0)), "complete", "resumed",
            )
            return {**state, "result_status": "staged"}
        if any(not path.is_file() for path in parts):
            state = {}

    rows_seen = int(state.get("rows_seen", 0))
    accepted = int(state.get("accepted", 0))
    estimated_tokens = int(state.get("estimated_tokens", 0))
    rejected = Counter(state.get("rejected_by_reason", {}))
    staging_parts = list(state.get("staging_parts", []))
    writer = ShardWriter(
        parts_dir, target_size_mb=int(cfg["output"].get("staging_size_mb", 128)),
        max_documents=int(cfg["output"].get("staging_max_documents", 250_000)),
    )
    writer.index = int(state.get("next_part", len(staging_parts)))
    started = time.perf_counter()
    last_status = started
    publish_status(job, rows_seen, accepted, sum(rejected.values()), "loading", "opening source")

    stream = iter(stream_hf_parquet_file(source, source_file, job["token"]))
    try:
        first = next(stream)
    except StopIteration:
        first = None
    if first is not None:
        available = sorted(first.keys()) if isinstance(first, dict) else []
        required = cfg.get("required_columns", [cfg.get("columns", {}).get("text", "text")])
        missing = [column for column in required if column not in available]
        if missing:
            raise RuntimeError(f"{source_file}: missing required columns {missing}; available={available}")
        rows: Iterable[dict[str, Any]] = islice(chain([first], stream), rows_seen, None)
        for raw in rows:
            if job.get("limit_rows") is not None and rows_seen >= int(job["limit_rows"]):
                break
            rows_seen += 1
            row = normalize_row(raw, cfg, cfg["name"])
            ok, reason = accepts(row, cfg.get("filters", {}))
            part = None
            if ok:
                accepted += 1
                estimated_tokens += int(row["estimated_tokens"] or 1)
                part = writer.add(row)
            else:
                rejected[reason or "rejected"] += 1
            now = time.perf_counter()
            if rows_seen % 2048 == 0 or now - last_status >= 1.0:
                publish_status(job, rows_seen, accepted, sum(rejected.values()), "processing")
                last_status = now
            if part is not None:
                staging_parts.append(str(part))
                save_source_state(state_path, {
                    "source_file": source_file, "source_index": int(job["source_index"]),
                    "rows_seen": rows_seen, "accepted": accepted,
                    "rejected": sum(rejected.values()), "rejected_by_reason": dict(rejected),
                    "estimated_tokens": estimated_tokens, "next_part": writer.index,
                    "staging_parts": staging_parts, "finished": False,
                })

    part = writer.flush()
    if part is not None:
        staging_parts.append(str(part))
    result = {
        "source_file": source_file, "source_index": int(job["source_index"]),
        "rows_seen": rows_seen, "accepted": accepted,
        "rejected": sum(rejected.values()), "rejected_by_reason": dict(rejected),
        "estimated_tokens": estimated_tokens, "next_part": writer.index,
        "staging_parts": staging_parts, "finished": True,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    save_source_state(state_path, result)
    publish_status(job, rows_seen, accepted, sum(rejected.values()), "complete", "staged locally")
    return {**result, "result_status": "staged"}


_NORMALIZED_SOURCE_FIELDS = (
    "text", "id", "url", "language", "dataset", "language_score",
    "fasttext_score", "score", "answer_count", "accepted_answer_id",
    "int_score", "token_count",
)


def _parts_exist(result: dict[str, Any]) -> bool:
    return all(Path(value).is_file() for value in result.get("staging_parts", []))


def _row_group_dir(parts_dir: Path, row_group: int) -> Path:
    return parts_dir / f"row-group-{row_group:05d}"


def _load_row_group_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not result.get("finished") or not _parts_exist(result):
        return None
    return result


def _source_read_columns(cfg: dict[str, Any], available: list[str]) -> list[str]:
    """Read only fields used by normalization/filtering, avoiding large PDF metadata columns."""
    columns = cfg.get("columns", {})
    requested: list[str] = []
    for logical_name in _NORMALIZED_SOURCE_FIELDS:
        value = columns.get(logical_name, logical_name)
        if value:
            requested.append(str(value).split(".", 1)[0])
    for value in cfg.get("required_columns", []):
        requested.append(str(value).split(".", 1)[0])
    available_set = set(available)
    return list(dict.fromkeys(value for value in requested if value in available_set))


def _filter_row_group(task: dict[str, Any]) -> dict[str, Any]:
    """Filter one local Parquet row group into deterministic staging parts."""
    cfg = task["cfg"]
    row_group = int(task["row_group"])
    parts_dir = Path(task["parts_dir"])
    result_dir = _row_group_dir(parts_dir, row_group)
    result_path = result_dir / "result.json"
    if task.get("resume"):
        previous = _load_row_group_result(result_path)
        if previous is not None:
            return previous

    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(
        result_dir,
        target_size_mb=int(cfg["output"].get("staging_size_mb", 128)),
        max_documents=int(cfg["output"].get("staging_max_documents", 250_000)),
    )
    parquet = pq.ParquetFile(task["local_source"])
    rejected: Counter[str] = Counter()
    rows_seen = 0
    accepted = 0
    estimated_tokens = 0
    staging_parts: list[str] = []
    started = time.perf_counter()
    for batch in parquet.iter_batches(
        row_groups=[row_group],
        columns=task["read_columns"],
        batch_size=int(task["source_batch_rows"]),
        use_threads=False,
    ):
        for raw in batch.to_pylist():
            rows_seen += 1
            row = normalize_row(raw, cfg, cfg["name"])
            ok, reason = accepts(row, cfg.get("filters", {}))
            if not ok:
                rejected[reason or "rejected"] += 1
                continue
            accepted += 1
            estimated_tokens += int(row["estimated_tokens"] or 1)
            part = writer.add(row)
            if part is not None:
                staging_parts.append(str(part))
    part = writer.flush()
    if part is not None:
        staging_parts.append(str(part))
    result = {
        "row_group": row_group,
        "rows_seen": rows_seen,
        "accepted": accepted,
        "rejected": sum(rejected.values()),
        "rejected_by_reason": dict(rejected),
        "estimated_tokens": estimated_tokens,
        "staging_parts": staging_parts,
        "finished": True,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(result_path, result)
    return result


def _worker_process_ready() -> int:
    """Small startup probe used to create the process pool from the main thread."""
    return os.getpid()


def _select_process_start_method(
    configured: str | None, platform_name: str, notebook_run: bool,
) -> str:
    if notebook_run:
        if platform_name == "nt":
            raise RuntimeError(
                "Multiprocessing through notebook %run is unsupported on Windows. "
                "Run `python -u smol_stage1_filter.py ...` in a terminal instead."
            )
        return "fork"
    if configured:
        return configured
    return "spawn" if platform_name == "nt" else "forkserver"


def resolve_process_start_method(requested: str | None = None) -> str:
    """Choose a multiprocessing mode that also works with IPython ``%run``.

    Python 3.12's spawn/forkserver preparation reads ``__main__.__spec__``.
    IPython's ``%run`` main module may not define that attribute, so POSIX
    notebooks must use ``fork``. Normal CLI runs retain the safer forkserver
    default, while Windows continues to use spawn.
    """
    configured = requested or os.environ.get("SMOL_PROCESS_START_METHOD")
    main_module = sys.modules.get("__main__")
    notebook_run = main_module is not None and not hasattr(main_module, "__spec__")
    return _select_process_start_method(configured, os.name, notebook_run)


def _summarize_row_groups(
    job: dict[str, Any], row_group_results: dict[int, dict[str, Any]],
    total_row_groups: int, started: float,
) -> dict[str, Any]:
    rejected: Counter[str] = Counter()
    staging_parts: list[str] = []
    rows_seen = 0
    accepted = 0
    estimated_tokens = 0
    for row_group in sorted(row_group_results):
        result = row_group_results[row_group]
        rows_seen += int(result.get("rows_seen", 0))
        accepted += int(result.get("accepted", 0))
        estimated_tokens += int(result.get("estimated_tokens", 0))
        rejected.update(result.get("rejected_by_reason", {}))
        staging_parts.extend(str(value) for value in result.get("staging_parts", []))
    finished = len(row_group_results) == total_row_groups
    return {
        "mode": "row_groups",
        "source_file": job["source_file"],
        "source_index": int(job["source_index"]),
        "rows_seen": rows_seen,
        "accepted": accepted,
        "rejected": sum(rejected.values()),
        "rejected_by_reason": dict(rejected),
        "estimated_tokens": estimated_tokens,
        "next_part": len(staging_parts),
        "staging_parts": staging_parts,
        "total_row_groups": total_row_groups,
        "completed_row_groups": sorted(row_group_results),
        "row_group_results": {
            str(index): row_group_results[index] for index in sorted(row_group_results)
        },
        "finished": finished,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _download_source_file(job: dict[str, Any]) -> Path:
    cfg = job["cfg"]
    download_root = local_work_root(cfg) / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=cfg["source"]["repo_id"],
        filename=job["source_file"],
        repo_type="dataset",
        revision=cfg["source"].get("revision", "main"),
        token=job["token"],
        local_dir=str(download_root),
    )
    return Path(local)


def filter_source_file(job: dict[str, Any]) -> dict[str, Any]:
    """Filter one source using parallel Parquet row groups when safe to do so."""
    workers_per_file = max(1, int(job.get("workers_per_file", 1)))
    state_path = source_state_path(job["cfg"], job["source_file"])
    previous: dict[str, Any] = {}
    if job["resume"] and state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("finished") and _parts_exist(previous):
            publish_status(
                job, int(previous.get("rows_seen", 0)), int(previous.get("accepted", 0)),
                int(previous.get("rejected", 0)), "complete", "resumed",
            )
            return {**previous, "result_status": "staged"}

    # Preserve legacy partial checkpoints and exact smoke-test row limits.
    if (
        workers_per_file == 1
        or job.get("limit_rows") is not None
        or (previous and previous.get("mode") != "row_groups")
    ):
        return filter_source_file_streaming(job)

    started = time.perf_counter()
    publish_status(job, 0, 0, 0, "loading", "downloading source")
    local_source = _download_source_file(job)
    parquet = pq.ParquetFile(local_source)
    available = list(parquet.schema_arrow.names)
    required = job["cfg"].get(
        "required_columns", [job["cfg"].get("columns", {}).get("text", "text")],
    )
    missing = [
        value for value in required if str(value).split(".", 1)[0] not in set(available)
    ]
    if missing:
        raise RuntimeError(
            f"{job['source_file']}: missing required columns {missing}; available={sorted(available)}"
        )
    total_row_groups = parquet.num_row_groups
    parts_dir = state_path.parent / "parts"
    row_group_results: dict[int, dict[str, Any]] = {}
    if previous.get("mode") == "row_groups":
        for key, value in previous.get("row_group_results", {}).items():
            if value.get("finished") and _parts_exist(value):
                row_group_results[int(key)] = value
    for row_group in range(total_row_groups):
        recovered = _load_row_group_result(
            _row_group_dir(parts_dir, row_group) / "result.json",
        )
        if recovered is not None:
            row_group_results[row_group] = recovered

    summary = _summarize_row_groups(job, row_group_results, total_row_groups, started)
    publish_status(
        job, int(summary["rows_seen"]), int(summary["accepted"]), int(summary["rejected"]),
        "processing", f"row groups {len(row_group_results)}/{total_row_groups}",
    )
    missing_row_groups = [
        row_group for row_group in range(total_row_groups)
        if row_group not in row_group_results
    ]
    if missing_row_groups:
        read_columns = _source_read_columns(job["cfg"], available)
        max_workers = min(workers_per_file, len(missing_row_groups))
        tasks = [
            {
                "cfg": job["cfg"],
                "local_source": str(local_source),
                "parts_dir": str(parts_dir),
                "row_group": row_group,
                "read_columns": read_columns,
                "source_batch_rows": int(job.get("source_batch_rows", 1024)),
                "resume": job["resume"],
            }
            for row_group in missing_row_groups
        ]
        def process_tasks(executor: ProcessPoolExecutor) -> None:
            task_iter = iter(tasks)
            pending: dict[Any, int] = {}
            for _ in range(max_workers):
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                pending[executor.submit(_filter_row_group, task)] = int(task["row_group"])
            while pending:
                done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    row_group = pending.pop(future)
                    row_group_results[row_group] = future.result()
                    summary = _summarize_row_groups(
                        job, row_group_results, total_row_groups, started,
                    )
                    save_source_state(state_path, summary)
                    publish_status(
                        job, int(summary["rows_seen"]), int(summary["accepted"]),
                        int(summary["rejected"]), "processing",
                        f"row groups {len(row_group_results)}/{total_row_groups}",
                    )
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        continue
                    pending[executor.submit(_filter_row_group, task)] = int(task["row_group"])

        shared_executor = job.get("process_executor")
        if shared_executor is not None:
            process_tasks(shared_executor)
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=get_context("spawn"),
            ) as local_executor:
                process_tasks(local_executor)

    summary = _summarize_row_groups(job, row_group_results, total_row_groups, started)
    save_source_state(state_path, summary)
    del parquet
    if not job.get("keep_source_files"):
        local_source.unlink(missing_ok=True)
    publish_status(
        job, int(summary["rows_seen"]), int(summary["accepted"]), int(summary["rejected"]),
        "complete", f"staged {total_row_groups} row groups locally",
    )
    return {**summary, "result_status": "staged"}


def download_file_if_present(
    repo_id: str, filename: str, revision: str, token: str, destination: Path,
) -> bool:
    try:
        cached = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision, token=token,
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404 or "Entry Not Found" in str(exc):
            return False
        raise
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)
    return True


def iter_batches(paths: Iterable[Path], batch_rows: int) -> Iterable[pa.RecordBatch]:
    for path in paths:
        if path.is_file() and path.stat().st_size:
            yield from pq.ParquetFile(path).iter_batches(batch_size=batch_rows)


def build_buffer_transaction(
    current_buffer: Path,
    staging_parts: list[Path],
    transaction_dir: Path,
    domain: str,
    next_shard: int,
    target_bytes: int,
    batch_rows: int,
) -> tuple[Path, list[tuple[int, Path]], int]:
    """Merge one source into the remainder and roll near-target compressed shards."""
    if transaction_dir.exists():
        shutil.rmtree(transaction_dir)
    transaction_dir.mkdir(parents=True)
    completed: list[tuple[int, Path]] = []
    shard_index = next_shard
    active_path = transaction_dir / "active.parquet"
    writer: pq.ParquetWriter | None = None

    def close_full() -> None:
        nonlocal writer, shard_index
        if writer is None:
            return
        writer.close()
        final = transaction_dir / f"{domain}_shard_{shard_index:05d}.parquet"
        os.replace(active_path, final)
        completed.append((shard_index, final))
        shard_index += 1
        writer = None

    paths = ([current_buffer] if current_buffer.is_file() else []) + staging_parts
    for batch in iter_batches(paths, batch_rows):
        if writer is None:
            writer = pq.ParquetWriter(active_path, OUTPUT_SCHEMA, compression="zstd", compression_level=6)
        writer.write_batch(batch)
        if active_path.stat().st_size >= target_bytes:
            close_full()

    buffer_next = transaction_dir / "buffer.parquet"
    if writer is not None:
        writer.close()
        writer = None
        if active_path.stat().st_size >= target_bytes:
            final = transaction_dir / f"{domain}_shard_{shard_index:05d}.parquet"
            os.replace(active_path, final)
            completed.append((shard_index, final))
            shard_index += 1
            pq.write_table(pa.Table.from_batches([], schema=OUTPUT_SCHEMA), buffer_next, compression="zstd")
        else:
            os.replace(active_path, buffer_next)
    else:
        pq.write_table(pa.Table.from_batches([], schema=OUTPUT_SCHEMA), buffer_next, compression="zstd")
    return buffer_next, completed, shard_index


def load_domain_state(
    local_state: Path, repo_id: str, remote_state: str, revision: str, token: str, resume: bool,
) -> dict[str, Any]:
    """Load the newest usable checkpoint, preferring persistent local progress."""
    local: dict[str, Any] | None = None
    if local_state.is_file():
        candidate = json.loads(local_state.read_text(encoding="utf-8"))
        if "completed_sources" in candidate:
            local = candidate
    if not resume:
        return empty_domain_state()
    remote = download_json_if_present(repo_id, remote_state, revision, token)
    if remote is None:
        return local or empty_domain_state()
    if "completed_sources" not in remote:
        return local or empty_domain_state()
    local_completed = len((local or {}).get("completed_sources", []))
    remote_completed = len(remote.get("completed_sources", []))
    if local is not None and (
        bool(local.get("local_checkpoint")) or local_completed > remote_completed
    ):
        return local
    write_json(local_state, remote)
    return remote


def resolve_hybrid_layout(
    worker_budget: int,
    file_count: int,
    requested_active_files: int | None,
    requested_workers_per_file: int | None,
) -> tuple[int, int]:
    """Choose bounded file concurrency while keeping the process budget explicit."""
    if requested_active_files is None:
        if worker_budget < 8:
            active_files = 1
        elif worker_budget <= 48:
            active_files = 2
        else:
            active_files = min(4, max(2, worker_budget // 16))
    else:
        active_files = int(requested_active_files)
    if active_files < 1:
        raise ValueError("--active-files/--max-inflight-files must be positive")
    active_files = min(active_files, file_count, worker_budget)
    budget_per_file = max(1, worker_budget // active_files)
    workers_per_file = (
        budget_per_file
        if requested_workers_per_file is None
        else min(int(requested_workers_per_file), budget_per_file)
    )
    if workers_per_file < 1:
        raise ValueError("--workers-per-file must be positive")
    return active_files, workers_per_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)),
        help="Total row-group processing process budget",
    )
    parser.add_argument(
        "--active-files", type=int, default=None,
        help="Source files processed concurrently; chosen automatically when omitted",
    )
    parser.add_argument(
        "--workers-per-file", type=int, default=None,
        help="Maximum row-group workers assigned to each active source file",
    )
    parser.add_argument(
        "--max-inflight-files", type=int, default=None,
        help="Backward-compatible alias for --active-files",
    )
    parser.add_argument(
        "--source-batch-rows", type=int,
        default=int(os.environ.get("SMOL_SOURCE_BATCH_ROWS", "1024")),
        help="Rows converted at once inside each Parquet row-group worker",
    )
    parser.add_argument(
        "--keep-source-files", action="store_true",
        help="Retain locally downloaded source Parquet files after staging completes",
    )
    parser.add_argument(
        "--process-start-method",
        choices=("spawn", "forkserver", "fork"),
        default=None,
        help=(
            "Multiprocessing start method for the shared row-group worker pool; "
            "auto-selects fork for POSIX notebook %run sessions"
        ),
    )
    parser.add_argument(
        "--start-file", type=int, default=0,
        help="Zero-based offset into the sorted source file list",
    )
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None, help="Per-file smoke-test row limit")
    parser.add_argument("--run-id", default=None, help="Override config run_id, useful for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--remote-checkpoint-files", type=int, default=None,
        help="Upload the remote remainder/progress every N files; 0 means only full-shard/final checkpoints",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if int(cfg.get("stage", -1)) != 1:
        raise ValueError("Stage-1 config must contain stage: 1")
    if cfg.get("enabled", True) is False:
        reason = cfg.get("disabled_reason", "This source needs a dedicated processing path.")
        raise RuntimeError(f"Stage-1 config {cfg.get('name', args.config)!r} is disabled: {reason}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.source_batch_rows < 1:
        raise ValueError("--source-batch-rows must be positive")
    if args.start_file < 0:
        raise ValueError("--start-file must be non-negative")
    if args.limit_files is not None and args.limit_files < 1:
        raise ValueError("--limit-files must be positive")
    if args.run_id:
        cfg = copy.deepcopy(cfg)
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["run_id"] = args.run_id
        cfg["output"]["path_prefix"] = f"_smoke/{args.run_id}"

    token = hf_token()
    api = HfApi(token=token)
    all_source_files = list(enumerate(list_hf_parquet_files(api, cfg["source"], token)))
    end_file = (
        args.start_file + int(args.limit_files)
        if args.limit_files is not None else len(all_source_files)
    )
    source_files = all_source_files[args.start_file:end_file]
    if not source_files:
        raise RuntimeError("No source files selected")
    output = cfg["output"]
    console = Console()
    domain = str(output.get("domain", cfg["name"].replace("_", "-"))).strip("/")
    remote_prefix = output_prefix(str(output.get("path_prefix", "")), domain)
    target_size_mb = int(output.get("target_size_mb", 1024))
    remote_checkpoint_files = (
        int(args.remote_checkpoint_files)
        if args.remote_checkpoint_files is not None
        else int(os.environ.get("SMOL_REMOTE_CHECKPOINT_FILES", "0"))
    )
    if remote_checkpoint_files < 0:
        raise ValueError("--remote-checkpoint-files must be non-negative")
    env_active_files = os.environ.get("SMOL_ACTIVE_FILES")
    env_workers_per_file = os.environ.get("SMOL_WORKERS_PER_FILE")
    requested_active_files = (
        args.active_files
        if args.active_files is not None
        else args.max_inflight_files
        if args.max_inflight_files is not None
        else int(env_active_files)
        if env_active_files
        else None
    )
    requested_workers_per_file = (
        args.workers_per_file
        if args.workers_per_file is not None
        else int(env_workers_per_file)
        if env_workers_per_file
        else None
    )
    active_files, workers_per_file = resolve_hybrid_layout(
        args.workers, len(source_files), requested_active_files, requested_workers_per_file,
    )
    process_start_method = resolve_process_start_method(args.process_start_method)
    if args.dry_run:
        plan = Table(box=None, show_header=False, padding=(0, 1))
        plan.add_column("Field", style="cyan")
        plan.add_column("Value")
        plan.add_row("Source", cfg["source"]["repo_id"])
        plan.add_row("Files selected", f"{len(source_files):,}")
        plan.add_row("File range", f"{args.start_file}..{args.start_file + len(source_files) - 1}")
        plan.add_row("Output", f"{output['repo_id']}/{remote_prefix}")
        plan.add_row("Full shards", f"{domain}_shard_00000.parquet (~{target_size_mb / 1024:.2f} GiB)")
        plan.add_row("Remainder", f"{remote_prefix}/buffer.parquet")
        plan.add_row("Processing workers", str(args.workers))
        plan.add_row("Active files", str(active_files))
        plan.add_row("Workers per file", str(workers_per_file))
        plan.add_row("Source batch rows", f"{args.source_batch_rows:,}")
        plan.add_row("Keep source files", str(args.keep_source_files))
        plan.add_row("Process start", process_start_method)
        plan.add_row("Resume", str(not args.no_resume))
        console.print(Panel(plan, title=f"Stage 1 dry-run · {domain}", border_style="cyan"))
        return

    ensure_dataset_repo(api, output["repo_id"], token, private=bool(output.get("private", True)))
    run_id = str(output.get("run_id", "v1"))
    local_root = local_work_root(cfg)
    local_root.mkdir(parents=True, exist_ok=True)
    local_progress = local_root / "progress.json"
    local_buffer = local_root / "buffer.parquet"
    remote_state = domain_checkpoint(run_id, domain)
    state = load_domain_state(
        local_progress, output["repo_id"], remote_state, output.get("revision", "main"),
        token, not args.no_resume,
    )
    if not args.no_resume and state.get("completed_sources") and not state.get("local_checkpoint"):
        download_file_if_present(
            output["repo_id"], f"{remote_prefix}/buffer.parquet", output.get("revision", "main"),
            token, local_buffer,
        )
    pending_parts = [Path(value) for value in state.get("pending_parts", [])]
    missing_pending = [path for path in pending_parts if not path.is_file()]
    if missing_pending:
        raise RuntimeError(f"Persistent checkpoint references missing pending files: {missing_pending[:3]}")
    completed_sources = {int(value) for value in state.get("completed_sources", [])}
    selected_indices = [index for index, _ in source_files]
    remaining = [(index, filename) for index, filename in source_files if index not in completed_sources]
    if not remaining:
        console.print(f"[dim]Stage 1 is already complete for the {len(source_files)} selected files.[/dim]")
        return

    jobs = [{
        "cfg": cfg, "source_file": filename, "source_index": index, "token": token,
        "limit_rows": args.limit_rows, "resume": not args.no_resume,
        "workers_per_file": workers_per_file,
        "source_batch_rows": args.source_batch_rows,
        "keep_source_files": args.keep_source_files,
    } for index, filename in remaining]
    ordered_indices = [index for index, _ in remaining]
    ready: dict[int, dict[str, Any]] = {}
    next_order = 0

    with ExitStack() as stack:
        process_executor = stack.enter_context(ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context(process_start_method),
        ))
        # Start workers here, before file coordinator threads exist. This is
        # required for safe POSIX fork usage under IPython/Kaggle notebooks.
        process_executor.submit(_worker_process_ready).result()
        for job in jobs:
            job["process_executor"] = process_executor
        # Status is updated only by file coordinator threads, never subprocesses.
        statuses: dict[str, dict[str, Any]] = {}
        for job in jobs:
            job["status"] = statuses
        for job in jobs:
            statuses[job["source_file"]] = {
                "file": job["source_file"], "source_index": int(job["source_index"]),
                "rows_seen": 0, "accepted": 0, "rejected": 0, "acceptance": 0.0,
                "state": "queued", "detail": "waiting for worker",
            }
        pending: dict[Any, int] = {}
        job_iter = iter(jobs)
        with Live(
            status_panel(domain, statuses, len(completed_sources & set(selected_indices)), len(source_files)),
            console=console, refresh_per_second=2,
            auto_refresh=console.is_terminal or console.is_jupyter,
        ) as live:
            # File coordinators share one process budget and cap their own in-flight row groups.
            # The main thread remains the only rolling-buffer writer/uploader.
            with ThreadPoolExecutor(max_workers=active_files) as executor:
                for _ in range(active_files):
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        break
                    statuses[job["source_file"]] = {
                        "file": job["source_file"], "source_index": int(job["source_index"]),
                        "rows_seen": 0, "accepted": 0, "rejected": 0, "acceptance": 0.0,
                        "state": "queued", "detail": "waiting for worker",
                    }
                    pending[executor.submit(filter_source_file, job)] = int(job["source_index"])
                while pending or ready:
                    if pending:
                        done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        for future in done:
                            index = pending.pop(future)
                            ready[index] = future.result()
                            try:
                                job = next(job_iter)
                            except StopIteration:
                                continue
                            statuses[job["source_file"]] = {
                                "file": job["source_file"], "source_index": int(job["source_index"]),
                                "rows_seen": 0, "accepted": 0, "rejected": 0, "acceptance": 0.0,
                                "state": "queued", "detail": "waiting for worker",
                            }
                            pending[executor.submit(filter_source_file, job)] = int(job["source_index"])

                    while next_order < len(ordered_indices) and ordered_indices[next_order] in ready:
                        index = ordered_indices[next_order]
                        result = ready.pop(index)
                        source_file = str(result["source_file"])
                        statuses[source_file] = {
                            **dict(statuses.get(source_file, {})),
                            "state": "buffering", "detail": "merging into rolling buffer",
                        }
                        new_parts = [Path(value) for value in result.get("staging_parts", [])]
                        pending_parts.extend(new_parts)
                        pending_bytes = (
                            local_buffer.stat().st_size if local_buffer.is_file() else 0
                        ) + sum(path.stat().st_size for path in pending_parts)
                        rejected_counts = Counter(state.get("rejected_by_reason", {}))
                        rejected_counts.update(result.get("rejected_by_reason", {}))
                        completed_sources.add(index)
                        finished_after_source = all(value in completed_sources for value in selected_indices)
                        files_since_remote_checkpoint = int(state.get("files_since_remote_checkpoint", 0)) + 1
                        should_remote_checkpoint = (
                            pending_bytes >= target_size_mb * 1024 * 1024
                            or finished_after_source
                            or (
                                remote_checkpoint_files > 0
                                and files_since_remote_checkpoint >= remote_checkpoint_files
                            )
                        )
                        next_shard = int(state.get("next_shard", 0))
                        state.update({
                            "stage": 1, "name": cfg["name"], "domain": domain, "run_id": run_id,
                            "completed_sources": sorted(completed_sources), "next_shard": next_shard,
                            "rows_seen": int(state.get("rows_seen", 0)) + int(result.get("rows_seen", 0)),
                            "accepted": int(state.get("accepted", 0)) + int(result.get("accepted", 0)),
                            "rejected": int(state.get("rejected", 0)) + int(result.get("rejected", 0)),
                            "estimated_tokens": int(state.get("estimated_tokens", 0)) + int(result.get("estimated_tokens", 0)),
                            "rejected_by_reason": dict(rejected_counts),
                            "buffer_bytes": pending_bytes,
                            "pending_parts": [str(path) for path in pending_parts],
                            "finished": finished_after_source,
                            "files_since_remote_checkpoint": files_since_remote_checkpoint,
                            "local_checkpoint": True,
                        })
                        if should_remote_checkpoint:
                            current_buffer = local_buffer if local_buffer.is_file() else local_root / ".aggregate" / "no-current-buffer"
                            buffer_next, shards, next_shard = build_buffer_transaction(
                                current_buffer, pending_parts, local_root / ".aggregate", domain,
                                next_shard, target_size_mb * 1024 * 1024,
                                int(output.get("buffer_batch_rows", 4096)),
                            )
                            upload_remainder = finished_after_source or (
                                remote_checkpoint_files > 0
                                and files_since_remote_checkpoint >= remote_checkpoint_files
                            )
                            statuses[source_file] = {
                                **dict(statuses.get(source_file, {})),
                                "state": "uploading",
                                "detail": (
                                    f"uploading {len(shards)} full shard(s)"
                                    + (" and remainder checkpoint" if upload_remainder else "; keeping remainder local")
                                ),
                            }
                            live.update(status_panel(domain, statuses, len(completed_sources & set(selected_indices)), len(source_files)))
                            for shard_index, shard_path in shards:
                                remote_part = f"{remote_prefix}/{domain}_shard_{shard_index:05d}.parquet"
                                upload_file(api, output["repo_id"], shard_path, remote_part, token)
                            state.update({
                                "next_shard": next_shard,
                                "buffer_bytes": buffer_next.stat().st_size,
                                "pending_parts": [],
                            })
                            if upload_remainder:
                                upload_file(api, output["repo_id"], buffer_next, f"{remote_prefix}/buffer.parquet", token)
                                state.update({
                                    "files_since_remote_checkpoint": 0,
                                    "local_checkpoint": False,
                                    "remote_completed_sources": sorted(completed_sources),
                                })
                                transaction_progress = local_root / ".aggregate" / "progress.json"
                                write_json(transaction_progress, state)
                                upload_file(api, output["repo_id"], transaction_progress, remote_state, token)
                                upload_file(api, output["repo_id"], transaction_progress, f"{remote_prefix}/progress.json", token)
                                checkpoint_detail = "remote checkpoint saved"
                            else:
                                state["local_checkpoint"] = True
                                checkpoint_detail = "full shard uploaded; remainder kept locally"
                            os.replace(buffer_next, local_buffer)
                            for path in pending_parts:
                                path.unlink(missing_ok=True)
                            pending_parts.clear()
                        else:
                            checkpoint_detail = "saved locally; remote checkpoint pending"
                        write_json(local_progress, state)
                        statuses[source_file] = {
                            **dict(statuses.get(source_file, {})),
                            "state": "complete", "detail": checkpoint_detail,
                        }
                        next_order += 1
                    live.update(status_panel(domain, statuses, len(completed_sources & set(selected_indices)), len(source_files)))

    acceptance = 100.0 * int(state["accepted"]) / int(state["rows_seen"]) if state.get("rows_seen") else 0.0
    console.print(
        f"[bold green]Stage 1 complete[/bold green] · {domain} · "
        f"accepted {int(state['accepted']):,}/{int(state['rows_seen']):,} ({acceptance:.2f}%) · "
        f"rejected {int(state.get('rejected', 0)):,} · "
        f"full shards {int(state['next_shard']):,} · buffer {int(state.get('buffer_bytes', 0)) / 1024**2:.1f} MiB"
    )


if __name__ == "__main__":
    main()
