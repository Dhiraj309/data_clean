#!/usr/bin/env python3
"""Stage 3: resumable, parallel tokenization of mixed Smol Parquet data.

The tokenizer workers process independent Parquet files in parallel.  The main
process consumes their temporary token streams in source order, writes exact
token-count binary shards, and owns all Hub checkpoint updates.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
)
from smol_pipeline import (
    download_json_if_present,
    ensure_dataset_repo,
    hf_token,
    load_config,
    stable_id,
)


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

DEFAULT_BATCH_SIZE = 10_000
DEFAULT_SHARD_SIZE_TOKENS = 250_000_000
DEFAULT_MAX_DOC_TOKENS = 8_192
DEFAULT_HARD_DOC_LIMIT = 65_536
DEFAULT_IO_CHUNK_TOKENS = 8_388_608

_WORKER_TOKENIZER: Any = None
_WORKER_TOKENIZER_ID = ""


def _load_tokenizer(tokenizer_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=True,
        model_max_length=10_000_000,
    )


def configure_tokenizer_parallelism(threads: int) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(max(1, int(threads)))


def tokenizer_info(tokenizer_id: str, tokenizer: Any) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    if not vocab:
        raise RuntimeError(f"Tokenizer has an empty vocabulary: {tokenizer_id}")
    max_token_id = max(int(value) for value in vocab.values())
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {tokenizer_id!r} has no eos_token_id")
    dtype = "uint16" if max_token_id <= np.iinfo(np.uint16).max else "uint32"
    if int(eos_id) > np.iinfo(np.dtype(dtype)).max:
        raise ValueError(f"EOS token {eos_id} does not fit {dtype}")
    return {
        "tokenizer_id": tokenizer_id,
        "vocab_size": int(len(tokenizer)),
        "max_token_id": max_token_id,
        "eos_token_id": int(eos_id),
        "dtype": dtype,
    }


def _tokenize_texts(
    texts: list[Any],
    tokenizer: Any,
    eos_id: int,
    *,
    max_doc_tokens: int,
    hard_doc_limit: int,
) -> tuple[list[int], int, int]:
    raw_count = len(texts)
    valid = [value for value in texts if isinstance(value, str) and len(value) > 10]
    skipped = raw_count - len(valid)
    if not valid:
        return [], 0, skipped

    encoded = tokenizer(
        valid,
        add_special_tokens=False,
        truncation=True,
        max_length=hard_doc_limit,
    )
    output: list[int] = []
    for ids in encoded["input_ids"]:
        for start in range(0, len(ids), max_doc_tokens):
            chunk = ids[start : start + max_doc_tokens]
            if chunk:
                output.extend(int(value) for value in chunk)
                output.append(int(eos_id))
    return output, len(valid), skipped


def _worker_init(tokenizer_id: str, tokenizer_threads: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_TOKENIZER_ID
    configure_tokenizer_parallelism(tokenizer_threads)
    _WORKER_TOKENIZER = _load_tokenizer(tokenizer_id)
    if not _WORKER_TOKENIZER.is_fast:
        raise RuntimeError(f"Tokenizer is not a fast tokenizer: {tokenizer_id}")
    _WORKER_TOKENIZER_ID = tokenizer_id


def _tokenize_file_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Download and tokenize one source file into an atomic local binary file."""
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized")

    output_path = Path(task["output_path"])
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    local_source = hf_hub_download(
        repo_id=task["repo_id"],
        filename=task["filename"],
        repo_type="dataset",
        revision=task["revision"],
        token=task["token"],
    )
    parquet = pq.ParquetFile(local_source)
    available = set(parquet.schema.names)
    text_column = str(task["text_column"])
    if text_column not in available:
        raise KeyError(
            f"Text column {text_column!r} is missing from {task['filename']}; "
            f"available columns: {sorted(available)}"
        )

    rows_seen = 0
    valid_rows = 0
    skipped_rows = 0
    token_count = 0
    dtype = np.dtype(task["dtype"])
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("wb") as output:
            for batch in parquet.iter_batches(
                batch_size=int(task["batch_size"]),
                columns=[text_column],
                use_threads=True,
            ):
                texts = batch.column(text_column).to_pylist()
                rows_seen += len(texts)
                tokens, valid, skipped = _tokenize_texts(
                    texts,
                    _WORKER_TOKENIZER,
                    int(task["eos_id"]),
                    max_doc_tokens=int(task["max_doc_tokens"]),
                    hard_doc_limit=int(task["hard_doc_limit"]),
                )
                valid_rows += valid
                skipped_rows += skipped
                token_count += len(tokens)
                if tokens:
                    np.asarray(tokens, dtype=dtype).tofile(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise

    return {
        "filename": task["filename"],
        "output_path": str(output_path),
        "rows_seen": rows_seen,
        "valid_rows": valid_rows,
        "skipped_rows": skipped_rows,
        "tokens": token_count,
    }


def _source_files(api: HfApi, source: dict[str, Any], token: str) -> tuple[str, list[str]]:
    repo_id = str(source["repo_id"])
    requested_revision = str(source.get("revision", "main"))
    info = api.repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=requested_revision,
        token=token,
    )
    revision = str(getattr(info, "sha", None) or requested_revision)
    prefix = str(source.get("path_prefix", "")).strip("/")
    prefix_path = f"{prefix}/" if prefix else ""
    all_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
    )

    progress_file = source.get("progress_file")
    filename_prefix = str(source.get("filename_prefix", ""))
    if progress_file and filename_prefix:
        progress_path = prefix_path + str(progress_file).lstrip("/")
        progress = download_json_if_present(repo_id, progress_path, revision, token)
        if progress is None:
            raise RuntimeError(f"Missing source progress file: {repo_id}:{progress_path}")
        if not bool(progress.get("finished", False)):
            raise RuntimeError(f"Source Stage 2 run is not complete: {repo_id}:{progress_path}")
        count = int(progress.get("next_part", progress.get("total_shards", 0)))
        files = [
            f"{prefix_path}{filename_prefix}_shard_{index:05d}.parquet"
            for index in range(count)
        ]
        missing = [filename for filename in files if filename not in all_files]
        if missing:
            raise RuntimeError(f"Progress references missing source files: {missing[:3]}")
    else:
        pattern = str(source.get("file_pattern", "*.parquet"))
        regex = re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")
        files = [
            filename
            for filename in all_files
            if filename.startswith(prefix_path)
            and filename.lower().endswith(".parquet")
            and regex.match(filename.rsplit("/", 1)[-1])
            and filename.rsplit("/", 1)[-1] != "buffer.parquet"
        ]
        files.sort()

    if not files:
        raise RuntimeError(f"No source Parquet files found in {repo_id}:{prefix_path}")
    return revision, files


def _default_state(config_hash: str, revision: str, files: list[str], info: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": 3,
        "version": 1,
        "config_hash": config_hash,
        "source_revision": revision,
        "source_files": files,
        "tokenizer_info": info,
        "files_done": [],
        "current_file": None,
        "current_token_offset": 0,
        "total_tokens": 0,
        "total_rows": 0,
        "valid_rows": 0,
        "skipped_rows": 0,
        "next_shard": 0,
        "buffer_tokens": 0,
        "finished": False,
        "updated_at": time.time(),
    }


def _validate_state(state: dict[str, Any], config_hash: str, revision: str, files: list[str], info: dict[str, Any]) -> None:
    if state.get("config_hash") != config_hash:
        raise RuntimeError("Stage-3 checkpoint does not match the current config/source manifest")
    if state.get("source_revision") != revision:
        raise RuntimeError("Stage-3 checkpoint points to a different source revision")
    if state.get("source_files") != files:
        raise RuntimeError("Stage-3 checkpoint source file list changed")
    if state.get("tokenizer_info") != info:
        raise RuntimeError("Stage-3 checkpoint tokenizer metadata does not match")


def _load_state(
    local_path: Path,
    remote_repo: str,
    remote_path: str,
    token: str,
    config_hash: str,
    revision: str,
    files: list[str],
    info: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if local_path.is_file():
        candidates.append(json.loads(local_path.read_text(encoding="utf-8")))
    remote = download_json_if_present(remote_repo, remote_path, "main", token)
    if remote is not None:
        candidates.append(remote)
    if not candidates:
        return _default_state(config_hash, revision, files, info)
    state = max(candidates, key=lambda item: float(item.get("updated_at", 0)))
    _validate_state(state, config_hash, revision, files, info)
    return state


def _save_local_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_commit(
    api: HfApi,
    repo_id: str,
    token: str,
    output_prefix: str,
    remote_checkpoint: str,
    local_checkpoint: Path,
    local_buffer: Path,
    state: dict[str, Any],
) -> None:
    progress_path = f"{output_prefix}/progress.json"
    buffer_path = f"{output_prefix}/buffer.bin"
    operations = [
        CommitOperationAdd(path_in_repo=remote_checkpoint, path_or_fileobj=str(local_checkpoint)),
        CommitOperationAdd(path_in_repo=progress_path, path_or_fileobj=str(local_checkpoint)),
        CommitOperationAdd(path_in_repo=buffer_path, path_or_fileobj=str(local_buffer)),
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            f"Stage-3 checkpoint: {len(state['files_done'])} files, "
            f"{int(state['total_tokens']):,} tokens"
        ),
        token=token,
    )


def _upload_binary_shard(
    api: HfApi,
    repo_id: str,
    token: str,
    output_prefix: str,
    filename_prefix: str,
    shard_index: int,
    local_path: Path,
) -> None:
    remote_path = f"{output_prefix}/{filename_prefix}_shard_{shard_index:05d}.bin"
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Add {remote_path}",
    )
    print(f"Pushed {remote_path} ({local_path.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def _resource_plan(cpu_count: int, requested_workers: int | None, requested_threads: int | None) -> tuple[int, int]:
    workers = int(requested_workers or min(16, max(1, cpu_count // 8)))
    workers = max(1, min(workers, cpu_count))
    threads = int(requested_threads or max(1, cpu_count // workers))
    threads = max(1, threads)
    return workers, threads


def select_file_range(files: list[str], start_file: int, max_files: int | None) -> list[str]:
    if start_file < 0:
        raise ValueError("start_file must be non-negative")
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be positive when provided")
    end = start_file + max_files if max_files is not None else len(files)
    return files[start_file:end]


def _print_status(
    *,
    files_done: int,
    total_files: int,
    current_file: str | None,
    rows: int,
    valid_rows: int,
    skipped_rows: int,
    tokens: int,
    buffer_tokens: int,
    workers: int,
    tokenizer_threads: int,
    started: float,
) -> None:
    elapsed = max(time.time() - started, 1e-9)
    speed = tokens / elapsed
    print(
        f"Stage 3 | files {files_done}/{total_files} | "
        f"current {current_file or 'idle'} | rows {rows:,} | "
        f"accepted {valid_rows:,} | skipped {skipped_rows:,} | "
        f"tokens {tokens:,} | buffer {buffer_tokens:,} | "
        f"speed {speed:,.0f} tok/s | workers {workers}×{tokenizer_threads}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-file", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--tokenizer-threads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shard-size-tokens", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--remote-checkpoint-every-files", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if int(cfg.get("stage", -1)) != 3:
        raise ValueError("Stage-3 config must contain stage: 3")
    source = dict(cfg.get("source") or {})
    output = dict(cfg.get("output") or {})
    tokenizer_cfg = dict(cfg.get("tokenizer") or {})
    if not source or not output or not tokenizer_cfg.get("id"):
        raise ValueError("Stage-3 config requires source, output, and tokenizer.id")
    for value, name in ((args.start_file, "--start-file"), (args.max_files, "--max-files"), (args.target_tokens, "--target-tokens")):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")

    token = hf_token()
    api = HfApi(token=token)
    ensure_dataset_repo(api, output["repo_id"], token, private=bool(output.get("private", True)))

    tokenizer_id = str(tokenizer_cfg["id"])
    configure_tokenizer_parallelism(int(tokenizer_cfg.get("threads", args.tokenizer_threads or 1)))
    tokenizer = _load_tokenizer(tokenizer_id)
    if not tokenizer.is_fast:
        raise RuntimeError("Stage 3 requires a fast tokenizer")
    info = tokenizer_info(tokenizer_id, tokenizer)
    eos_id = int(info["eos_token_id"])

    revision, all_files = _source_files(api, source, token)
    config_hash = stable_id({"config": cfg, "source_revision": revision, "source_files": all_files, "tokenizer": info})
    run_id = str(args.run_id or output.get("run_id", "v1"))
    local_root = Path(output.get("local_dir", "work/smol_stage3")) / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    local_state_path = local_root / "state.json"
    local_buffer = local_root / "buffer.bin"
    local_buffer.touch(exist_ok=True)
    remote_checkpoint = f"_checkpoints/stage3/{run_id}.json"
    output_prefix = str(output.get("path_prefix", "laughlm-v1")).strip("/")
    filename_prefix = str(output.get("filename_prefix", cfg.get("name", "laughlm-v1")))
    if args.run_id:
        output_prefix = f"_smoke/{run_id}/{output_prefix}"

    state = _default_state(config_hash, revision, all_files, info)
    if not args.no_resume:
        state = _load_state(local_state_path, output["repo_id"], remote_checkpoint, token, config_hash, revision, all_files, info)
    elif local_state_path.exists():
        raise RuntimeError("--no-resume cannot overwrite an existing local Stage-3 checkpoint")

    dtype = np.dtype(info["dtype"])
    shard_size = int(args.shard_size_tokens or output.get("shard_size_tokens", DEFAULT_SHARD_SIZE_TOKENS))
    batch_size = int(args.batch_size or output.get("batch_size", DEFAULT_BATCH_SIZE))
    max_doc_tokens = int(tokenizer_cfg.get("max_doc_tokens", DEFAULT_MAX_DOC_TOKENS))
    hard_doc_limit = int(tokenizer_cfg.get("hard_doc_limit", DEFAULT_HARD_DOC_LIMIT))
    io_chunk_tokens = int(output.get("io_chunk_tokens", DEFAULT_IO_CHUNK_TOKENS))
    checkpoint_every = int(
        args.remote_checkpoint_every_files
        if args.remote_checkpoint_every_files is not None
        else output.get("remote_checkpoint_every_files", 5)
    )
    if shard_size <= 0 or batch_size <= 0 or io_chunk_tokens <= 0 or checkpoint_every <= 0:
        raise ValueError("shard size, batch size, io chunk size, and checkpoint interval must be positive")

    selected_files = select_file_range(all_files, args.start_file, args.max_files)
    if not selected_files:
        raise RuntimeError("No source files selected")

    workers, tokenizer_threads = _resource_plan(os.cpu_count() or 1, args.workers, args.tokenizer_threads)
    workers = min(workers, len(selected_files))
    print(
        f"Stage 3 tokenization | source={source['repo_id']} | files={len(selected_files)}/{len(all_files)} | "
        f"tokenizer={tokenizer_id} | workers={workers}×{tokenizer_threads} | "
        f"shard={shard_size:,} tokens | batch={batch_size:,}",
        flush=True,
    )

    # The local buffer is the durable residual token stream.  Reconcile a full
    # buffer after a crash so a shard upload cannot be duplicated in content.
    actual_buffer_tokens = local_buffer.stat().st_size // dtype.itemsize if local_buffer.exists() else 0
    if int(state.get("buffer_tokens", 0)) != actual_buffer_tokens:
        if state.get("current_file") is not None:
            raise RuntimeError(
                f"Local buffer/state mismatch: file has {actual_buffer_tokens} tokens, "
                f"checkpoint says {state.get('buffer_tokens', 0)}"
            )
        state["buffer_tokens"] = actual_buffer_tokens
    if actual_buffer_tokens >= shard_size:
        if actual_buffer_tokens != shard_size:
            raise RuntimeError("Local buffer contains more than one uncheckpointed full shard")
        local_shard = local_root / f"recovery_{int(state['next_shard']):05d}.bin"
        os.replace(local_buffer, local_shard)
        _upload_binary_shard(api, output["repo_id"], token, output_prefix, filename_prefix, int(state["next_shard"]), local_shard)
        local_shard.unlink(missing_ok=True)
        state["next_shard"] = int(state["next_shard"]) + 1
        state["buffer_tokens"] = 0
        _save_local_state(local_state_path, state)
        local_buffer.touch()

    token_dir = local_root / "file_tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    files_done = set(str(value) for value in state.get("files_done", []))
    pending_files = [filename for filename in selected_files if filename not in files_done]
    started = time.time()
    run_rows = run_valid = run_skipped = run_tokens = 0

    def make_task(filename: str) -> dict[str, Any]:
        safe_name = stable_id(filename)
        return {
            "repo_id": source["repo_id"],
            "revision": revision,
            "filename": filename,
            "token": token,
            "text_column": source.get("text_column", "text"),
            "batch_size": batch_size,
            "max_doc_tokens": max_doc_tokens,
            "hard_doc_limit": hard_doc_limit,
            "eos_id": eos_id,
            "dtype": info["dtype"],
            "output_path": str(token_dir / f"{safe_name}.bin"),
        }

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(tokenizer_id, tokenizer_threads),
    ) as executor:
        futures: dict[str, Future[dict[str, Any]]] = {}
        next_submit = 0

        def fill_window() -> None:
            nonlocal next_submit
            while next_submit < len(pending_files) and len(futures) < workers:
                filename = pending_files[next_submit]
                futures[filename] = executor.submit(_tokenize_file_worker, make_task(filename))
                next_submit += 1

        fill_window()
        try:
            for filename in pending_files:
                result = futures.pop(filename)
                file_result = result.result()
                fill_window()
                token_path = Path(file_result["output_path"])
                previous_current_file = state.get("current_file")
                offset = (
                    int(state.get("current_token_offset", 0))
                    if previous_current_file == filename else 0
                )
                state["current_file"] = filename
                if offset:
                    expected_bytes = offset * dtype.itemsize
                    if token_path.stat().st_size < expected_bytes:
                        raise RuntimeError(f"Checkpoint offset exceeds token temp file for {filename}")

                with token_path.open("rb") as token_stream:
                    token_stream.seek(offset * dtype.itemsize)
                    while True:
                        if args.target_tokens is not None and int(state["total_tokens"]) >= args.target_tokens:
                            break
                        remaining = shard_size - int(state["buffer_tokens"])
                        count = min(io_chunk_tokens, remaining)
                        chunk = np.fromfile(token_stream, dtype=dtype, count=count)
                        if chunk.size == 0:
                            break
                        with local_buffer.open("ab") as buffer_stream:
                            chunk.tofile(buffer_stream)
                            buffer_stream.flush()
                            os.fsync(buffer_stream.fileno())
                        consumed = int(chunk.size)
                        offset += consumed
                        state["current_token_offset"] = offset
                        state["buffer_tokens"] = int(state["buffer_tokens"]) + consumed
                        state["total_tokens"] = int(state["total_tokens"]) + consumed
                        _save_local_state(local_state_path, state)

                        if int(state["buffer_tokens"]) == shard_size:
                            local_shard = local_root / f"{filename_prefix}_shard_{int(state['next_shard']):05d}.bin"
                            os.replace(local_buffer, local_shard)
                            _upload_binary_shard(api, output["repo_id"], token, output_prefix, filename_prefix, int(state["next_shard"]), local_shard)
                            local_shard.unlink(missing_ok=True)
                            local_buffer.touch()
                            state["next_shard"] = int(state["next_shard"]) + 1
                            state["buffer_tokens"] = 0
                            _save_local_state(local_state_path, state)

                    if args.target_tokens is not None and int(state["total_tokens"]) >= args.target_tokens:
                        break
                    if offset < int(file_result["tokens"]):
                        raise RuntimeError(f"Tokenization stopped before consuming {filename}")

                run_rows += int(file_result["rows_seen"])
                run_valid += int(file_result["valid_rows"])
                run_skipped += int(file_result["skipped_rows"])
                run_tokens += int(file_result["tokens"])
                state["total_rows"] = int(state["total_rows"]) + int(file_result["rows_seen"])
                state["valid_rows"] = int(state["valid_rows"]) + int(file_result["valid_rows"])
                state["skipped_rows"] = int(state["skipped_rows"]) + int(file_result["skipped_rows"])
                state["files_done"] = list(dict.fromkeys([*state.get("files_done", []), filename]))
                files_done.add(filename)
                state["current_file"] = None
                state["current_token_offset"] = 0
                state["finished"] = len(files_done) == len(all_files)
                _save_local_state(local_state_path, state)

                if len(files_done) % checkpoint_every == 0 or state["finished"]:
                    _checkpoint_commit(
                        api,
                        output["repo_id"],
                        token,
                        output_prefix,
                        remote_checkpoint,
                        local_state_path,
                        local_buffer,
                        state,
                    )
                token_path.unlink(missing_ok=True)
                _print_status(
                    files_done=len(files_done),
                    total_files=len(all_files),
                    current_file=None,
                    rows=state["total_rows"],
                    valid_rows=state["valid_rows"],
                    skipped_rows=state["skipped_rows"],
                    tokens=state["total_tokens"],
                    buffer_tokens=state["buffer_tokens"],
                    workers=workers,
                    tokenizer_threads=tokenizer_threads,
                    started=started,
                )
        finally:
            for future in futures.values():
                future.cancel()

    if state.get("finished"):
        _save_local_state(local_state_path, state)
        if not (len(files_done) % checkpoint_every == 0):
            _checkpoint_commit(api, output["repo_id"], token, output_prefix, remote_checkpoint, local_state_path, local_buffer, state)

    print(
        f"Stage 3 {'complete' if state.get('finished') else 'paused'} | "
        f"files={len(files_done)}/{len(all_files)} | tokens={int(state['total_tokens']):,} | "
        f"buffer={int(state['buffer_tokens']):,} | elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
