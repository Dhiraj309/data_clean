#!/usr/bin/env python3
"""Parallel, resumable Stage 1 filtering for Smol Data Parquet sources."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import chain, islice
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from smol_pipeline import (
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
    ShardWriter,
)


def file_key(filename: str) -> str:
    return filename.replace("/", "__").replace("\\", "__")


def output_prefix(path_prefix: str, domain: str) -> str:
    parts = [path_prefix.strip("/"), domain.strip("/")]
    return "/".join(part for part in parts if part)


def checkpoint_path(run_id: str, domain: str, source_file: str) -> str:
    return f"_checkpoints/stage1/{run_id}/{domain}/{file_key(source_file)}.json"


def empty_state() -> dict[str, Any]:
    return {
        "rows_seen": 0,
        "accepted": 0,
        "rejected": 0,
        "rejected_by_reason": {},
        "estimated_tokens": 0,
        "next_part": 0,
        "finished": False,
    }


def load_state(local_state: Path, repo_id: str, remote_state: str, revision: str, token: str) -> dict[str, Any]:
    if local_state.is_file():
        return json.loads(local_state.read_text(encoding="utf-8"))
    remote = download_json_if_present(repo_id, remote_state, revision, token)
    if remote:
        local_state.parent.mkdir(parents=True, exist_ok=True)
        write_json(local_state, remote)
        return remote
    return empty_state()


def save_state(
    state: dict[str, Any],
    local_state: Path,
    api: HfApi,
    repo_id: str,
    remote_state: str,
    token: str,
    part: Path | None = None,
) -> None:
    write_json(local_state, state)
    upload_file(api, repo_id, local_state, remote_state, token)
    if part is not None:
        part.unlink(missing_ok=True)


def upload_pending(
    pending_path: Path,
    state: dict[str, Any],
    local_state: Path,
    api: HfApi,
    repo_id: str,
    remote_state: str,
    token: str,
    remote_prefix: str,
) -> None:
    if not pending_path.is_file():
        return
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    part = Path(pending["local_part"])
    if state.get("next_part", 0) > pending["part_index"]:
        pending_path.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        return
    if not part.is_file():
        raise RuntimeError(f"Checkpoint references missing local buffer: {part}")
    source_index = int(pending["source_index"])
    part_index = int(pending["part_index"])
    shard_suffix = (
        f"{source_index:05d}"
        if part_index == 0
        else f"{source_index:05d}_{part_index:03d}"
    )
    remote_part = f"{remote_prefix}/{pending['domain']}_shard_{shard_suffix}.parquet"
    upload_file(api, repo_id, part, remote_part, token)
    state.update(pending["state_after"])
    save_state(state, local_state, api, repo_id, remote_state, token, part=part)
    pending_path.unlink(missing_ok=True)


def process_file(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["cfg"]
    source = cfg["source"]
    output = cfg["output"]
    source_name = cfg["name"]
    source_file = job["source_file"]
    source_index = int(job["source_index"])
    token = job["token"]
    api = HfApi(token=token)
    revision = output.get("revision", "main")
    run_id = str(output.get("run_id", "v1"))
    domain = str(output.get("domain", source_name.replace("_", "-"))).strip("/")
    path_prefix = str(output.get("path_prefix", ""))
    remote_prefix = output_prefix(path_prefix, domain)
    local_root = (
        Path(output.get("local_dir", "work/smol_stage1"))
        / source_name
        / run_id
        / file_key(source_file)
    )
    parts_dir = local_root / "parts"
    local_state = local_root / "state.json"
    pending_path = local_root / "pending.json"
    local_root.mkdir(parents=True, exist_ok=True)
    remote_state = checkpoint_path(run_id, domain, source_file)
    if job["resume"]:
        state = load_state(local_state, output["repo_id"], remote_state, revision, token)
    else:
        state = empty_state()
    upload_pending(
        pending_path,
        state,
        local_state,
        api,
        output["repo_id"],
        remote_state,
        token,
        remote_prefix,
    )
    if state.get("finished"):
        return state

    stream = iter(stream_hf_parquet_file(source, source_file, token))
    try:
        first = next(stream)
    except StopIteration:
        state["finished"] = True
        save_state(state, local_state, api, output["repo_id"], remote_state, token)
        return state
    available = sorted(first.keys()) if isinstance(first, dict) else []
    required = cfg.get("required_columns", [cfg.get("columns", {}).get("text", "text")])
    missing = [column for column in required if column not in available]
    if missing:
        raise RuntimeError(f"{source_file}: missing required columns {missing}; available={available}")

    writer = ShardWriter(
        parts_dir,
        target_size_mb=int(output.get("target_size_mb", 256)),
        max_documents=int(output.get("max_documents", 1_000_000)),
    )
    writer.index = int(state.get("next_part", 0))
    rejected = Counter(state.get("rejected_by_reason", {}))
    rows_seen = int(state.get("rows_seen", 0))
    accepted = int(state.get("accepted", 0))
    estimated_tokens = int(state.get("estimated_tokens", 0))
    limit_rows = job.get("limit_rows")
    started = time.perf_counter()

    rows = islice(chain([first], stream), rows_seen, None)
    for raw in rows:
        if limit_rows is not None and rows_seen >= int(limit_rows):
            break
        rows_seen += 1
        row = normalize_row(raw, cfg, source_name)
        ok, reason = accepts(row, cfg.get("filters", {}))
        if not ok:
            rejected[reason or "rejected"] += 1
            continue
        accepted += 1
        estimated_tokens += int(row["estimated_tokens"] or 1)
        part = writer.add(row)
        if part is None:
            continue
        part_index = writer.index - 1
        state_after = {
            "source": source,
            "source_file": source_file,
            "run_id": run_id,
            "rows_seen": rows_seen,
            "accepted": accepted,
            "rejected": sum(rejected.values()),
            "rejected_by_reason": dict(rejected),
            "estimated_tokens": estimated_tokens,
            "next_part": writer.index,
            "finished": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(
            pending_path,
            {
                "local_part": str(part),
                "part_index": part_index,
                "source_index": source_index,
                "domain": domain,
                "state_after": state_after,
            },
        )
        upload_pending(
            pending_path,
            state,
            local_state,
            api,
            output["repo_id"],
            remote_state,
            token,
            remote_prefix,
        )

    part = writer.flush()
    if part is not None:
        part_index = writer.index - 1
        state_after = {
            "source": source,
            "source_file": source_file,
            "run_id": run_id,
            "rows_seen": rows_seen,
            "accepted": accepted,
            "rejected": sum(rejected.values()),
            "rejected_by_reason": dict(rejected),
            "estimated_tokens": estimated_tokens,
            "next_part": writer.index,
            "finished": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(
            pending_path,
            {
                "local_part": str(part),
                "part_index": part_index,
                "source_index": source_index,
                "domain": domain,
                "state_after": state_after,
            },
        )
        upload_pending(
            pending_path,
            state,
            local_state,
            api,
            output["repo_id"],
            remote_state,
            token,
            remote_prefix,
        )

    state.update(
        {
            "source": source,
            "source_file": source_file,
            "run_id": run_id,
            "rows_seen": rows_seen,
            "accepted": accepted,
            "rejected": sum(rejected.values()),
            "rejected_by_reason": dict(rejected),
            "estimated_tokens": estimated_tokens,
            "next_part": writer.index,
            "finished": limit_rows is None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    save_state(state, local_state, api, output["repo_id"], remote_state, token)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--max-inflight-files", type=int, default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None, help="Per-file smoke-test row limit")
    parser.add_argument("--run-id", default=None, help="Override config run_id, useful for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if int(cfg.get("stage", -1)) != 1:
        raise ValueError("Stage-1 config must contain stage: 1")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    token = hf_token()
    api = HfApi(token=token)
    source_files = list(enumerate(list_hf_parquet_files(api, cfg["source"], token)))
    if args.limit_files is not None:
        source_files = source_files[: int(args.limit_files)]
    if not source_files:
        raise RuntimeError("No source files selected")

    if args.run_id:
        cfg = copy.deepcopy(cfg)
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["run_id"] = args.run_id
        cfg["output"]["path_prefix"] = f"_smoke/{args.run_id}"

    output = cfg["output"]
    domain = str(output.get("domain", cfg["name"].replace("_", "-"))).strip("/")
    remote_prefix = output_prefix(str(output.get("path_prefix", "")), domain)
    max_inflight = max(1, min(args.max_inflight_files or args.workers, len(source_files)))
    print(
        {
            "source": cfg["source"]["repo_id"],
            "files_selected": len(source_files),
            "first_files": [source_file for _, source_file in source_files[:5]],
            "output_repo": output["repo_id"],
            "output_folder": remote_prefix,
            "shard_pattern": f"{remote_prefix}/{domain}_shard_00000.parquet",
            "workers": args.workers,
            "max_inflight_files": max_inflight,
            "resume": not args.no_resume,
            "run_id": output.get("run_id", "v1"),
        }
    )
    if args.dry_run:
        return

    ensure_dataset_repo(api, output["repo_id"], token, private=bool(output.get("private", True)))
    jobs = [
        {
            "cfg": cfg,
            "source_file": source_file,
            "source_index": source_index,
            "token": token,
            "limit_rows": args.limit_rows,
            "resume": not args.no_resume,
        }
        for source_index, source_file in source_files
    ]
    results = []
    pending = {}
    job_iter = iter(jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(max_inflight):
            try:
                job = next(job_iter)
            except StopIteration:
                break
            pending[executor.submit(process_file, job)] = job["source_file"]
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                source_file = pending.pop(future)
                result = future.result()
                results.append(result)
                print({"source_file": source_file, "status": "processed", **result})
                try:
                    job = next(job_iter)
                except StopIteration:
                    continue
                pending[executor.submit(process_file, job)] = job["source_file"]
    progress = {
        "stage": 1,
        "name": cfg["name"],
        "domain": domain,
        "run_id": output.get("run_id", "v1"),
        "files_selected": len(source_files),
        "files_processed": len(results),
        "rows_seen": sum(int(result.get("rows_seen", 0)) for result in results),
        "accepted": sum(int(result.get("accepted", 0)) for result in results),
        "rejected": sum(int(result.get("rejected", 0)) for result in results),
        "estimated_tokens": sum(int(result.get("estimated_tokens", 0)) for result in results),
        "finished": all(bool(result.get("finished")) for result in results),
    }
    local_progress = (
        Path(output.get("local_dir", "work/smol_stage1"))
        / cfg["name"]
        / str(output.get("run_id", "v1"))
        / "progress.json"
    )
    write_json(local_progress, progress)
    upload_file(api, output["repo_id"], local_progress, f"{remote_prefix}/progress.json", token)
    print({**progress, "output_repo": output["repo_id"], "output_folder": remote_prefix})


if __name__ == "__main__":
    main()
