#!/usr/bin/env python3
"""Resumable Stage 2 weighted mixing of Stage-1 Parquet repositories."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from smol_pipeline import (
    ShardWriter,
    download_json_if_present,
    ensure_dataset_repo,
    hf_token,
    load_config,
    stream_parquet_prefix,
    upload_file,
    write_json,
)
from smol_stage1_buffered import build_buffer_transaction, download_file_if_present


def choose_source(rng: random.Random, sources: list[dict], exhausted: set[str]) -> dict:
    available = [source for source in sources if source["name"] not in exhausted]
    if not available:
        raise StopIteration
    total = sum(float(source["weight"]) for source in available)
    point = rng.random() * total
    for source in available:
        point -= float(source["weight"])
        if point <= 0:
            return source
    return available[-1]


def empty_state(sources: list[dict]) -> dict[str, Any]:
    return {
        "rows": 0,
        "estimated_tokens": 0,
        "rows_by_source": {source["name"]: 0 for source in sources},
        "next_part": 0,
        "finished": False,
    }


def mix_status_panel(
    sources: list[dict],
    counts: dict[str, int],
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
    table.add_column("Actual %", justify="right", width=12)
    table.add_column("Target %", justify="right", width=12)
    total_weight = sum(float(source["weight"]) for source in sources)
    for source in sources:
        count = int(counts.get(source["name"], 0))
        actual = (100.0 * count / rows) if rows else 0.0
        target = 100.0 * float(source["weight"]) / total_weight
        table.add_row(source["name"], f"{count:,}", f"{actual:.2f}%", f"{target:.2f}%")
    return Panel(Group(summary, table), title="Stage 2 · weighted mixture", border_style="magenta")


def load_state(local_state: Path, repo_id: str, remote_state: str, token: str, sources: list[dict]) -> dict[str, Any]:
    remote = download_json_if_present(repo_id, remote_state, "main", token)
    if remote:
        local_state.parent.mkdir(parents=True, exist_ok=True)
        write_json(local_state, remote)
        return remote
    if local_state.is_file():
        return json.loads(local_state.read_text(encoding="utf-8"))
    return empty_state(sources)


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
    output_prefix: str,
    filename_prefix: str,
) -> str | None:
    if not pending_path.is_file():
        return None
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    part = Path(pending["local_part"])
    if state.get("next_part", 0) > pending["part_index"]:
        pending_path.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        return None
    if not part.is_file():
        raise RuntimeError(f"Checkpoint references missing local buffer: {part}")
    remote_part = (
        f"{output_prefix}/{filename_prefix}_shard_"
        f"{pending['part_index']:05d}.parquet"
    )
    upload_file(api, repo_id, part, remote_part, token)
    state.update(pending["state_after"])
    save_state(state, local_state, api, repo_id, remote_state, token, part=part)
    pending_path.unlink(missing_ok=True)
    return remote_part


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Override config run_id, useful for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
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
            f"Target tokens: {int(args.target_tokens or cfg.get('target_tokens') or 0):,}"
        )
        console.print(Panel(plan, title="Stage 2 dry-run", border_style="magenta"))
        return

    ensure_dataset_repo(api, output["repo_id"], token, private=bool(output.get("private", True)))
    run_id = str(output.get("run_id", "v1"))
    local_root = Path(output.get("local_dir", "work/smol_stage2")) / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    local_state = local_root / "state.json"
    local_buffer = local_root / "buffer.parquet"
    remote_state = f"_checkpoints/stage2/{run_id}.json"
    state = (
        load_state(local_state, output["repo_id"], remote_state, token, sources)
        if not args.no_resume
        else empty_state(sources)
    )
    output_prefix_value = output.get("path_prefix", "data/" + run_id).rstrip("/")
    filename_prefix = str(output.get("filename_prefix", cfg.get("name", "training")))
    buffer_ready = False
    if not args.no_resume and int(state.get("rows", 0)):
        buffer_ready = download_file_if_present(
            output["repo_id"], f"{output_prefix_value}/buffer.parquet", "main", token, local_buffer
        )
    if state.get("finished"):
        console.print("[dim]Stage 2 is already complete; nothing to process.[/dim]")
        return

    rows = int(state.get("rows", 0))
    estimated_tokens = int(state.get("estimated_tokens", 0))
    counts = dict(state.get("rows_by_source", {source["name"]: 0 for source in sources}))
    target_tokens = args.target_tokens or cfg.get("target_tokens")
    live = Live(
        mix_status_panel(sources, counts, rows, estimated_tokens, target_tokens, "", "initializing"),
        console=console,
        refresh_per_second=2,
        auto_refresh=console.is_terminal or console.is_jupyter,
    )
    live.start()

    iterators = {source["name"]: iter(stream_parquet_prefix(api, source, token)) for source in sources}
    rng = random.Random(int(cfg.get("seed", 42)))
    exhausted: set[str] = set()
    replay_rows = int(state.get("rows", 0))
    replayed = 0
    while replayed < replay_rows:
        try:
            source = choose_source(rng, sources, exhausted)
        except StopIteration:
            raise RuntimeError("Cannot replay the previous Stage-2 checkpoint: all sources exhausted")
        name = source["name"]
        try:
            next(iterators[name])
        except StopIteration:
            exhausted.add(name)
            continue
        replayed += 1
        if replayed % 4096 == 0:
            live.update(
                mix_status_panel(
                    sources, counts, rows, estimated_tokens, target_tokens, name,
                    f"replaying checkpoint {replayed:,}/{replay_rows:,}",
                )
            )

    writer = ShardWriter(
        local_root / "parts",
        target_size_mb=int(output.get("staging_size_mb", 128)),
        max_documents=int(output.get("staging_max_documents", 250_000)),
    )
    started = time.perf_counter()
    last_status = started
    live.update(mix_status_panel(sources, counts, rows, estimated_tokens, target_tokens, "", "mixing"))

    def commit_staging_part(part: Path) -> None:
        nonlocal buffer_ready
        current_buffer = local_buffer if buffer_ready else local_root / ".aggregate" / "no-current-buffer"
        buffer_next, shards, next_shard = build_buffer_transaction(
            current_buffer, [part], local_root / ".aggregate", filename_prefix,
            int(state.get("next_part", 0)), int(output.get("target_size_mb", 1024)) * 1024 * 1024,
            int(output.get("buffer_batch_rows", 4096)),
        )
        for shard_index, shard_path in shards:
            remote_part = f"{output_prefix_value}/{filename_prefix}_shard_{shard_index:05d}.parquet"
            upload_file(api, output["repo_id"], shard_path, remote_part, token)
            console.print(f"[green]Pushed[/green] {remote_part} ({shard_path.stat().st_size / 1024**3:.3f} GiB)")
        upload_file(api, output["repo_id"], buffer_next, f"{output_prefix_value}/buffer.parquet", token)
        state_after = {
            "stage": 2, "name": cfg.get("name", "smol_mix"), "run_id": run_id,
            "rows": rows, "estimated_tokens": estimated_tokens, "rows_by_source": dict(counts),
            "next_part": next_shard, "buffer_bytes": buffer_next.stat().st_size,
            "finished": False, "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        transaction_state = local_root / ".aggregate" / "state.json"
        write_json(transaction_state, state_after)
        upload_file(api, output["repo_id"], transaction_state, remote_state, token)
        os.replace(buffer_next, local_buffer)
        buffer_ready = True
        write_json(local_state, state_after)
        part.unlink(missing_ok=True)
        state.update(state_after)

    while True:
        if args.limit_rows is not None and rows >= int(args.limit_rows):
            break
        if target_tokens is not None and estimated_tokens >= int(target_tokens):
            break
        try:
            source = choose_source(rng, sources, exhausted)
        except StopIteration:
            break
        name = source["name"]
        try:
            row = dict(next(iterators[name]))
        except StopIteration:
            exhausted.add(name)
            continue
        row["source_name"] = name
        row["mix_name"] = cfg.get("name", "smol_mix")
        row["mix_weight"] = float(source["weight"]) / total_weight
        row.setdefault("estimated_tokens", max(1, int(round(len(str(row.get("text", "")).split()) * 1.3))))
        estimated_tokens += int(row["estimated_tokens"] or 1)
        rows += 1
        counts[name] = int(counts.get(name, 0)) + 1
        part = writer.add(row)
        now = time.perf_counter()
        if rows % 2048 == 0 or now - last_status >= 1.0:
            live.update(
                mix_status_panel(
                    sources, counts, rows, estimated_tokens, target_tokens, name, "mixing"
                )
            )
            last_status = now
        if part is None:
            continue
        commit_staging_part(part)

    part = writer.flush()
    if part is not None:
        commit_staging_part(part)

    state.update({
        "stage": 2,
        "name": cfg.get("name", "smol_mix"),
        "run_id": run_id,
        "rows": rows,
        "estimated_tokens": estimated_tokens,
        "rows_by_source": counts,
        "next_part": int(state.get("next_part", 0)),
        "buffer_bytes": local_buffer.stat().st_size if local_buffer.is_file() else 0,
        "finished": args.limit_rows is None and (target_tokens is None or estimated_tokens >= int(target_tokens)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    save_state(state, local_state, api, output["repo_id"], remote_state, token)
    live.update(
        mix_status_panel(sources, counts, rows, estimated_tokens, target_tokens, "", "complete")
    )
    live.stop()
    console.print(
        f"[bold green]Stage 2 complete[/bold green] · rows {rows:,} · "
        f"estimated tokens {estimated_tokens:,}"
    )


if __name__ == "__main__":
    main()
