from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import smol_stage2_mix as stage2
from smol_stage2_mix import OutputShardWriter, choose_source_by_token_debt, make_resource_plan


def test_resource_plan_scales_without_oversubscribing() -> None:
    small = make_resource_plan(cpu_count=4)
    large = make_resource_plan(cpu_count=224)
    assert small.cpu_workers == 2
    assert small.download_workers == 1
    assert large.cpu_workers == 64
    assert large.download_workers == 24
    assert large.upload_workers == 1


def test_token_debt_prefers_source_below_target() -> None:
    sources = [
        {"name": "general", "weight": 0.75},
        {"name": "math", "weight": 0.25},
    ]
    selected = choose_source_by_token_debt(
        sources,
        {"general": 750, "math": 100},
        set(),
        seed=42,
        target_tokens=1_000,
    )
    assert selected["name"] == "math"


def test_output_writer_rotates_by_compressed_file_size(tmp_path: Path) -> None:
    writer = OutputShardWriter(
        tmp_path,
        next_part=0,
        target_bytes=1,
        batch_rows=2,
        max_documents=100,
    )
    row = {
        "text": "a small test document",
        "id": "1",
        "url": None,
        "language": "en",
        "dataset": "fixture",
        "source_name": "general",
        "source_repo": "fixture/repo",
        "source_config": None,
        "source_split": "train",
        "language_score": None,
        "fasttext_score": None,
        "score": None,
        "answer_count": None,
        "accepted_answer_id": None,
        "int_score": None,
        "token_count": 5,
        "word_count": 5,
        "char_count": 21,
        "estimated_tokens": 5,
        "mix_name": "fixture",
        "mix_weight": 1.0,
    }
    assert writer.add(row) is None
    part = writer.add(dict(row, id="2"))
    assert part is not None
    assert part.name == "part-00000.parquet"
    assert pq.ParquetFile(part).metadata.num_rows == 2
    assert pa.parquet.ParquetFile(part).metadata.num_rows == 2


def test_unfinished_state_accepts_append_only_source_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"stage": 2, "name": "fixture", "seed": 42}
    sources = [{"name": "general", "weight": 1.0}]
    saved_revisions = {"general": "old-revision"}
    saved_files = {"general": ["general/shard_00000.parquet"]}
    saved_hash = stage2.stable_id({
        "mixer_version": stage2.MIXER_VERSION,
        "config": config,
        "source_revisions": saved_revisions,
        "source_files": saved_files,
    })
    candidate = {
        "mixer_version": stage2.MIXER_VERSION,
        "config_hash": saved_hash,
        "source_revisions": saved_revisions,
        "source_files": saved_files,
        "source_cursors": {
            "general": {
                "file_index": 1,
                "row_offset": 0,
                "exhausted": True,
            }
        },
    }
    monkeypatch.setattr(stage2, "download_json_if_present", lambda *args: candidate)
    current_revisions = {"general": "new-revision"}
    current_files = {
        "general": [
            "general/shard_00000.parquet",
            "general/shard_00001.parquet",
        ]
    }
    current_hash = stage2.stable_id({
        "mixer_version": stage2.MIXER_VERSION,
        "config": config,
        "source_revisions": current_revisions,
        "source_files": current_files,
    })

    state = stage2.load_state(
        tmp_path / "state.json",
        "fixture/repo",
        "checkpoint.json",
        "token",
        sources,
        config,
        current_hash,
        current_revisions,
        current_files,
    )

    assert state["source_files"] == current_files
    assert state["source_revisions"] == current_revisions
    assert state["config_hash"] == current_hash
    assert state["source_cursors"]["general"]["exhausted"] is False


def test_state_rejects_reordered_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"stage": 2, "name": "fixture", "seed": 42}
    sources = [{"name": "general", "weight": 1.0}]
    saved_revisions = {"general": "old-revision"}
    saved_files = {"general": ["general/shard_00000.parquet"]}
    saved_hash = stage2.stable_id({
        "mixer_version": stage2.MIXER_VERSION,
        "config": config,
        "source_revisions": saved_revisions,
        "source_files": saved_files,
    })
    candidate = {
        "mixer_version": stage2.MIXER_VERSION,
        "config_hash": saved_hash,
        "source_revisions": saved_revisions,
        "source_files": saved_files,
    }
    monkeypatch.setattr(stage2, "download_json_if_present", lambda *args: candidate)
    current_revisions = {"general": "new-revision"}
    current_files = {"general": ["general/shard_00001.parquet"]}
    current_hash = stage2.stable_id({
        "mixer_version": stage2.MIXER_VERSION,
        "config": config,
        "source_revisions": current_revisions,
        "source_files": current_files,
    })

    with pytest.raises(RuntimeError, match="removed, reordered, or changed"):
        stage2.load_state(
            tmp_path / "state.json",
            "fixture/repo",
            "checkpoint.json",
            "token",
            sources,
            config,
            current_hash,
            current_revisions,
            current_files,
        )
