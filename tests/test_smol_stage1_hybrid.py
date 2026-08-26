from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import smol_stage1_buffered as stage1


def test_resolve_hybrid_layout_uses_bounded_file_concurrency() -> None:
    assert stage1.resolve_hybrid_layout(4, 10, None, None) == (1, 4)
    assert stage1.resolve_hybrid_layout(24, 10, None, None) == (2, 12)
    assert stage1.resolve_hybrid_layout(24, 10, 4, 12) == (4, 6)


def test_parallel_row_groups_filter_and_resume(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.parquet"
    table = pa.table(
        {
            "text": [f"document number {index} with enough words" for index in range(8)],
            "id": [str(index) for index in range(8)],
            "url": [f"https://example.test/{index}" for index in range(8)],
            "language": ["eng_Latn" if index % 2 == 0 else "fra_Latn" for index in range(8)],
            "token_count": [20] * 8,
            "dataset": ["fixture"] * 8,
        }
    )
    pq.write_table(table, source_path, row_group_size=2)
    cfg = {
        "stage": 1,
        "name": "fixture",
        "source": {"repo_id": "example/fixture", "split": "train"},
        "columns": {
            "text": "text",
            "id": "id",
            "url": "url",
            "language": "language",
            "token_count": "token_count",
            "dataset": "dataset",
        },
        "required_columns": ["text", "id", "language", "token_count"],
        "filters": {"languages": ["eng_Latn"], "min_token_count": 10},
        "output": {
            "repo_id": "example/output",
            "run_id": "test",
            "local_dir": str(tmp_path / "work"),
            "staging_size_mb": 1,
            "staging_max_documents": 100,
        },
    }
    job = {
        "cfg": cfg,
        "source_file": "data/source.parquet",
        "source_index": 0,
        "token": "unused",
        "resume": True,
        "limit_rows": None,
        "workers_per_file": 2,
        "source_batch_rows": 2,
        "keep_source_files": True,
    }
    monkeypatch.setattr(stage1, "_download_source_file", lambda unused_job: source_path)

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
        job["process_executor"] = executor
        result = stage1.filter_source_file(job)

        assert result["mode"] == "row_groups"
        assert result["rows_seen"] == 8
        assert result["accepted"] == 4
        assert result["rejected"] == 4
        assert result["completed_row_groups"] == [0, 1, 2, 3]
        assert all(Path(value).is_file() for value in result["staging_parts"])
        assert sum(
            pq.ParquetFile(value).metadata.num_rows for value in result["staging_parts"]
        ) == 4

        monkeypatch.setattr(
            stage1,
            "_download_source_file",
            lambda unused_job: (_ for _ in ()).throw(AssertionError("resume downloaded source")),
        )
        resumed = stage1.filter_source_file(job)
        assert resumed["result_status"] == "staged"
        assert resumed["accepted"] == 4


def test_two_active_files_share_one_process_budget(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "shared-source.parquet"
    pq.write_table(
        pa.table(
            {
                "text": [f"valid document {index}" for index in range(12)],
                "id": [str(index) for index in range(12)],
                "language": ["eng_Latn"] * 12,
                "token_count": [50] * 12,
            }
        ),
        source_path,
        row_group_size=2,
    )
    cfg = {
        "stage": 1,
        "name": "shared-fixture",
        "source": {"repo_id": "example/shared", "split": "train"},
        "columns": {
            "text": "text", "id": "id", "language": "language",
            "token_count": "token_count",
        },
        "required_columns": ["text", "id", "language", "token_count"],
        "filters": {"languages": ["eng_Latn"], "min_token_count": 10},
        "output": {
            "repo_id": "example/output", "run_id": "test",
            "local_dir": str(tmp_path / "work"), "staging_size_mb": 1,
        },
    }
    monkeypatch.setattr(stage1, "_download_source_file", lambda unused_job: source_path)
    jobs = [
        {
            "cfg": cfg, "source_file": f"data/source-{index}.parquet",
            "source_index": index, "token": "unused", "resume": True,
            "limit_rows": None, "workers_per_file": 2, "source_batch_rows": 2,
            "keep_source_files": True,
        }
        for index in range(2)
    ]
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as processes:
        for job in jobs:
            job["process_executor"] = processes
        with ThreadPoolExecutor(max_workers=2) as files:
            results = list(files.map(stage1.filter_source_file, jobs))

    assert [result["rows_seen"] for result in results] == [12, 12]
    assert [result["accepted"] for result in results] == [12, 12]
    assert all(result["completed_row_groups"] == list(range(6)) for result in results)
