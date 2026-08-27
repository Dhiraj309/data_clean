from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from smol_stage2_mix import OutputShardWriter, choose_source_by_token_debt


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
