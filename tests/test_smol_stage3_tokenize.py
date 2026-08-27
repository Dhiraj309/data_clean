from __future__ import annotations

from pathlib import Path

from smol_stage3_tokenize import (
    _resource_plan,
    _save_local_state,
    _tokenize_texts,
    select_file_range,
)


class DummyTokenizer:
    is_fast = True

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 4
        return {"input_ids": [[index + 1] * 4 for index, _ in enumerate(texts)]}


def test_resource_plan_uses_file_workers_and_tokenizer_threads() -> None:
    assert _resource_plan(4, None, None) == (1, 4)
    assert _resource_plan(32, None, None) == (4, 8)
    assert _resource_plan(224, None, None) == (16, 14)
    assert _resource_plan(224, 8, 4) == (8, 4)


def test_select_file_range_supports_batched_resume() -> None:
    files = [f"part-{index:02d}.parquet" for index in range(10)]
    assert select_file_range(files, 0, 3) == files[:3]
    assert select_file_range(files, 3, 3) == files[3:6]
    assert select_file_range(files, 8, None) == files[8:]


def test_tokenize_texts_truncates_before_chunking() -> None:
    tokens, valid, skipped = _tokenize_texts(
        ["valid document", None, "another valid document"],
        DummyTokenizer(),
        eos_id=99,
        max_doc_tokens=2,
        hard_doc_limit=4,
    )
    assert valid == 2
    assert skipped == 1
    assert tokens == [1, 1, 99, 1, 1, 99, 2, 2, 99, 2, 2, 99]


def test_save_local_state_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = {"total_tokens": 123}
    _save_local_state(path, state)
    assert path.is_file()
    assert not path.with_suffix(".json.tmp").exists()
    assert '"total_tokens": 123' in path.read_text(encoding="utf-8")
