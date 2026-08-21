"""Deterministic train/evaluation split assignment."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict

import xxhash


def _metadata_value(metadata: Dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = metadata.get(name)
        if value is not None:
            return str(value).strip().casefold()
    return ""


def _bucket(group_key: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("split modulo must be > 0")
    return xxhash.xxh64_intdigest(group_key.encode("utf-8")) % modulo


def assign_split(
    *,
    dataset_name: str,
    group_key: str,
    metadata: Dict[str, Any],
    policy: Dict[str, Any],
) -> str:
    """Apply explicit precedence so split assignment is reproducible."""
    sealed_values = {str(v).casefold() for v in policy.get("sealed_metadata_values", [])}
    synthetic_values = {str(v).casefold() for v in policy.get("synthetic_metadata_values", [])}
    eval_split = _metadata_value(metadata, ("split", "eval_split", "dataset_split"))
    if eval_split in sealed_values or _metadata_value(metadata, ("is_sealed",)) in {"1", "true", "yes"}:
        return "sealed"
    if dataset_name in set(policy.get("held_out_source_datasets", [])):
        return "held_out_source"
    source_kind = _metadata_value(metadata, ("source_type", "data_type", "synthetic"))
    if source_kind in synthetic_values or _metadata_value(metadata, ("is_synthetic",)) in {"1", "true", "yes"}:
        return "synthetic"
    cutoff = policy.get("temporal_cutoff")
    raw_date = _metadata_value(metadata, ("date", "timestamp", "created_at"))
    if cutoff and raw_date:
        try:
            if date.fromisoformat(raw_date[:10]) >= date.fromisoformat(str(cutoff)[:10]):
                return "temporal"
        except ValueError:
            pass

    test_remainder = policy.get("test_remainder")
    if test_remainder is not None and _bucket(group_key, int(policy.get("test_modulo", 100))) == int(test_remainder):
        return "test"
    if _bucket(group_key, int(policy.get("validation_modulo", 100))) == int(policy.get("validation_remainder", 0)):
        return "validation"
    return "train"
