"""Deterministic document identity and optional near-duplicate fingerprints."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict

import xxhash


@dataclass(frozen=True)
class DocumentIdentity:
    exact_digest: bytes
    normalized_digest: bytes
    near_fingerprint: int


def normalize_text(text: str, policy: Dict[str, Any] | None = None) -> str:
    policy = policy or {}
    value = unicodedata.normalize(str(policy.get("unicode_form", "NFKC")), text)
    if bool(policy.get("casefold", True)):
        value = value.casefold()
    if bool(policy.get("collapse_whitespace", True)):
        value = re.sub(r"\s+", " ", value).strip()
    return value


def _hash_bytes(value: str, algorithm: str) -> bytes:
    if algorithm == "xxh3_128":
        return xxhash.xxh3_128(value.encode("utf-8")).digest()
    if algorithm == "xxh64":
        return xxhash.xxh64(value.encode("utf-8")).digest()
    raise ValueError(f"Unsupported document identity algorithm: {algorithm!r}")


def simhash64(normalized_text: str) -> int:
    """Return a stable 64-bit token-weighted fingerprint."""
    tokens = normalized_text.split()
    if not tokens:
        return 0
    scores = [0] * 64
    for token in tokens:
        value = xxhash.xxh64_intdigest(token.encode("utf-8"))
        for bit in range(64):
            scores[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(scores) if score >= 0)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def document_identity(text: str, algorithm: str, policy: Dict[str, Any] | None = None) -> DocumentIdentity:
    normalized = normalize_text(text, policy)
    return DocumentIdentity(
        exact_digest=_hash_bytes(text, algorithm),
        normalized_digest=_hash_bytes(normalized, algorithm),
        near_fingerprint=simhash64(normalized),
    )
