"""Streaming contamination indexes and comparisons."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import xxhash

from document_identity import hamming_distance, normalize_text, simhash64


def text_ngrams(text: str, n: int) -> Iterator[str]:
    tokens = normalize_text(text).split()
    for index in range(max(0, len(tokens) - n + 1)):
        yield " ".join(tokens[index:index + n])


def build_eval_index(texts: Iterable[str], n_grams: int, near_bucket_bits: int = 8) -> Dict[str, Any]:
    exact: set[str] = set()
    normalized: set[str] = set()
    ngrams: set[str] = set()
    near: Dict[int, List[int]] = defaultdict(list)
    documents = 0
    for text in texts:
        documents += 1
        exact.add(xxhash.xxh3_128_hexdigest(text.encode("utf-8")))
        normalized_text = normalize_text(text)
        normalized.add(xxhash.xxh3_128_hexdigest(normalized_text.encode("utf-8")))
        ngrams.update(text_ngrams(text, n_grams))
        fingerprint = simhash64(normalized_text)
        bucket = fingerprint >> (64 - near_bucket_bits)
        near[bucket].append(fingerprint)
    return {
        "documents": documents,
        "exact": exact,
        "normalized": normalized,
        "ngrams": ngrams,
        "near": near,
    }


def compare_training_texts(
    texts: Iterable[str],
    evaluation_index: Dict[str, Any],
    n_grams: int,
    near_bucket_bits: int = 8,
    near_distance: int = 3,
) -> Dict[str, Any]:
    training_documents = 0
    exact_documents = 0
    normalized_documents = 0
    ngram_documents = 0
    near_documents = 0
    ngram_hits = 0
    for text in texts:
        training_documents += 1
        exact_hash = xxhash.xxh3_128_hexdigest(text.encode("utf-8"))
        normalized_text = normalize_text(text)
        normalized_hash = xxhash.xxh3_128_hexdigest(normalized_text.encode("utf-8"))
        if exact_hash in evaluation_index["exact"]:
            exact_documents += 1
        if normalized_hash in evaluation_index["normalized"]:
            normalized_documents += 1
        hits = sum(1 for ngram in text_ngrams(text, n_grams) if ngram in evaluation_index["ngrams"])
        if hits:
            ngram_documents += 1
            ngram_hits += hits
        fingerprint = simhash64(normalized_text)
        bucket = fingerprint >> (64 - near_bucket_bits)
        if any(
            hamming_distance(fingerprint, candidate) <= near_distance
            for candidate in evaluation_index["near"].get(bucket, [])
        ):
            near_documents += 1
    denominator = max(training_documents, 1)
    return {
        "training_documents": training_documents,
        "evaluation_documents": evaluation_index["documents"],
        "exact_match_documents": exact_documents,
        "normalized_match_documents": normalized_documents,
        "ngram_match_documents": ngram_documents,
        "ngram_match_count": ngram_hits,
        "near_duplicate_documents": near_documents,
        "rates": {
            "exact": exact_documents / denominator,
            "normalized": normalized_documents / denominator,
            "ngram": ngram_documents / denominator,
            "near_duplicate": near_documents / denominator,
        },
        "n_grams": n_grams,
        "near_bucket_bits": near_bucket_bits,
        "near_distance": near_distance,
    }
