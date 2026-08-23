#!/usr/bin/env python3
"""Estimate whether a named Stage-4 output has enough post-Stage-3 text."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from huggingface_hub import HfApi

from config_loader import load_mixture_bundle
from pipeline_utils import download_file, hf_token, setup_logging
from stage4_build import (
    SourcePart,
    iter_parquet_text,
    list_stage3_parts,
    load_tokenizer,
    resolve_eos_id,
    resolve_output_spec,
)


def sample_parts(parts: List[SourcePart], count: int, seed: int) -> List[SourcePart]:
    ordered = sorted(parts, key=lambda item: item.path)
    if len(ordered) <= count:
        return ordered
    rng = random.Random(seed)
    return [ordered[index] for index in sorted(rng.sample(range(len(ordered)), count))]


def count_tokens(part: SourcePart, *, tokenizer, eos_id: int, allowed_splits: List[str], common, token: str, work: Path, batch_size: int) -> tuple[int, int]:
    local = download_file(part.repo_id, part.path, "main", token, common, work)
    documents = 0
    tokens = 0
    batch: List[str] = []

    def flush() -> None:
        nonlocal documents, tokens, batch
        if not batch:
            return
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        documents += len(encodings)
        tokens += sum(len(encoding.ids) + 1 for encoding in encodings)
        batch = []

    try:
        for text in iter_parquet_text(local, allowed_splits=allowed_splits):
            batch.append(text)
            if len(batch) >= batch_size:
                flush()
        flush()
    finally:
        local.unlink(missing_ok=True)
    del eos_id  # EOS is represented by the explicit +1 above.
    return documents, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="train")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--sample-parts-per-source", type=int, default=2)
    parser.add_argument("--minimum-estimated-capacity-ratio", type=float, default=1.10)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    if args.sample_parts_per_source < 1:
        raise ValueError("--sample-parts-per-source must be >= 1")
    if args.minimum_estimated_capacity_ratio <= 0:
        raise ValueError("--minimum-estimated-capacity-ratio must be > 0")

    bundle = load_mixture_bundle(args.config, args.common, args.registry)
    common, mix = bundle["common"], bundle["mixture"]
    setup_logging(common, "stage4_capacity_preflight")
    output = resolve_output_spec(mix, args.output)
    token = hf_token(common)
    api = HfApi(token=token)
    tokenizer = load_tokenizer(mix["tokenizer_name_or_path"])
    eos_id = resolve_eos_id(tokenizer, mix)
    work = Path(common["runtime"]["local_temp_dir"]).expanduser() / "stage4_capacity_preflight" / bundle["mixture_hash"][:12] / output["name"]
    work.mkdir(parents=True, exist_ok=True)

    sources: Dict[str, dict] = {}
    for offset, (name, info) in enumerate(bundle["sources"].items()):
        parts = list_stage3_parts(info["stage3_bundle"], api, token, common)
        selected = sample_parts(parts, args.sample_parts_per_source, int(mix.get("seed", 309)) + offset)
        docs = 0
        tokens = 0
        for part in selected:
            part_docs, part_tokens = count_tokens(
                part,
                tokenizer=tokenizer,
                eos_id=eos_id,
                allowed_splits=output["allowed_splits"] or info["spec"].get("allowed_splits", ["train"]),
                common=common,
                token=token,
                work=work / name,
                batch_size=int(mix.get("tokenizer_batch_size", 256)),
            )
            docs += part_docs
            tokens += part_tokens
        estimated = (tokens / len(selected) * len(parts)) if selected else 0.0
        required = output["source_tokens"][name]
        sources[name] = {
            "stage3_parts": len(parts),
            "sampled_parts": len(selected),
            "sampled_documents": docs,
            "sampled_tokens_including_eos": tokens,
            "estimated_tokens_including_eos": round(estimated),
            "required_tokens": required,
            "estimated_capacity_ratio": estimated / required if required else None,
        }

    passed = all(
        item["estimated_capacity_ratio"] is not None
        and item["estimated_capacity_ratio"] >= args.minimum_estimated_capacity_ratio
        for item in sources.values()
    )
    report = {
        "format": "laughlm_stage4_capacity_preflight_v1",
        "status": "pass" if passed else "fail",
        "mixture": mix["name"],
        "mixture_hash": bundle["mixture_hash"],
        "output": output["name"],
        "allowed_splits": output["allowed_splits"],
        "minimum_estimated_capacity_ratio": args.minimum_estimated_capacity_ratio,
        "sampling_note": "Estimate extrapolates sampled post-Stage-3 parts; it is a launch guard, not an exact capacity proof.",
        "sources": sources,
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[stage4-capacity] report written: {report_path}")
    if not passed:
        raise SystemExit("[stage4-capacity] FAIL")
    print("[stage4-capacity] PASS")


if __name__ == "__main__":
    main()
