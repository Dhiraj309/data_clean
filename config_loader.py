"""Configuration loading for the split LaughLM dataset pipeline.

Stage configs deliberately contain only stage-specific behavior.  Shared runtime
settings live in configs/common.yaml and dataset identity/repositories live in
configs/datasets.yaml.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_COMMON = ROOT / "configs" / "common.yaml"
DEFAULT_REGISTRY = ROOT / "configs" / "datasets.yaml"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {p}")
    return data


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_common(path: str | Path | None = None) -> Dict[str, Any]:
    return load_yaml(path or DEFAULT_COMMON)


def load_registry(path: str | Path | None = None) -> Dict[str, Any]:
    data = load_yaml(path or DEFAULT_REGISTRY)
    datasets = data.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("configs/datasets.yaml must contain a non-empty 'datasets' mapping")
    return datasets


def get_dataset(registry: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in registry:
        raise KeyError(f"Unknown dataset {name!r}. Registered: {sorted(registry)}")
    entry = dict(registry[name])
    entry["name"] = name
    return entry


def resolve_relative(path_value: str, owner_file: str | Path) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = Path(owner_file).expanduser().resolve().parent / p
    return p.resolve()


def load_stage_bundle(
    stage_config_path: str | Path,
    expected_stage: int,
    common_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> Dict[str, Any]:
    stage_path = Path(stage_config_path).expanduser().resolve()
    stage_cfg = load_yaml(stage_path)
    stage_number = int(stage_cfg.get("stage", -1))
    if stage_number != expected_stage:
        raise ValueError(f"Expected stage={expected_stage} in {stage_path}, found {stage_cfg.get('stage')!r}")
    dataset_name = stage_cfg.get("dataset")
    if not dataset_name:
        raise ValueError(f"Missing 'dataset' in {stage_path}")
    common = load_common(common_path)
    registry = load_registry(registry_path)
    dataset = get_dataset(registry, dataset_name)
    bundle = {
        "common": common,
        "registry": registry,
        "dataset": dataset,
        "stage": stage_cfg,
        "stage_path": stage_path,
        "stage_hash": stable_hash(stage_cfg),
        "common_hash": stable_hash(common),
    }
    if expected_stage == 2 and stage_cfg.get("stage1_config"):
        s1_path = resolve_relative(stage_cfg["stage1_config"], stage_path)
        bundle["stage1_bundle"] = load_stage_bundle(s1_path, 1, common_path, registry_path)
        if bundle["stage1_bundle"]["dataset"]["name"] != dataset_name:
            raise ValueError("Stage-2 stage1_config points at a different dataset")
    return bundle


def load_stage3_bundle(
    stage_config_path: str | Path,
    common_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> Dict[str, Any]:
    bundle = load_stage_bundle(stage_config_path, 3, common_path, registry_path)
    stage_cfg = bundle["stage"]
    benchmark_path = resolve_relative(stage_cfg["benchmark_config"], bundle["stage_path"])
    stage2_path = resolve_relative(stage_cfg["stage2_config"], bundle["stage_path"])
    benchmark_cfg = load_yaml(benchmark_path)
    stage2_bundle = load_stage_bundle(stage2_path, 2, common_path, registry_path)
    if stage2_bundle["dataset"]["name"] != bundle["dataset"]["name"]:
        raise ValueError("Stage-3 stage2_config points at a different dataset")
    bundle.update(
        benchmark=benchmark_cfg,
        benchmark_path=benchmark_path,
        benchmark_hash=stable_hash(benchmark_cfg),
        stage2_bundle=stage2_bundle,
    )
    return bundle


def load_mixture_bundle(
    mixture_config_path: str | Path,
    common_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> Dict[str, Any]:
    mix_path = Path(mixture_config_path).expanduser().resolve()
    mix = load_yaml(mix_path)
    if int(mix.get("stage", -1)) != 4:
        raise ValueError(f"Expected stage=4 in {mix_path}")
    common = load_common(common_path)
    registry = load_registry(registry_path)
    resolved_sources: Dict[str, Any] = {}
    for name, spec in (mix.get("sources") or {}).items():
        stage3_path = resolve_relative(spec["stage3_config"], mix_path)
        s3 = load_stage3_bundle(stage3_path, common_path, registry_path)
        if s3["dataset"]["name"] != name:
            raise ValueError(f"Mixture source {name} points to Stage-3 config for {s3['dataset']['name']}")
        resolved_sources[name] = {"spec": spec, "stage3_bundle": s3}
    return {
        "common": common,
        "registry": registry,
        "mixture": mix,
        "mixture_path": mix_path,
        "mixture_hash": stable_hash(mix),
        "sources": resolved_sources,
    }


def stage1_semantic_hash(bundle: Dict[str, Any]) -> str:
    d = bundle["dataset"]
    return stable_hash(
        {
            "pipeline_version": bundle["common"]["pipeline"]["version"],
            "dataset": d["name"],
            "source": d["source"],
            "stage1": bundle["stage"],
        }
    )


def stage2_semantic_hash(bundle: Dict[str, Any]) -> str:
    d = bundle["dataset"]
    stage = bundle["stage"]
    return stable_hash(
        {
            "pipeline_version": bundle["common"]["pipeline"]["version"],
            "dataset": d["name"],
            "stage1_semantic_hash": stage1_semantic_hash(
                bundle.get("stage1_bundle")
                or load_stage_bundle(resolve_relative(stage["stage1_config"], bundle["stage_path"]), 1)
            ),
            "dedup_namespace": stage["dedup_namespace"],
            "stage2": stage,
        }
    )


def stage3_semantic_hash(bundle: Dict[str, Any]) -> str:
    return stable_hash(
        {
            "pipeline_version": bundle["common"]["pipeline"]["version"],
            "dataset": bundle["dataset"]["name"],
            "stage2_semantic_hash": stage2_semantic_hash(bundle["stage2_bundle"]),
            "benchmark_hash": bundle["benchmark_hash"],
            "stage3": bundle["stage"],
        }
    )
