"""Versioned benchmark and sealed-evaluation contracts."""
from __future__ import annotations

from typing import Any, Dict

from config_loader import stable_hash


def benchmark_task_errors(tasks: Any) -> list[str]:
    if not isinstance(tasks, list) or not tasks:
        return ["frozen benchmark must contain at least one task"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        if isinstance(task, str):
            identifier = task.strip()
            if not identifier:
                errors.append(f"task[{index}] must be non-empty")
            if identifier.startswith("REPLACE_"):
                errors.append(f"task[{index}] still contains a placeholder")
        elif isinstance(task, dict):
            identifier = str(task.get("id") or task.get("name") or "").strip()
            if not identifier:
                errors.append(f"task[{index}] must contain a non-empty id or name")
        else:
            errors.append(f"task[{index}] must be a string or mapping")
            continue
        fingerprint = stable_hash(task)
        if fingerprint in seen:
            errors.append(f"task[{index}] duplicates an earlier task")
        seen.add(fingerprint)
    return errors


def sealed_evaluation_errors(
    config: Dict[str, Any] | None,
    training_repo_id: str | None = None,
) -> list[str]:
    if not config:
        return ["a separate sealed evaluation config is required"]
    errors: list[str] = []
    if config.get("freeze_status") != "frozen":
        errors.append("sealed evaluation must set freeze_status='frozen'")
    if not str(config.get("name", "")).strip():
        errors.append("sealed evaluation name is required")
    if int(config.get("version", 0)) <= 0:
        errors.append("sealed evaluation version must be > 0")
    repo_id = str(config.get("repo_id", ""))
    if not repo_id or repo_id.startswith("REPLACE_"):
        errors.append("sealed evaluation must specify a real repo_id")
    configured_training_repo = str(config.get("training_repo_id", ""))
    if training_repo_id and configured_training_repo != training_repo_id:
        errors.append("sealed evaluation training_repo_id must match the training repository")
    if repo_id and training_repo_id and repo_id == training_repo_id:
        errors.append("sealed evaluation repo must differ from training repo")
    if not str(config.get("text_column", "")).strip():
        errors.append("sealed evaluation text_column is required")
    paths = config.get("paths")
    if not isinstance(paths, list) or not paths:
        errors.append("sealed evaluation must freeze an explicit non-empty paths list")
    return errors


def benchmark_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    tasks = list(config.get("lighteval_tasks") or [])
    sealed = dict(config.get("sealed_evaluation") or {})
    return {
        "name": str(config.get("name", "")),
        "version": int(config.get("version", 0)),
        "freeze_status": str(config.get("freeze_status", "draft")),
        "config_hash": stable_hash(config),
        "task_hash": stable_hash(tasks),
        "tasks": tasks,
        "language": str(config.get("language", "en")),
        "n_grams": int(config.get("n_grams", 13)),
        "sealed_evaluation": sealed,
    }


def benchmark_errors(config: Dict[str, Any]) -> list[str]:
    manifest = benchmark_manifest(config)
    errors: list[str] = []
    if not manifest["name"]:
        errors.append("benchmark name is required")
    if manifest["version"] <= 0:
        errors.append("benchmark version must be > 0")
    if manifest["n_grams"] <= 0:
        errors.append("benchmark n_grams must be > 0")
    if manifest["freeze_status"] == "frozen":
        errors.extend(benchmark_task_errors(manifest["tasks"]))
    if manifest["freeze_status"] not in {"draft", "frozen"}:
        errors.append("freeze_status must be draft or frozen")
    sealed = manifest["sealed_evaluation"]
    if sealed and sealed.get("repo_id") == sealed.get("training_repo_id"):
        errors.append("sealed evaluation repo must differ from training repo")
    return errors


def require_frozen_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    errors = benchmark_errors(config)
    if errors:
        raise ValueError("Invalid benchmark contract: " + "; ".join(errors))
    manifest = benchmark_manifest(config)
    if manifest["freeze_status"] != "frozen":
        raise ValueError("Benchmark must set freeze_status='frozen' before Stage 3 production")
    return manifest


def require_frozen_sealed_evaluation(config: Dict[str, Any] | None, training_repo_id: str) -> Dict[str, Any]:
    errors = sealed_evaluation_errors(config, training_repo_id)
    if errors:
        raise ValueError("Invalid sealed evaluation contract: " + "; ".join(errors))
    assert config is not None
    repo_id = str(config.get("repo_id", ""))
    return {
        "name": str(config.get("name", "")),
        "version": int(config["version"]),
        "repo_id": repo_id,
        "revision": str(config.get("revision", "main")),
        "config_hash": stable_hash(config),
    }
