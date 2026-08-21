"""Shared LaughLM dataset-artifact contract helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping

CONTRACT_NAME = "laughlm_dataset_artifact"
CONTRACT_VERSION = 1
VALID_ARTIFACT_TYPES = {"dataset_stage", "training_run"}
VALID_STAGES = {"stage1", "stage2", "stage3", "stage4", "training"}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_artifact_contract(
    *, artifact_type: str, stage: str, dataset_id: str, run_id: str,
    config_hash: str, source_refs: Iterable[Mapping[str, Any]] = (),
    attributes: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    contract = {
        "name": CONTRACT_NAME, "version": CONTRACT_VERSION,
        "artifact_type": str(artifact_type), "stage": str(stage),
        "dataset_id": str(dataset_id), "run_id": str(run_id),
        "config_hash": str(config_hash),
        "source_refs": [dict(ref) for ref in source_refs],
        "attributes": dict(attributes or {}),
    }
    require_valid_artifact_contract(contract)
    return contract


def validate_artifact_contract(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["artifact_contract must be an object"]
    errors: list[str] = []
    if value.get("name") != CONTRACT_NAME:
        errors.append(f"name must be {CONTRACT_NAME!r}")
    if value.get("version") != CONTRACT_VERSION:
        errors.append(f"version must be {CONTRACT_VERSION}")
    if value.get("artifact_type") not in VALID_ARTIFACT_TYPES:
        errors.append(f"artifact_type must be one of {sorted(VALID_ARTIFACT_TYPES)}")
    if value.get("stage") not in VALID_STAGES:
        errors.append(f"stage must be one of {sorted(VALID_STAGES)}")
    for field in ("dataset_id", "run_id", "config_hash"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if not isinstance(value.get("source_refs"), list):
        errors.append("source_refs must be a list")
    elif any(not isinstance(ref, dict) for ref in value["source_refs"]):
        errors.append("source_refs entries must be objects")
    if not isinstance(value.get("attributes"), dict):
        errors.append("attributes must be an object")
    return errors


def require_valid_artifact_contract(value: Any) -> None:
    errors = validate_artifact_contract(value)
    if errors:
        raise ValueError("Invalid artifact contract: " + "; ".join(errors))


def validate_committed_manifest(
    manifest: Any,
    *,
    expected_stage: str,
    available_files: Iterable[str] | None = None,
) -> list[str]:
    """Validate the resumability contract for a per-source stage manifest."""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    errors.extend(validate_artifact_contract(manifest.get("artifact_contract")))
    contract = manifest.get("artifact_contract")
    if isinstance(contract, dict) and contract.get("stage") != expected_stage:
        errors.append(f"artifact_contract.stage must be {expected_stage!r}")
    if manifest.get("processing_status") != "committed":
        errors.append("processing_status must be 'committed'")
    output_parts = manifest.get("output_parts")
    if not isinstance(output_parts, list) or any(not isinstance(path, str) or not path for path in output_parts):
        errors.append("output_parts must be a list of non-empty paths")
        output_parts = []
    details = manifest.get("output_part_details")
    if not isinstance(details, list) or len(details) != len(output_parts):
        errors.append("output_part_details must match output_parts length")
        details = []
    for index, detail in enumerate(details):
        if not isinstance(detail, dict):
            errors.append(f"output_part_details[{index}] must be an object")
            continue
        for field in ("path", "algorithm", "digest", "bytes"):
            if field not in detail:
                errors.append(f"output_part_details[{index}] missing {field}")
        if index < len(output_parts) and detail.get("path") != output_parts[index]:
            errors.append(f"output_part_details[{index}].path does not match output_parts")
    if available_files is not None:
        available = set(available_files)
        missing = [path for path in output_parts if path not in available]
        if missing:
            errors.append(f"manifest references missing output files: {missing[:3]}")
    return errors


def require_committed_manifest(
    manifest: Any,
    *,
    expected_stage: str,
    available_files: Iterable[str] | None = None,
) -> None:
    errors = validate_committed_manifest(
        manifest, expected_stage=expected_stage, available_files=available_files
    )
    if errors:
        raise ValueError("Invalid committed manifest: " + "; ".join(errors))
