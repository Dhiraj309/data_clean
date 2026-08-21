#!/usr/bin/env python3
"""Build a portable, checksum-verified LaughLM data-pipeline handoff.

This utility is deliberately local and does not contact Hugging Face or run a
processing stage.  It packages the checked-in contracts plus user-supplied
manifests/reports so another operator can reproduce or resume the pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from config_loader import load_yaml, stable_hash


REPO_ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def repo_relative(path: Path, repo_root: Path) -> Path | None:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return None


def copy_artifact(
    source: Path,
    *,
    category: str,
    output_dir: Path,
    repo_root: Path,
) -> str:
    """Copy a file into a stable bundle location and return its bundle path."""
    relative = repo_relative(source, repo_root)
    if relative is None:
        relative = Path("external") / f"{sha256_file(source)[:12]}_{source.name}"
    destination_relative = Path(category) / relative
    destination = output_dir / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination_relative.as_posix()


def discover_configs(repo_root: Path) -> List[Path]:
    config_root = repo_root / "configs"
    return sorted(path for path in config_root.rglob("*.yaml") if path.is_file())


def command_inventory(configs: Iterable[Path], repo_root: Path) -> List[str]:
    commands: List[str] = []
    stage_scripts = {
        "stage1": "stage1_filter.py",
        "stage2": "stage2_process.py",
        "stage3": "stage3_decontam.py",
        "stage4": "stage4_build.py",
    }
    for config in configs:
        try:
            data = load_yaml(config)
        except (FileNotFoundError, ValueError):
            continue
        stage = str(data.get("stage", ""))
        script = stage_scripts.get(f"stage{stage}")
        if script is None:
            continue
        commands.append(f"python {script} --config {config.relative_to(repo_root).as_posix()}")
    return commands


def resource_profile(common_path: Path) -> Dict[str, Any]:
    common = load_yaml(common_path)
    return {
        "pipeline_version": (common.get("pipeline") or {}).get("version"),
        "storage": common.get("storage", {}),
        "runtime": common.get("runtime", {}),
        "retry": common.get("retry", {}),
        "hashing": common.get("hashing", {}),
    }


def build_dataset_release(
    final_manifest: Dict[str, Any] | None,
    *,
    final_manifest_bundle_path: str | None,
    stage4_configs: List[str],
    release_id: str,
) -> Dict[str, Any]:
    """Create the stable descriptor consumed by LaughLM data loading."""
    if final_manifest is None:
        return {
            "format": "laughlm_dataset_release_v1",
            "status": "pending_final_manifest",
            "release_id": release_id,
            "stage4_configs": stage4_configs,
            "message": "Supply a committed Stage-4 corpus_manifest.json to mark this release ready.",
        }

    contract = final_manifest.get("artifact_contract") or {}
    attributes = contract.get("attributes") or {}
    committed = final_manifest.get("processing_status") == "committed"
    return {
        "format": "laughlm_dataset_release_v1",
        "status": "ready" if committed else "invalid",
        "release_id": release_id,
        "dataset": {
            "name": final_manifest.get("name"),
            "repo_id": attributes.get("output_repo"),
            "revision": "main",
            "manifest": final_manifest_bundle_path,
            "manifest_contract": contract,
        },
        "loader_contract": {
            "repo_id": attributes.get("output_repo"),
            "revision": "main",
            "shards": final_manifest.get("shards", []),
            "target_tokens": final_manifest.get("target_tokens"),
            "written_tokens": final_manifest.get("written_tokens"),
            "dtype": final_manifest.get("dtype"),
            "tokenizer_contract": final_manifest.get("tokenizer_contract"),
            "packing_contract": final_manifest.get("packing_contract"),
        },
        "lineage": {
            "mixture_hash": final_manifest.get("mixture_hash"),
            "contract_hash": final_manifest.get("contract_hash"),
            "source_stage3_hashes": final_manifest.get("source_stage3_hashes", {}),
            "split_policy_version": final_manifest.get("split_policy_version"),
        },
        "stage4_configs": stage4_configs,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LaughLM data release handoff bundle")
    parser.add_argument("--output", required=True, help="New or existing local bundle directory")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--manifest", action="append", default=[], help="Stage manifest to include")
    parser.add_argument("--report", action="append", default=[], help="Audit report to include")
    parser.add_argument("--final-manifest", default=None, help="Committed Stage-4 corpus_manifest.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_files = discover_configs(repo_root)
    common_path = repo_root / "configs" / "common.yaml"
    if not common_path.is_file():
        raise FileNotFoundError(common_path)

    copied: List[Tuple[str, str]] = []
    handoff_files = [
        repo_root / "README.md",
        repo_root / "ROADMAP.md",
        repo_root / "requirements.txt",
    ]
    handoff_files.extend(sorted((repo_root / "schemas").glob("*.json")))
    handoff_files.extend(sorted(repo_root.glob("*.py")))
    for source in handoff_files:
        if source.is_file():
            category = "source" if source.suffix == ".py" else "docs"
            bundle_path = copy_artifact(
                source, category=category, output_dir=output_dir, repo_root=repo_root
            )
            copied.append((bundle_path, "handoff"))
    for source in config_files:
        bundle_path = copy_artifact(source, category="configs", output_dir=output_dir, repo_root=repo_root)
        copied.append((bundle_path, "config"))
    for source_value in args.manifest:
        source = resolve_file(source_value, repo_root)
        bundle_path = copy_artifact(source, category="manifests", output_dir=output_dir, repo_root=repo_root)
        copied.append((bundle_path, "manifest"))
    for source_value in args.report:
        source = resolve_file(source_value, repo_root)
        bundle_path = copy_artifact(source, category="reports", output_dir=output_dir, repo_root=repo_root)
        copied.append((bundle_path, "report"))

    final_manifest: Dict[str, Any] | None = None
    final_manifest_bundle_path: str | None = None
    if args.final_manifest:
        source = resolve_file(args.final_manifest, repo_root)
        final_manifest_bundle_path = copy_artifact(
            source, category="manifests", output_dir=output_dir, repo_root=repo_root
        )
        copied.append((final_manifest_bundle_path, "final_manifest"))
        final_manifest = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(final_manifest, dict):
            raise ValueError("Final Stage-4 manifest must contain a JSON object")

    stage4_configs = [
        path.relative_to(repo_root).as_posix()
        for path in config_files
        if path.parent.name == "stage4"
    ]
    release_id = stable_hash({"files": sorted(copied), "stage4_configs": stage4_configs})[:20]
    dataset_release = build_dataset_release(
        final_manifest,
        final_manifest_bundle_path=final_manifest_bundle_path,
        stage4_configs=stage4_configs,
        release_id=release_id,
    )
    dataset_release_path = output_dir / "laughlm_dataset_release_v1.json"
    dataset_release_path.write_text(json.dumps(dataset_release, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    commands = command_inventory(config_files, repo_root)
    write_text(
        output_dir / "commands.md",
        "# Reproducible pipeline commands\n\n"
        "Run from the data_clean repository root with the required Python dependencies and HF_TOKEN.\n\n"
        + "\n".join(f"```bash\n{command}\n```" for command in commands),
    )
    profile = resource_profile(common_path)
    write_text(
        output_dir / "OPERATIONS.md",
        "# Operational handoff\n\n"
        "1. Run Stage 1, Stage 2, and Stage 3 per source, then audit committed manifests and splits.\n"
        "2. Run Stage 4 only after the tokenizer, benchmark, sealed-evaluation, and mixture contracts are frozen.\n"
        "3. Stages resume from committed manifests; Stage 4 resumes from its remote progress marker.\n"
        "4. Retry failed units after reviewing their failure manifest. Do not delete committed remote artifacts.\n"
        "5. Use `--fresh` only when intentionally replacing a Stage-4 run with the same mixture identity.\n"
        "6. The release descriptor is ready for LaughLM only when `status` is `ready` and all referenced hashes/checksums are verified.\n\n"
        "## Resource profile\n\n"
        "```json\n" + json.dumps(profile, indent=2, sort_keys=True) + "\n```\n",
    )

    release_files: List[Dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "release_manifest.json":
            continue
        release_files.append({
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    release_manifest = {
        "format": "laughlm_data_release_bundle_v1",
        "release_id": release_id,
        "created_at_utc": utc_now(),
        "status": dataset_release["status"],
        "bundle_files": release_files,
        "bundle_hash": stable_hash(release_files),
        "resource_profile": profile,
        "reproducible_commands": commands,
        "dataset_release": "laughlm_dataset_release_v1.json",
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[release-bundle] wrote {len(release_files)} files to {output_dir}")
    print(f"[release-bundle] manifest: {output_dir / 'release_manifest.json'}")


if __name__ == "__main__":
    main()
