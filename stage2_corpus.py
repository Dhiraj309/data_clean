#!/usr/bin/env python3
"""Canonical, priority-safe Stage-2 corpus driver.

Stage 2 exact/normalized dedup keeps the first encountered document.  This
driver therefore owns ordering across all participating datasets: higher
``source_priority`` always runs first, and ties use the dataset name.  It is
the normal entry point for a shared dedup namespace.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table

import filters as custom_filters
from config_loader import load_stage_bundle, load_yaml, resolve_relative
from pipeline_utils import ensure_repo, hf_token, local_work_root, setup_logging, slug
from stage2_process import CommittedHashStore, list_stage1_manifests, process_source


def load_plan(path: str | Path) -> Dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    plan = load_yaml(plan_path)
    if int(plan.get("stage", -1)) != 2:
        raise ValueError(f"Expected stage=2 in {plan_path}")
    if not isinstance(plan.get("sources"), list) or not plan["sources"]:
        raise ValueError("Stage-2 corpus plan requires a non-empty sources list")
    if not plan.get("dedup_namespace"):
        raise ValueError("Stage-2 corpus plan requires dedup_namespace")
    plan["_path"] = plan_path
    return plan


def load_bundles(plan: Dict[str, Any], common: str | None, registry: str | None) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    names: set[str] = set()
    for spec in plan["sources"]:
        if not isinstance(spec, dict) or not spec.get("config"):
            raise ValueError("Each Stage-2 corpus source requires config")
        config_path = resolve_relative(str(spec["config"]), plan["_path"])
        bundle = load_stage_bundle(config_path, 2, common, registry)
        name = bundle["dataset"]["name"]
        if name in names:
            raise ValueError(f"Duplicate dataset in Stage-2 corpus plan: {name}")
        names.add(name)
        namespace = bundle["stage"].get("dedup_namespace")
        if namespace != plan["dedup_namespace"]:
            raise ValueError(
                f"{config_path} uses dedup_namespace={namespace!r}; "
                f"plan requires {plan['dedup_namespace']!r}"
            )
        bundles.append(bundle)
    return sorted(bundles, key=lambda item: (-int(item["dataset"].get("source_priority", 0)), item["dataset"]["name"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/stage2_corpus.yaml")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    bundles = load_bundles(plan, args.common, args.registry)
    common = bundles[0]["common"]
    if any(bundle["common"] != common for bundle in bundles[1:]):
        raise ValueError("All Stage-2 corpus configs must resolve to the same common configuration")
    setup_logging(common, "stage2_corpus")
    if common.get("plugins", {}).get("filter_paths"):
        custom_filters.load_plugins(common["plugins"]["filter_paths"])

    table = Table(title=f"Stage 2 canonical plan — {plan['dedup_namespace']}")
    table.add_column("Priority", justify="right")
    table.add_column("Dataset")
    table.add_column("Stage-1 repo")
    for bundle in bundles:
        dataset = bundle["dataset"]
        table.add_row(str(dataset.get("source_priority", 0)), dataset["name"], dataset["repos"]["stage1"])
    Console().print(table)
    if args.dry_run:
        return

    token = hf_token(common)
    api = HfApi(token=token)
    for bundle in bundles:
        ensure_repo(api, bundle["dataset"]["repos"]["stage1"], token, common, must_exist=True)
        ensure_repo(api, bundle["dataset"]["repos"]["stage2"], token, common)

    db = local_work_root(common) / "dedup" / f"{slug(plan['dedup_namespace'])}.sqlite3"
    store = CommittedHashStore(db, plan["dedup_namespace"], common["hashing"]["algorithm"])
    try:
        for bundle in bundles:
            dataset = bundle["dataset"]
            manifests = list_stage1_manifests(bundle, bundle["stage1_bundle"], api, token)
            Console().print(
                f"[stage2-corpus] {dataset['name']} priority={dataset.get('source_priority', 0)} "
                f"sources={len(manifests)}"
            )
            for manifest in manifests:
                Console().print(process_source(bundle, manifest, api, token, store))
    finally:
        store.close()


if __name__ == "__main__":
    main()
