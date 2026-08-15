#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from config_loader import load_mixture_bundle, load_stage_bundle, load_stage3_bundle, stage1_semantic_hash, stage2_semantic_hash, stage3_semantic_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all LaughLM pipeline YAML files without HF access")
    parser.add_argument("--configs-root", default="configs")
    args = parser.parse_args()
    root = Path(args.configs_root)
    console = Console()
    for path in sorted((root / "stage1").glob("*.yaml")):
        b = load_stage_bundle(path, 1)
        console.print(f"[green]OK[/green] stage1 {b['dataset']['name']} {stage1_semantic_hash(b)[:12]}")
    for path in sorted((root / "stage2").glob("*.yaml")):
        b = load_stage_bundle(path, 2)
        console.print(f"[green]OK[/green] stage2 {b['dataset']['name']} {stage2_semantic_hash(b)[:12]}")
    for path in sorted((root / "stage3").glob("*.yaml")):
        if path.name == "benchmarks.yaml":
            continue
        b = load_stage3_bundle(path)
        console.print(f"[green]OK[/green] stage3 {b['dataset']['name']} {stage3_semantic_hash(b)[:12]}")
    for path in sorted((root / "stage4").glob("*.yaml")):
        b = load_mixture_bundle(path)
        total = sum(int(v['spec']['tokens']) for v in b['sources'].values())
        if total != int(b['mixture']['target_tokens']):
            raise ValueError(f"{path}: source tokens {total} != target {b['mixture']['target_tokens']}")
        console.print(f"[green]OK[/green] stage4 {b['mixture']['name']} {b['mixture_hash'][:12]}")


if __name__ == "__main__":
    main()
