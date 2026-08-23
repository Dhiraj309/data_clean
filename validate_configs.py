#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from benchmark_contract import require_frozen_benchmark, require_frozen_sealed_evaluation
from config_loader import load_mixture_bundle, load_stage_bundle, load_stage3_bundle, load_yaml, stage1_semantic_hash, stage2_semantic_hash, stage3_semantic_hash
from stage4_build import resolve_output_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all LaughLM pipeline YAML files without HF access")
    parser.add_argument("--configs-root", default="configs")
    parser.add_argument("--common", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--production-stage4", default=None, help="Validate one frozen, production-ready Stage-4 mixture")
    args = parser.parse_args()
    root = Path(args.configs_root)
    console = Console()
    for path in sorted((root / "stage1").glob("*.yaml")):
        b = load_stage_bundle(path, 1, args.common, args.registry)
        console.print(f"[green]OK[/green] stage1 {b['dataset']['name']} {stage1_semantic_hash(b)[:12]}")
    for path in sorted((root / "stage2").glob("*.yaml")):
        b = load_stage_bundle(path, 2, args.common, args.registry)
        console.print(f"[green]OK[/green] stage2 {b['dataset']['name']} {stage2_semantic_hash(b)[:12]}")
    for path in sorted((root / "stage3").glob("*.yaml")):
        if int(load_yaml(path).get("stage", -1)) != 3:
            continue
        b = load_stage3_bundle(path, args.common, args.registry)
        console.print(f"[green]OK[/green] stage3 {b['dataset']['name']} {stage3_semantic_hash(b)[:12]}")
    for path in sorted((root / "stage4").glob("*.yaml")):
        b = load_mixture_bundle(path, args.common, args.registry)
        output_names = sorted((b["mixture"].get("outputs") or {"train": {}}))
        for output_name in output_names:
            output = resolve_output_spec(b["mixture"], output_name)
            console.print(f"[green]OK[/green] stage4-output {output_name} {output['target_tokens']:,} tokens")
        console.print(f"[green]OK[/green] stage4 {b['mixture']['name']} {b['mixture_hash'][:12]}")

    if args.production_stage4:
        b = load_mixture_bundle(args.production_stage4, args.common, args.registry)
        mix = b["mixture"]
        if any(str(mix.get(field, "")).startswith("REPLACE_") for field in ("tokenizer_name_or_path", "eos_token")):
            raise ValueError("Production Stage-4 mixture still contains tokenizer or EOS placeholders")
        for info in b["sources"].values():
            stage3 = info["stage3_bundle"]
            require_frozen_benchmark(stage3["benchmark"])
            require_frozen_sealed_evaluation(stage3["sealed_evaluation"], stage3["dataset"]["repos"]["stage3"])
        console.print(f"[green]OK[/green] production Stage-4 contract {mix['name']} {b['mixture_hash'][:12]}")


if __name__ == "__main__":
    main()
