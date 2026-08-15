# LaughLM HQ Dataset Pipeline v5

A source-aware, stage-separated pipeline for building a 20–50B-token LaughLM pretraining corpus from curated Hugging Face datasets.

## Folder tree

```text
laughlm_dataset_pipeline_v5/
├── README.md
├── requirements.txt
├── config_loader.py
├── pipeline_utils.py
├── filters.py
├── validate_configs.py
├── stage1_filter.py
├── stage2_process.py
├── stage3_decontam.py
├── stage4_build.py
└── configs/
    ├── common.yaml
    ├── datasets.yaml
    ├── stage1/
    │   ├── fineweb_edu.yaml
    │   ├── finepdfs_edu.yaml
    │   ├── finemath.yaml
    │   ├── stack_edu.yaml
    │   └── dolma3_150b.yaml
    ├── stage2/
    │   ├── fineweb_edu.yaml
    │   ├── finepdfs_edu.yaml
    │   ├── finemath.yaml
    │   ├── stack_edu.yaml
    │   └── dolma3_150b.yaml
    ├── stage3/
    │   ├── benchmarks.yaml
    │   ├── fineweb_edu.yaml
    │   ├── finepdfs_edu.yaml
    │   ├── finemath.yaml
    │   ├── stack_edu.yaml
    │   └── dolma3_150b.yaml
    └── stage4/
        ├── laughlm_hq_20b.yaml
        └── laughlm_hq_50b.yaml
```

## Configuration responsibilities

- `configs/common.yaml`: runtime/HF/storage/retry/compression settings only.
- `configs/datasets.yaml`: dataset identity, source repo/format, and durable Stage-1/2/3 HF repos only.
- `configs/stage1/<dataset>.yaml`: source-file selection + metadata filters + output columns.
- `configs/stage2/<dataset>.yaml`: dataset-specific cleanup, custom filters, PII, and shared dedup namespace.
- `configs/stage3/benchmarks.yaml`: ONE frozen LaughLM evaluation/decontamination suite.
- `configs/stage3/<dataset>.yaml`: exact Stage-2 config + benchmark config lineage.
- `configs/stage4/<mixture>.yaml`: final source quotas, tokenizer, EOS, token budget, output repo.

## Data flow

```text
Original HF source
    ↓
Stage 1 YAML for that dataset
    ↓
Your dataset-specific Stage-1 HF repo
    ↓
Stage 2 YAML for that dataset
    ↓  (shared dedup_namespace across all datasets)
Your dataset-specific Stage-2 HF repo
    ↓
Stage 3 YAML for that dataset + shared benchmarks.yaml
    ↓
Your dataset-specific final text/HQ HF repo
    ↓
Stage 4 mixture YAML
    ↓
LaughLM-HQ-20B / LaughLM-HQ-50B token repo
```

Stages 1–3 are run separately for every source dataset. Stage 4 is run once after the desired source datasets are ready.

## Install

```bash
python --version  # DataTrove 0.9.0 requires Python >= 3.10
python -m pip install -r requirements.txt
export HF_TOKEN=hf_...
python validate_configs.py
```

## Pilot FineWeb-Edu

```bash
python stage1_filter.py \
  --config configs/stage1/fineweb_edu.yaml \
  --limit-files 10 \
  --workers 2

python stage2_process.py \
  --config configs/stage2/fineweb_edu.yaml \
  --limit-sources 10
```

Stage 1 and Stage 2 use remote `manifest.json` files as commit markers. Re-running the same config skips committed source units.

## Process each dataset independently

FinePDFs-Edu:

```bash
python stage1_filter.py --config configs/stage1/finepdfs_edu.yaml
python stage2_process.py --config configs/stage2/finepdfs_edu.yaml
```

FineMath:

```bash
python stage1_filter.py --config configs/stage1/finemath.yaml
python stage2_process.py --config configs/stage2/finemath.yaml
```

Stack-Edu:

```bash
python stage1_filter.py --config configs/stage1/stack_edu.yaml
python stage2_process.py --config configs/stage2/stack_edu.yaml
```

Dolma 3 150B sample (JSONL+zstd ingestion is supported):

```bash
python stage1_filter.py --config configs/stage1/dolma3_150b.yaml
python stage2_process.py --config configs/stage2/dolma3_150b.yaml
```

## Cross-dataset exact dedup

Every Stage-2 config currently uses:

```yaml
dedup_namespace: "laughlm-hq-v1"
```

The local DB is automatically stored at:

```text
./work/dedup/laughlm-hq-v1.sqlite3
```

Before processing a dataset, Stage 2 scans registered Stage-2 repos for committed manifests in the same namespace and reconstructs the local exact-hash DB from their `accepted.hashes` sidecars. HF therefore remains the durable dedup record.

**Important:** treat the namespace as immutable. If you materially change Stage-2 filtering rules and want a fresh corpus generation, bump ALL participating Stage-2 configs to e.g. `laughlm-hq-v2`.

Processing order determines which identical copy wins. Process preferred/highest-quality sources first if exact duplicates overlap.

## Stage 3: benchmark decontamination

First edit:

```text
configs/stage3/benchmarks.yaml
```

and fill `lighteval_tasks` with the frozen LaughLM evaluation suite.

Build the shared index once:

```bash
python stage3_decontam.py \
  --config configs/stage3/fineweb_edu.yaml \
  --build-index \
  --index-only
```

Then run each source:

```bash
python stage3_decontam.py --config configs/stage3/fineweb_edu.yaml
python stage3_decontam.py --config configs/stage3/finepdfs_edu.yaml
python stage3_decontam.py --config configs/stage3/finemath.yaml
python stage3_decontam.py --config configs/stage3/stack_edu.yaml
python stage3_decontam.py --config configs/stage3/dolma3_150b.yaml
```

The benchmark hash is part of Stage-3 lineage, so changing `benchmarks.yaml` creates a new Stage-3 run rather than silently mixing results.

## Stage 4: combine datasets only here

Freeze the LaughLM tokenizer and EOS first, then edit either:

```text
configs/stage4/laughlm_hq_20b.yaml
configs/stage4/laughlm_hq_50b.yaml
```

Each source entry points to an exact Stage-3 config and has an exact token quota.

Validate availability without tokenizing:

```bash
python stage4_build.py --config configs/stage4/laughlm_hq_20b.yaml --dry-run
```

Build:

```bash
python stage4_build.py --config configs/stage4/laughlm_hq_20b.yaml
```

Stage 4 interleaves documents across source datasets using remaining token quotas, tokenizes them, adds EOS, enforces exact quotas, and writes memory-mappable little-endian uint16/uint32 `.bin` shards.

Final output layout:

```text
<final HF repo>/
├── ACTIVE.json
└── runs/<mixture-hash>/
    ├── corpus_manifest.json
    └── tokens/
        ├── shard_00000.bin
        ├── shard_00001.bin
        └── ...
```

A changed mixture config produces a different run hash, so old final corpora remain reproducible.

## Source-specific notes

- FineWeb-Edu config is the most concrete pilot and retains score/language metadata filtering.
- FinePDFs-Edu registry currently narrows ingestion to the English `eng_Latn` directory and preserves all upstream OCR/quality/dedup metadata in Stage 1.
- FineMath Stage 1 is intentionally permissive until the exact desired subset/path (for example 4+) is frozen; narrow `source_patterns` before the production run.
- Stack-Edu gets an additional high-confidence secret filter in Stage 2. Verify the selected HF release actually exposes usable text/code rather than only source identifiers before a production run.
- Dolma 3 ingestion supports `.jsonl.zst`; the example excludes directories explicitly labeled `adult_content` at Stage 1.

## Oracle A1 operational guidance

Start with `--workers 2` on Stage 1. The worker buffers are intentionally bounded by the Stage-1 shard target, and Stage 2/3 process one committed source at a time. Increase concurrency only after observing RAM and local-disk usage.
