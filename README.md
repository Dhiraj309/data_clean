# LaughLM HQ Dataset Pipeline v5

A source-aware, stage-separated pipeline for building a 20–50B-token LaughLM pretraining corpus from curated Hugging Face datasets.

See [`ROADMAP.md`](ROADMAP.md) for the data-pipeline implementation milestones
and the ownership boundary with LaughLM.

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
- `configs/stage3/benchmarks.yaml`: LaughLM evaluation/decontamination suite draft;
  freeze it before production Stage 3.
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

Committed Stage 1-3 manifests include `processing_status: "committed"`, source
or input-part details, and streaming checksum/byte-count records for every
output part. Resume checks validate the artifact contract, committed status,
output detail alignment, and referenced output-file presence before skipping a
source. An old, incomplete, or partially uploaded manifest is reprocessed
safely under the same semantic output prefix.

Processing failures write a `processing_status: "failed"` manifest with the
exception type/message and retry counters. Failed manifests are intentionally
not treated as commit markers, so the next run retries the source.

Audit committed manifests without processing data:

```bash
python audit_manifests.py \
  --repo-id YOUR_STAGE1_DATASET_REPO \
  --stage stage1 \
  --output reports/stage1_manifest_audit.json
```

Add `--verify-checksums` when a full output download-and-hash audit is worth
the extra bandwidth and time.

After Stage 3, audit split counts and ensure duplicate families do not cross
split boundaries:

```bash
python audit_splits.py \
  --repo-id YOUR_STAGE3_DATASET_REPO \
  --output reports/stage3_split_audit.json
```

Stage 2 records exact and normalized hash sidecars in the shared dedup
namespace. Optional SimHash near-duplicate rejection is controlled by
`deduplication.near_duplicate.enabled` and is disabled by default until
its false-positive rate is reviewed. Dataset `source_priority` values are
recorded in manifests; higher-priority sources should be processed first when
building a shared dedup namespace.

Stage 3 assigns `train`, `validation`, `test`, `held_out_source`, `temporal`,
`synthetic`, or `sealed` using the precedence in `configs/common.yaml`. Stage
4 mixtures explicitly declare `allowed_splits` and default to consuming only
`train`.

Contamination audit against the frozen sealed repository:

```bash
python audit_contamination.py \
  --training-repo-id YOUR_STAGE3_DATASET_REPO \
  --sealed-config configs/stage3/sealed_evaluation.yaml \
  --output reports/contamination.json
```

The audit reports exact, normalized, n-gram, and near-duplicate matches. It
rejects placeholder or same-repository sealed configurations.

### Logical-output invariance

Resource profiles may change worker counts, VM sizes, cache paths, retry
timing, and execution timestamps, but they must not change accepted content,
deduplication, split assignment, quotas, lineage, or output checksums. Compare
two artifact snapshots after running the same semantic configs:

```bash
python invariance_audit.py \
  --left-root path/to/profile_a \
  --right-root path/to/profile_b \
  --output reports/invariance.json
```

The audit ignores operational fields only; a changed content hash, count,
split, contract, lineage entry, or shard record fails the comparison. Stage 2
namespace reconstruction also processes sources in deterministic priority and
path order rather than relying on HF listing order.

Stage 4 freezes tokenizer, EOS, packing, shard-size, dtype, and split-selection
contracts in `corpus_manifest.json`. Completed shards update a remote
`progress.json`; rerunning the same mixture resumes from the last committed
shard. Use `--fresh` only when intentionally replacing the same mixture run.
The final manifest also records source quotas, domain/time labels, actual token
exposure, and upstream Stage-3 hashes.

The shared runtime profile lives in `configs/common.yaml`:

```yaml
runtime:
  file_workers: 2
  download_workers: 1
  upload_workers: 1
  download_queue_size: 2
  upload_queue_size: 2
  max_inflight_files: 2
  batch_rows: 4096
  min_free_disk_gb: 5.0
  local_cache_dir: null
  local_temp_dir: null
```

Stage 1 uses bounded file scheduling and the configured temporary/cache roots.
Stage 2 and Stage 3 remain source-sequential because their deduplication and
decontamination state is global. Separate download/upload worker pools are
reserved for a later overlapped-I/O milestone; they are not implied to be
active by the current profile.

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

and fill `lighteval_tasks` with the final LaughLM evaluation suite. Freeze the
benchmark and sealed repository contract without mutating the drafts:

```bash
python freeze_benchmark.py \
  --benchmark-config configs/stage3/benchmarks.yaml \
  --sealed-config configs/stage3/sealed_evaluation.yaml \
  --training-repo-id dignity045/laughlm-fineweb-edu-hq \
  --benchmark-output configs/stage3/benchmarks.frozen.yaml \
  --sealed-output configs/stage3/sealed_evaluation.frozen.yaml \
  --manifest-output reports/benchmark_freeze.json
```

The command fails until the task IDs, sealed repository revision, explicit
sealed file paths, and training-repository identity are real and complete.
Update each Stage-3 config to reference `benchmarks.frozen.yaml` before
processing. The resulting freeze manifest records the benchmark and sealed
configuration hashes.

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
Optional `domain_quotas` and `time_quotas` maps can require exact token budgets
for source metadata labels. A source may declare `domain` and either
`time_bucket` or a labeled `time_range`; the scheduler only selects sources
whose source, domain, and time budgets all have remaining capacity. An enabled
quota with missing or inconsistent labels fails before processing, rather than
silently producing an unbalanced mixture. Empty maps preserve source-only
quota behavior:

```yaml
domain_quotas: {}
time_quotas: {}
```

Validate availability without tokenizing:

```bash
python stage4_build.py --config configs/stage4/laughlm_hq_20b.yaml --dry-run
```

Build:

```bash
python stage4_build.py --config configs/stage4/laughlm_hq_20b.yaml
```

Stage 4 interleaves documents across source datasets using remaining source and
optional domain/time quotas, tokenizes them, adds EOS, enforces exact quotas,
and writes memory-mappable little-endian uint16/uint32 `.bin` shards. The final
manifest records declared and observed exposure for each quota dimension.

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

## M6: release handoff bundle

After Stage 4 and its audits are complete, create a portable handoff bundle
from the checked-in pipeline plus the manifests and reports produced by the
run:

```bash
python release_bundle.py \
  --output releases/laughlm-hq-20b \
  --manifest path/to/stage1_manifest.json \
  --manifest path/to/stage2_manifest.json \
  --manifest path/to/stage3_manifest.json \
  --final-manifest path/to/corpus_manifest.json \
  --report path/to/manifest_audit.json
```

The bundle contains the source snapshot, all checked-in configs, supplied
manifests/reports, reproducible stage commands, the runtime/retry profile,
`release_manifest.json` with SHA-256 checksums, and
`laughlm_dataset_release_v1.json`. LaughLM should consume the latter only when
its status is `ready`; without a committed Stage-4 manifest the bundle remains
an explicit `pending_final_manifest` handoff.

## Source-specific notes

- FineWeb-Edu config is the most concrete pilot and retains score/language metadata filtering.
- FinePDFs-Edu registry currently narrows ingestion to the English `eng_Latn` directory and preserves all upstream OCR/quality/dedup metadata in Stage 1.
- FineMath Stage 1 is intentionally permissive until the exact desired subset/path (for example 4+) is frozen; narrow `source_patterns` before the production run.
- Stack-Edu gets an additional high-confidence secret filter in Stage 2. Verify the selected HF release actually exposes usable text/code rather than only source identifiers before a production run.
- Dolma 3 ingestion supports `.jsonl.zst`; the example excludes directories explicitly labeled `adult_content` at Stage 1.

## Oracle A1 operational guidance

Start with `--workers 2` on Stage 1. File workers, download workers, and upload
workers are independently bounded; the download/upload queue sizes apply
backpressure before another network operation is admitted. The worker buffers
are intentionally bounded by the Stage-1 shard target, and Stage 2/3 process
one committed source at a time. Increase concurrency only after observing RAM,
network, and local-disk usage.
