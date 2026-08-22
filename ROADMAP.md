# LaughLM Data Pipeline Roadmap

This repository owns corpus construction for LaughLM. The complete
cross-repository roadmap, including LaughLM training and evaluation work, is
maintained in `LaughLM/docs/data_pipeline/ROADMAP.md`.

Status flags: `[ ]` not started, `[~]` partial/in progress, `[x]` complete,
and `[d]` intentionally deferred.

## Ownership boundary

`data_clean` owns source ingestion, stage-wise processing, provenance,
deduplication, split assignment, decontamination, mixing, tokenization, and
final dataset manifests. LaughLM owns training, checkpointing, model
evaluation, memorization audits, and architecture comparisons.

## Milestones

### M0 — Shared artifact contract `[x]`

- [x] Freeze the shared manifest schema.
- [x] Define dataset, source-file, document, stage, run, and shard identities.
- [x] Define semantic config hashes and schema-version migration.
- [x] Guarantee logical-output invariance across worker counts and VM sizes.

### M1 — Bounded runtime execution `[x]`

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
  local_cache_dir: "/path/to/cache"
  local_temp_dir: "/path/to/ssd"
```

- [x] Add a versioned runtime profile with worker, batch, cache, and temp-root
  controls.
- [x] Preserve one-file-at-a-time processing within each worker.
- [x] Bound Stage 1 in-flight files.
- [x] Add bounded download/upload queues and backpressure.
- [x] Add disk-space preflight for the configured temporary root.
- [x] Record Stage 1 per-source download, processing, upload, and total timings.

### M2 — Provenance and resumability `[x]`

- [x] Record immutable source revision, file hash, size, and output checksums.
- [x] Record accepted, rejected, and duplicate counts by reason in committed manifests.
- [x] Persist failed-unit error manifests with error reasons while keeping the unit retryable.
- [x] Keep manifest-last commit semantics with explicit `processing_status`.
- [x] Resume only verified committed units with present output files.
- [x] Add manifest consistency auditing with optional checksum verification.

### M3 — Deduplication and deterministic splits `[x]`

- [x] Preserve the existing cross-dataset exact deduplication namespace.
- [x] Add normalized hashes and optional near-duplicate clustering.
- [x] Define source-priority metadata and winner policy for duplicate processing.
- [x] Assign train, validation, held-out-source, temporal, synthetic, and sealed
  splits after global deduplication.
- [x] Keep duplicate document families in one split through `split_group`.
- [x] Emit split and overlap reports.

### M4 — Decontamination and sealed evaluation `[~]`

- [x] Preserve the existing benchmark n-gram decontamination stage.
- [x] Add a versioned benchmark freeze contract and hash lineage.
- [x] Add a deterministic freeze command that emits benchmark/sealed hashes.
- [ ] Populate and freeze the final benchmark task list.
- [x] Add exact, normalized, n-gram, and near-duplicate contamination reports.
- [x] Keep sealed evaluation text outside the training input path.

### M5 — Mixing, tokenization, and packing `[x]`

- [x] Preserve deterministic source-quota mixing and exact token budgets.
- [x] Add source quotas, domain/time labels, and exposure statistics.
- [x] Freeze tokenizer, EOS, packing, shard, dtype, and split-selection contracts.
- [x] Add resumable shard-level tokenization and upload.
- [x] Preserve provenance from final `.bin` shards to source stages.
- [x] Enforce metadata-level domain/time quota selection when required by a mix.

### M6 — Pipeline release handoff `[x]`

- [x] Bundle all stage manifests, reports, configs, hashes, and checksums.
- [x] Provide reproducible commands for every stage.
- [x] Document resource profiles and operational recovery.
- [x] Produce the final manifest consumed by LaughLM.

### M7 - Stage-2 throughput and canonical execution `[~]`

- [x] Skip optional SimHash work when near-duplicate detection is disabled.
- [x] Record Stage-2 per-input-part download and processing timings.
- [x] Add a canonical corpus driver that enforces source-priority ordering.
- [x] Version the dedup namespace after the non-canonical smoke attempt.
- [ ] Add resumable parallel map/reduce execution for Stage 2 on multi-core VMs.
- [ ] Benchmark and freeze a 24-core resource profile against a representative corpus slice.

## Implementation order

M0 → M1 → M2 → M3 → M4 → M5 → M6.

These milestones do not require TPU execution. LaughLM TPU validation starts
only after the final dataset contract and manifests are available.
