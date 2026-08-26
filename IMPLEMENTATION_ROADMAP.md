# LaughLM Mixed-Data POC Implementation Roadmap

This roadmap tracks the current plan for the 135M model and 20B-token POC.
The order is intentional: make `data_clean` capable of producing a valid,
traceable mixed corpus first, then connect it to LaughLM while larger data
processing continues.

Status markers:

- `[ ]` not started
- `[~]` partial or in progress
- `[x]` complete and verified
- `[d]` intentionally deferred

## Target workflow

```text
HF source datasets
    -> Stage 1 bounded hybrid filtering by file and Parquet row group
    -> fixed-size filtered Parquet shards
    -> Stage 2 token-aware mixing
    -> tokenizer and train/validation manifests
    -> LaughLM mixed-shard loader
    -> 100M/1B-token gate
    -> 20B-token POC
```

## Scope decisions

| Decision | POC policy |
|---|---|
| Dataset source | Use upstream HF Smol/pretraining datasets |
| Processing | Bound active files and parallelize their Parquet row groups with resumable checkpoints |
| Filter output | Fixed-size Parquet shards plus a resumable buffer |
| Mixing | Match target percentages by tokens, not document rows |
| Deduplication | Use upstream deduplication and lightweight local exact hashing |
| Global near-deduplication | `[d]` Deferred because the current CPU environment is not suitable |
| Stack-Edu | Include only after source text is available; otherwise exclude explicitly |
| First training gate | 100M–1B tokens before the full 20B run |

## Phase 1 — `data_clean` foundation

### DC-1: Finalize source contract

- `[ ]` Confirm the final dataset list and target weights.
- `[ ]` Confirm HF repository revisions and source subsets.
- `[ ]` Confirm actual field names for every selected dataset.
- `[ ]` Record which datasets are text-ready and which require special handling.
- `[ ]` Decide whether Stack-Edu is included in the first POC.

### DC-2: Stage 1 filtering

- `[x]` Bound active files and share one CPU process budget across per-file row-group work.
- `[x]` Keep resumable source-file and row-group checkpoints.
- `[~]` Keep fixed-size filtered Parquet shards and `buffer.parquet`.
- `[~]` Keep compact single-table progress logging.
- `[ ]` Verify every dataset-specific YAML threshold against the source schema.
- `[ ]` Record accepted, rejected and rejection-reason counts.
- `[ ]` Record word, character, estimated-token and exact-token statistics.
- `[ ]` Add lightweight exact hashing within the current file/buffer.
- `[ ]` Add malformed, empty and missing-text counters.

### DC-3: Stage 2 token-aware mixing

- `[~]` Keep resumable rolling output shards.
- `[~]` Keep source provenance in every mixed row.
- `[ ]` Replace row-based mixture reporting with token-based reporting.
- `[ ]` Track `tokens_by_source` in the state and live table.
- `[ ]` Track target percentage versus actual percentage by tokens.
- `[ ]` Stop mixing against a token budget, not only estimated row totals.
- `[ ]` Validate that exhausted sources do not distort the final mixture.

### DC-4: Special and deferred sources

- `[ ]` Implement Stack-Edu `blob_id` to source-text reconstruction, or exclude it.
- `[ ]` Confirm FineMath and MegaMath text/token fields.
- `[ ]` Confirm Cosmopedia-v2 text and metadata handling.
- `[ ]` Treat OpenMathReasoning and OpenCodeReasoning as optional later-stage sources.

### DC-5: Tokenization and manifests

- `[ ]` Tokenize the mixed Parquet output with the training tokenizer.
- `[ ]` Produce fixed-size token shards.
- `[ ]` Produce exact train and validation token counts.
- `[ ]` Produce train/validation manifests with source provenance.
- `[ ]` Verify tokenizer hash, vocabulary size, EOS ID and sequence length.
- `[ ]` Verify no train/validation shard overlap.

### DC-6: `data_clean` completion gate

The Data Clean phase is ready for full processing when:

- `[ ]` one small batch from every selected source completes successfully;
- `[ ]` Stage 1 can resume after interruption;
- `[ ]` Stage 2 reports actual token percentages;
- `[ ]` mixed Parquet files tokenize successfully;
- `[ ]` train/validation manifests pass validation;
- `[ ]` source counts and token totals reconcile.

At this point, start the larger Data Clean processing run and begin Phase 2
in parallel.

## Phase 2 — LaughLM changes in parallel

### LL-1: Mixed-corpus ingestion

- `[ ]` Point the trainer to the mixed tokenized repository.
- `[ ]` Support the final train/validation manifest contract.
- `[ ]` Preserve source/domain provenance at manifest or shard level.
- `[ ]` Report source exposure by tokens during training.
- `[ ]` Verify the loader consumes every shard exactly once as intended.

### LL-2: Numerical and training diagnostics

- `[~]` Keep loss, perplexity, learning rate, gradient norm and tokens/sec logging.
- `[ ]` Add parameter norm logging.
- `[ ]` Add update norm and update/parameter ratio logging.
- `[ ]` Add explicit NaN/Inf detection and failure behavior.
- `[ ]` Enable sparse training-integrity checks for the POC.
- `[ ]` Verify token IDs, labels, EOS, padding and ignore-index handling.
- `[ ]` Verify packed-document boundaries and sequence shapes.

### LL-3: Scheduler and checkpoint safety

- `[~]` Keep the WSD schedule with 1% warmup, 95% stable phase and 5% minimum LR.
- `[ ]` Verify schedule behavior by cumulative tokens.
- `[~]` Keep optimizer, scheduler, RNG and token-counter checkpoint metadata.
- `[ ]` Run checkpoint/resume equivalence testing.
- `[ ]` Verify resumed loader/sampler state does not repeat or skip data unexpectedly.

### LL-4: Validation and evaluation

- `[~]` Keep fixed validation-loss evaluation.
- `[ ]` Add domain-specific validation sets for web, education, PDF, code, math and Q&A.
- `[ ]` Freeze benchmark tasks and evaluation settings.
- `[ ]` Add evaluation checkpoints at early training, 2B, 5B, 10B, 15B and 20B tokens.
- `[ ]` Add lightweight benchmark-contamination checks.

### LL-5: Runtime profiling

- `[~]` Keep device, host, transfer and data-wait timing.
- `[ ]` Measure cold compile time separately from steady-state throughput.
- `[ ]` Record median and percentile tokens/sec.
- `[ ]` Record checkpoint and evaluation overhead.
- `[ ]` Produce a measured 20B runtime estimate.

## Phase 3 — Integration gates

### INT-1: Small end-to-end smoke test

- `[ ]` Filter a small batch from every selected source.
- `[ ]` Mix a small token budget.
- `[ ]` Tokenize and build manifests.
- `[ ]` Train for a few steps from the mixed shards.
- `[ ]` Confirm metrics, validation, checkpointing and resume.

### INT-2: 100M-token gate

- `[ ]` Complete a 100M-token run with no data or numerical failures.
- `[ ]` Confirm actual source exposure matches the configured token mixture.
- `[ ]` Confirm validation loss behaves normally.
- `[ ]` Confirm checkpoint/resume equivalence.
- `[ ]` Save the gate report.

### INT-3: 1B-token gate

- `[ ]` Run the same configuration to 1B tokens.
- `[ ]` Evaluate early capability and domain metrics.
- `[ ]` Review throughput and projected 20B cost/time.
- `[ ]` Decide whether to proceed to 20B.

## Phase 4 — Full 20B-token POC

- `[ ]` Freeze source revisions, filter configs, tokenizer and manifests.
- `[ ]` Freeze the training configuration and WSD horizon.
- `[ ]` Start the 20B run from a clean checkpoint directory.
- `[ ]` Monitor loss, numerical health, source exposure and validation loss.
- `[ ]` Run scheduled benchmark checkpoints.
- `[ ]` Produce the final POC report.

## Final report requirements

- `[ ]` Source revisions and filter configuration hashes.
- `[ ]` Accepted/rejected counts and rejection reasons.
- `[ ]` Exact and estimated token totals.
- `[ ]` Target versus actual token mixture.
- `[ ]` Duplicate policy and limitations.
- `[ ]` Train/validation manifest details.
- `[ ]` Loss, LR, gradient, parameter and update-norm curves.
- `[ ]` NaN/Inf and integrity results.
- `[ ]` Validation and benchmark results.
- `[ ]` Throughput, checkpoint overhead and projected runtime.
- `[ ]` Known limitations, including deferred global near-deduplication.

## Estimated schedule

| Workstream | Expected effort | Can run in parallel? |
|---|---:|---|
| Data Clean MVP | 1–2 working days | No; first gate |
| Data Clean POC-ready pipeline | 3–5 working days total | Starts processing after MVP |
| LaughLM changes | 3–5 working days | Yes, after Data Clean MVP |
| Integration and smoke gates | 1–2 working days | Partly |
| Stack-Edu reconstruction, if required | Add 1–3 working days | Optional |

The actual dataset processing time is separate from implementation time and
can continue unattended after the Data Clean MVP gate passes.
