# Smol Data cleaning pipeline

This is the new lightweight path for Smol Data sources. It deliberately has only two data-processing stages:

1. Stage 1 lists the source Parquet files, downloads a bounded number of files, filters their Parquet row groups with one shared CPU process pool, and merges completed files in source order into one persistent domain buffer. Full approximately 1 GiB Parquet shards are uploaded to the filtered-data repository.
2. Stage 2 streams those domain folders, samples them according to the configured weights, and uploads the final training Parquet shards.

The pipeline keeps only the bounded set of active source files on local disk, removes each download after its staging checkpoint completes, does not run the old multi-pass corpus deduplication flow, and does not mix raw sources before filtering.

## Minimal live logging

In Jupyter, use `%run smol_stage1_filter.py ...` and `%run smol_stage2_mix.py ...` as demonstrated in `smol_data_pipeline.ipynb`. `%run` executes in the active kernel and lets Rich update one table in place. Running through `!python` creates a subprocess, so notebook frontends may append each table refresh instead.

Stage 1 keeps one live box on screen showing each active source file, its status, rows seen, accepted rows, and acceptance percentage. It prints one concise `Buffered` line per completed source and one `Pushed` line only when a full shard is created. Stage 2 uses one live box for total rows, estimated tokens, current source, and actual versus target mixture percentages, plus one line per uploaded output shard. Hugging Face download and upload progress bars are disabled by default so notebook logs stay compact.

## Environment

From a fresh machine or Kaggle session, first clone the repository that contains this pipeline:

```bash
git clone https://github.com/Dhiraj309/data_clean.git
cd data_clean
```

If the repository already exists locally, use `git pull` instead. The working tree must contain `smol_stage1_filter.py`, `smol_stage2_mix.py`, and `configs/smol/` before running the commands below.

Run from the `data_clean` directory:

```bash
export HF_USER=dignity045
export HF_TOKEN=hf_your_write_token
export FILTERED_REPO=$HF_USER/LaughLM-Filtered-Smol
pip install -r requirements-smol.txt
```

On Kaggle, initialize the same values with Python instead of `export`:

```python
import os
os.environ["HF_USER"] = "dignity045"
os.environ["HF_TOKEN"] = "hf_your_write_token"
os.environ["FILTERED_REPO"] = f'{os.environ["HF_USER"]}/LaughLM-Filtered-Smol'
```

Do not commit the token to a notebook or repository. Prefer a Kaggle secret and assign its value to `HF_TOKEN`.

## Stage 1: bounded hybrid processing

First inspect the source schema without creating the output repository:

```bash
python -u smol_stage1_filter.py \
  --config configs/smol/stage1/dclm.yaml \
  --dry-run
```

Run a small smoke test before a larger job. The `smoke` run ID keeps the test output separate from the real `v1` run:

```bash
python -u smol_stage1_filter.py \
  --config configs/smol/stage1/dclm.yaml \
  --limit-files 1 \
  --limit-rows 10000 \
  --workers 2 \
  --active-files 1 \
  --workers-per-file 2 \
  --run-id smoke \
  --no-resume
```

Then run each source separately. The following commands are intentionally sequential:

```bash
python -u smol_stage1_filter.py --config configs/smol/stage1/dclm.yaml --workers 24 --active-files 2 --workers-per-file 12
python -u smol_stage1_filter.py --config configs/smol/stage1/fineweb_edu.yaml --workers 24 --active-files 2 --workers-per-file 12
python -u smol_stage1_filter.py --config configs/smol/stage1/finepdfs_edu.yaml --workers 24 --active-files 2 --workers-per-file 12
```

To process files in deterministic batches, use the zero-based `--start-file` offset together with `--limit-files`. For batches of 20 files:

```bash
# files 0..19
python -u smol_stage1_filter.py --config configs/smol/stage1/dclm.yaml --start-file 0 --limit-files 20

# files 20..39
python -u smol_stage1_filter.py --config configs/smol/stage1/dclm.yaml --start-file 20 --limit-files 20

# files 40..59
python -u smol_stage1_filter.py --config configs/smol/stage1/dclm.yaml --start-file 40 --limit-files 20
```

The file list is sorted before slicing, and checkpoint indices use the full list, so the next batch continues with `N+1` rather than starting over. If a command stops, rerun the same command without `--no-resume`; completed files, the rolling buffer, and checkpoint manifests are reused.

All Stage-1 datasets share `FILTERED_REPO` and use flat domain folders:

```text
LaughLM-Filtered-Smol/
├── dclm/dclm_shard_00000.parquet
├── fineweb-edu/fineweb-edu_shard_00000.parquet
├── finepdfs-edu/finepdfs-edu_shard_00000.parquet
└── _checkpoints/stage1/v1/...
```

`--workers` is the total CPU process budget. `--active-files` bounds source-file concurrency, and `--workers-per-file` caps the row groups from each file that may occupy that pool. When these values are omitted, Stage 1 chooses one active file below 8 workers, two active files from 8 through 48 workers, and at most four above that. `--max-inflight-files` remains a backward-compatible alias for `--active-files`.

Row-group workers write 128 MiB local staging pieces so filtering remains memory-safe. They never write the shared buffer and never upload. The coordinator consumes completed sources in deterministic source order and is the only owner of `buffer.parquet`, shard numbering, and Hub uploads. Filtering of other active files continues while the coordinator compacts or uploads a full shard. Each row-group result has its own manifest, so an interrupted file resumes only its unfinished row groups. Whenever the compressed buffer crosses the configured 1024 MiB target, it becomes `domain_shard_00000.parquet`, `domain_shard_00001.parquet`, and so on; only the remainder stays in `buffer.parquet`. Because Parquet compression and row groups are indivisible, full shards are close to 1 GiB rather than byte-for-byte identical.

The filters are in the individual YAML files. They currently require English, use a minimum language score of `0.95`, require at least 50 words and 200 characters, and apply source-specific quality fields when available. `missing_policy: ignore` means a source without a particular optional score is not rejected for that missing score; `missing_policy: reject` is used for DCLM's required score fields.

## Stage 2: weighted training mixture

After the three domain folders in `FILTERED_REPO` contain Parquet files, preview the mixture:

```bash
python -u smol_stage2_mix.py \
  --config configs/smol/stage2/mix.yaml \
  --dry-run
```

Run a smoke mixture:

```bash
python -u smol_stage2_mix.py \
  --config configs/smol/stage2/mix.yaml \
  --limit-rows 10000
```

Then produce the configured training set:

```bash
python -u smol_stage2_mix.py \
  --config configs/smol/stage2/mix.yaml
```

Stage 2 uses the same rolling layout: approximately 1 GiB files named `smol-v1_shard_00000.parquet`, followed by one `train/buffer.parquet` remainder. It checkpoints after every 128 MiB staging commit and uploads the current remainder, so rerunning the same command resumes the deterministic mixture. A restart may replay already-read Stage-1 rows to reconstruct the sampler state, but it will not create a second output sequence.

The initial weights are 50% FinePDFs-Edu, 30% DCLM, and 20% FineWeb-Edu. Change only the `weight` values in `configs/smol/stage2/mix.yaml` to try another mixture. The output is `HF_USER/laughlm-smol-v1-training/train/smol-v1_shard_00000.parquet` and subsequent compressed Parquet shards.

## Important scope

This first version filters and mixes; it does not perform cross-source deduplication. That is intentional so the pipeline remains practical on a CPU notebook. Validate the resulting mixture and add a separate deduplication pass only if the smoke statistics show it is needed.
