# Smol Data cleaning pipeline

This is the new lightweight path for Smol Data sources. It deliberately has only two data-processing stages:

1. Stage 1 lists the source Parquet files, processes one source file per worker, applies row-level filters, and uploads all compressed Parquet shards to domain folders in one filtered-data repository.
2. Stage 2 streams those domain folders, samples them according to the configured weights, and uploads the final training Parquet shards.

The pipeline does not download a whole source to local disk, does not run the old multi-pass corpus deduplication flow, and does not mix raw sources before filtering.

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

## Stage 1: one source at a time

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
  --max-inflight-files 1 \
  --run-id smoke \
  --no-resume
```

Then run each source separately. The following commands are intentionally sequential:

```bash
python -u smol_stage1_filter.py --config configs/smol/stage1/dclm.yaml --workers 12 --max-inflight-files 6
python -u smol_stage1_filter.py --config configs/smol/stage1/fineweb_edu.yaml --workers 12 --max-inflight-files 6
python -u smol_stage1_filter.py --config configs/smol/stage1/finepdfs_edu.yaml --workers 12 --max-inflight-files 6
```

To process only the first 20 source files, add `--limit-files 20`. If a command stops, rerun the same command without `--no-resume`; completed file shards and checkpoint manifests are reused.

All Stage-1 datasets share `FILTERED_REPO` and use flat domain folders:

```text
LaughLM-Filtered-Smol/
├── dclm/dclm_shard_00000.parquet
├── fineweb-edu/fineweb-edu_shard_00000.parquet
├── finepdfs-edu/finepdfs-edu_shard_00000.parquet
└── _checkpoints/stage1/v1/...
```

The number identifies the original source file. If a large source file creates multiple 500 MB outputs, later parts use names such as `finepdfs-edu_shard_00000_001.parquet`. This deterministic naming prevents collisions when multiple source files run in parallel and ensures retries overwrite the same shard rather than duplicate rows. Each domain folder also receives `progress.json`. Partial buffers remain local until they form an uploadable Parquet shard.

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

Stage 2 also writes a local pending buffer and a checkpoint manifest to the output HF repository after each uploaded shard. If it stops, rerun the same command to continue the deterministic mixture from the last completed output shard. A restart may replay already-read Stage-1 rows to reconstruct the deterministic sampler state, but it will not create a second output shard sequence.

The initial weights are 50% FinePDFs-Edu, 30% DCLM, and 20% FineWeb-Edu. Change only the `weight` values in `configs/smol/stage2/mix.yaml` to try another mixture. The output is `HF_USER/laughlm-smol-v1-training/train/smol-v1_shard_00000.parquet` and subsequent compressed Parquet shards.

## Important scope

This first version filters and mixes; it does not perform cross-source deduplication. That is intentional so the pipeline remains practical on a CPU notebook. Validate the resulting mixture and add a separate deduplication pass only if the smoke statistics show it is needed.
