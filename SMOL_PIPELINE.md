# Smol Data cleaning pipeline

This is the new lightweight path for Smol Data sources. It deliberately has only two data-processing stages:

1. Stage 1 streams one source from the Hugging Face Hub, applies row-level filters, and uploads compressed Parquet shards to that source's Stage-1 dataset repository.
2. Stage 2 streams those Stage-1 repositories one at a time, samples them according to the configured weights, and uploads the final training Parquet shards.

The pipeline does not download a whole source to local disk, does not run the old multi-pass corpus deduplication flow, and does not mix raw sources before filtering.

## Environment

Run from the `data_clean` directory:

```bash
export HF_USER=dignity045
export HF_TOKEN=hf_your_write_token
pip install -r requirements-smol.txt
```

On Kaggle, initialize the same values with Python instead of `export`:

```python
import os
os.environ["HF_USER"] = "dignity045"
os.environ["HF_TOKEN"] = "hf_your_write_token"
```

Do not commit the token to a notebook or repository. Prefer a Kaggle secret and assign its value to `HF_TOKEN`.

## Stage 1: one source at a time

First inspect the source schema without creating the output repository:

```bash
python -u smol_stage1_filter.py \
  --config configs/smol/stage1/dclm.yaml \
  --dry-run
```

Run a small smoke test before a larger job:

```bash
python -u smol_stage1_filter.py \
  --config configs/smol/stage1/dclm.yaml \
  --limit-rows 10000
```

Then run each source separately. The following commands are intentionally sequential:

```bash
python -u smol_stage1_filter.py --config configs/smol/stage1/dclm.yaml
python -u smol_stage1_filter.py --config configs/smol/stage1/fineweb_edu.yaml
python -u smol_stage1_filter.py --config configs/smol/stage1/finepdfs_edu.yaml
```

The filters are in the individual YAML files. They currently require English, use a minimum language score of `0.95`, require at least 50 words and 200 characters, and apply source-specific quality fields when available. `missing_policy: ignore` means a source without a particular optional score is not rejected for that missing score; `missing_policy: reject` is used for DCLM's required score fields.

## Stage 2: weighted training mixture

After the three Stage-1 repositories contain Parquet files under `data/v1`, preview the mixture:

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

The initial weights are 50% FinePDFs-Edu, 30% DCLM, and 20% FineWeb-Edu. Change only the `weight` values in `configs/smol/stage2/mix.yaml` to try another mixture. The output is `HF_USER/laughlm-smol-v1-training` with compressed Parquet shards and a manifest.

## Important scope

This first version filters and mixes; it does not perform cross-source deduplication. That is intentional so the pipeline remains practical on a CPU notebook. Validate the resulting mixture and add a separate deduplication pass only if the smoke statistics show it is needed.
