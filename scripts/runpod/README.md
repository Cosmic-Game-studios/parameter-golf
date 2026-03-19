# Runpod Quickstart

These scripts make the current CUDA path easier to run on Runpod without reassembling long env blocks by hand.

The intent here is the real challenge path:

- official published FineWeb/docs cache
- official raw-doc rebuild for the `Unigram-4096` tokenizer
- no `local_u4k_unigram` proxy dataset
- no silent fallback to local exploratory data

## 1. Setup the workspace

From the repository root:

```bash
scripts/runpod/setup_workspace.sh
```

This creates `.venv` if needed, installs `requirements.txt`, and prints basic CUDA/PyTorch checks.

## 2. Build the official U4K docs export

```bash
scripts/runpod/rebuild_official_u4k_docs.sh
```

This does two things:

1. Downloads the published `sp1024` docs cache with `--with-docs`
2. Rebuilds the official `Unigram-4096` export under:

```text
data/official_u4k_unigram/datasets/fineweb10B_spu4096_docs
```

Useful overrides:

```bash
TRAIN_SHARDS=10 scripts/runpod/rebuild_official_u4k_docs.sh
FORCE_REBUILD=1 scripts/runpod/rebuild_official_u4k_docs.sh
```

This also leaves the published docs manifest at:

```text
data/official_u4k_unigram/docs_selected.source_manifest.json
```

The dense U4K launcher below refuses to run if that manifest is missing, or if you accidentally point it at `local_u4k_unigram`.

## 3. Run a control baseline

```bash
RUN_ID=runpod_baseline_sp1024 \
NPROC_PER_NODE=1 \
scripts/runpod/train_baseline_sp1024.sh
```

## 4. Run the current dense U4K frontier

```bash
RUN_ID=runpod_dense_u4k_12x608_lfqat \
NPROC_PER_NODE=1 \
scripts/runpod/train_dense_u4k_12x608_lfqat.sh
```

Important:

- The best local `12x608` result was a continuation, not a pure-from-scratch run.
- This launcher is still useful on Runpod because it gives you the correct architecture, tokenizer, data path, legal export policy, and LFQAT knobs in one place.
- If you have a warm-start checkpoint, pass it through `INIT_MODEL_PATH=/abs/path/to/checkpoint.pt`.
- By default it targets the official raw-doc rebuild at `data/official_u4k_unigram/datasets/fineweb10B_spu4096_docs`.

## 5. Scale the same launcher up

For a larger pod, keep the same script and only change the process count and wallclock:

```bash
RUN_ID=runpod_dense_u4k_12x608_lfqat_7xh100 \
NPROC_PER_NODE=7 \
MAX_WALLCLOCK_SECONDS=686 \
scripts/runpod/train_dense_u4k_12x608_lfqat.sh
```

That `686s` target is the compute-matched approximation for `7xH100` vs the official `8xH100 * 600s` budget.
