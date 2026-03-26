#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  TORCHRUN_BIN="torchrun"
fi

DATA_PATH_DEFAULT="./data/official_u4k_unigram/datasets/fineweb10B_spu4096_docs"
TOKENIZER_PATH_DEFAULT="./data/official_u4k_unigram/tokenizers/fineweb_4096_unigram.model"
DOCS_MANIFEST_DEFAULT="./data/official_u4k_unigram/docs_selected.source_manifest.json"
BASE_CHECKPOINT_DEFAULT="./checkpoints/runpod_dense_u4k_12x608_lfqat_1xh100_20260319_rerun1_final_model.pt"
FC_PROJ_PATTERNS="blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight"

export RUN_ID="${RUN_ID:-runpod_dense_u4k_12x608_lfqat_continue80m}"
export DATA_PATH="${DATA_PATH:-$DATA_PATH_DEFAULT}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$TOKENIZER_PATH_DEFAULT}"
export DOCS_MANIFEST="${DOCS_MANIFEST:-$DOCS_MANIFEST_DEFAULT}"
export INIT_MODEL_PATH="${INIT_MODEL_PATH-$BASE_CHECKPOINT_DEFAULT}"
export VOCAB_SIZE="${VOCAB_SIZE:-4096}"
export NUM_LAYERS="${NUM_LAYERS:-12}"
export NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-12}"
export MODEL_DIM="${MODEL_DIM:-608}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-20}"
export MIN_LR_SCALE="${MIN_LR_SCALE:-0.2}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.002}"
export MATRIX_LR="${MATRIX_LR:-0.0015}"
export SCALAR_LR="${SCALAR_LR:-0.0015}"
export ITERATIONS="${ITERATIONS:-2500}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-4800}"
export QUANT_FORMAT="${QUANT_FORMAT:-mixed_int4_int8_packed_v2}"
export INT4_BLOCK_SIZE="${INT4_BLOCK_SIZE:-64}"
export TARGET_EXPORT_NAME_PATTERNS="${TARGET_EXPORT_NAME_PATTERNS:-$FC_PROJ_PATTERNS}"
export INT4_NAME_PATTERNS="${INT4_NAME_PATTERNS:-$TARGET_EXPORT_NAME_PATTERNS}"
export TRAIN_QAT_NAME_PATTERNS="${TRAIN_QAT_NAME_PATTERNS:-$TARGET_EXPORT_NAME_PATTERNS}"
export TRAIN_QAT_BLOCK_SIZE="${TRAIN_QAT_BLOCK_SIZE:-64}"
export TRAIN_COMPRESSION_AWARE_WEIGHT="${TRAIN_COMPRESSION_AWARE_WEIGHT:-0.0015}"
export TRAIN_COMPRESSION_AWARE_NAME_PATTERNS="${TRAIN_COMPRESSION_AWARE_NAME_PATTERNS:-$TARGET_EXPORT_NAME_PATTERNS}"
export TRAIN_COMPRESSION_AWARE_BLOCK_SIZE="${TRAIN_COMPRESSION_AWARE_BLOCK_SIZE:-64}"
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS="${INT8_KEEP_FLOAT_FP16_NAME_PATTERNS:-}"
export LFQAT_KL_WEIGHT="${LFQAT_KL_WEIGHT:-0.05}"
export LFQAT_FISHER_WEIGHT="${LFQAT_FISHER_WEIGHT:-0.01}"
export LFQAT_TEMPERATURE="${LFQAT_TEMPERATURE:-2.0}"
export LFQAT_FISHER_DECAY="${LFQAT_FISHER_DECAY:-0.95}"
export LFQAT_START_STEP="${LFQAT_START_STEP:-0}"
export LFQAT_FULL_STEP="${LFQAT_FULL_STEP:-0}"
export LFQAT_MIN_PROB="${LFQAT_MIN_PROB:-1.0}"
export LFQAT_MAX_PROB="${LFQAT_MAX_PROB:-1.0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing dataset at $DATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "Missing tokenizer at $TOKENIZER_PATH" >&2
  exit 1
fi
if [[ ! -f "$DOCS_MANIFEST" ]]; then
  echo "Missing official docs manifest at $DOCS_MANIFEST" >&2
  exit 1
fi
if [[ -n "$INIT_MODEL_PATH" && ! -f "$INIT_MODEL_PATH" ]]; then
  echo "Missing INIT_MODEL_PATH checkpoint at $INIT_MODEL_PATH" >&2
  exit 1
fi
if [[ "$DATA_PATH" == *"local_u4k_unigram"* || "$TOKENIZER_PATH" == *"local_u4k_unigram"* ]]; then
  echo "Refusing local proxy U4K path. Use the official raw-doc rebuild under data/official_u4k_unigram." >&2
  exit 1
fi

exec "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py
