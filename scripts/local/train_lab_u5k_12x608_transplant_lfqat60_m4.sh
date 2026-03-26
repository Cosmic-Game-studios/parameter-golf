#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

MIN_FREE_MB="${MIN_FREE_MB:-1024}"
AVAILABLE_MB="$(df -Pm . | awk 'NR==2 {print $4}')"
if [[ -n "${AVAILABLE_MB}" && "${AVAILABLE_MB}" -lt "${MIN_FREE_MB}" ]]; then
  echo "error:not_enough_disk_space avail_mb:${AVAILABLE_MB} required_mb:${MIN_FREE_MB} cwd:${ROOT}" >&2
  exit 1
fi

export RUN_ID="${RUN_ID:-lab_u5k_12x608_kv2_transplant_lfqat60_fcproj}"
export DATA_PATH="${DATA_PATH:-./data/local_u5k_unigram/datasets/fineweb10B_spu5120_local}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/local_u5k_unigram/tokenizers/fineweb_5120_unigram.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-5120}"

export NUM_LAYERS="${NUM_LAYERS:-12}"
export NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-12}"
export MODEL_DIM="${MODEL_DIM:-608}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export INIT_MODEL_PATH="${INIT_MODEL_PATH:-./logs/lab_u5k_12x608_kv2_transplant_init_mlx_model.npz}"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export MLX_MAX_MICROBATCH_TOKENS="${MLX_MAX_MICROBATCH_TOKENS:-4096}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-262144}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.9}"

export QK_GAIN_INIT="${QK_GAIN_INIT:-1.45}"
export ROPE_BASE="${ROPE_BASE:-10000.0}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30.0}"

export LR_SCHEDULE="${LR_SCHEDULE:-constant}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-0}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-0}"
export MIN_LR_SCALE="${MIN_LR_SCALE:-1.0}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.0025}"
export MATRIX_LR="${MATRIX_LR:-0.0018}"
export SCALAR_LR="${SCALAR_LR:-0.0018}"
export TIED_EMBED_INIT_STD="${TIED_EMBED_INIT_STD:-0.0050}"

export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-0}"
export BETA1="${BETA1:-0.9}"
export BETA2="${BETA2:-0.96}"
export ADAM_EPS="${ADAM_EPS:-1e-08}"

export ITERATIONS="${ITERATIONS:-60}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-60}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"

export LFQAT_START_STEP="${LFQAT_START_STEP:-0}"
export LFQAT_FULL_STEP="${LFQAT_FULL_STEP:-20}"
export LFQAT_MIN_PROB="${LFQAT_MIN_PROB:-0.15}"
export LFQAT_MAX_PROB="${LFQAT_MAX_PROB:-1.0}"
export LFQAT_KL_WEIGHT="${LFQAT_KL_WEIGHT:-0.03}"
export LFQAT_FISHER_WEIGHT="${LFQAT_FISHER_WEIGHT:-0.008}"
export LFQAT_TEMPERATURE="${LFQAT_TEMPERATURE:-2.0}"
export TRAIN_COMPRESSION_AWARE_WEIGHT="${TRAIN_COMPRESSION_AWARE_WEIGHT:-0.0015}"
export TRAIN_INT8_AWARE_WEIGHT="${TRAIN_INT8_AWARE_WEIGHT:-0.0}"
export TRAIN_GRAD_ONLY_NAME_PATTERNS="${TRAIN_GRAD_ONLY_NAME_PATTERNS:-}"
export TRAIN_GRAD_SKIP_NAME_PATTERNS="${TRAIN_GRAD_SKIP_NAME_PATTERNS:-}"

export QUANT_FORMAT="${QUANT_FORMAT:-mixed_int4_int8_packed_v2}"
export INT4_BLOCK_SIZE="${INT4_BLOCK_SIZE:-64}"
export INT4_CLIP_PERCENTILE="${INT4_CLIP_PERCENTILE:-99.9}"
export TRAIN_QAT_BLOCK_SIZE="${TRAIN_QAT_BLOCK_SIZE:-64}"
export TARGET_EXPORT_NAME_PATTERNS="${TARGET_EXPORT_NAME_PATTERNS:-blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight}"
export INT4_NAME_PATTERNS="${INT4_NAME_PATTERNS:-$TARGET_EXPORT_NAME_PATTERNS}"
export TRAIN_QAT_NAME_PATTERNS="${TRAIN_QAT_NAME_PATTERNS:-$TARGET_EXPORT_NAME_PATTERNS}"
export TRAIN_COMPRESSION_AWARE_NAME_PATTERNS="${TRAIN_COMPRESSION_AWARE_NAME_PATTERNS:-$TARGET_EXPORT_NAME_PATTERNS}"
export TRAIN_INT8_AWARE_NAME_PATTERNS="${TRAIN_INT8_AWARE_NAME_PATTERNS:-}"
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS="${INT8_KEEP_FLOAT_FP16_NAME_PATTERNS:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing dataset at $DATA_PATH" >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "Missing tokenizer at $TOKENIZER_PATH" >&2
  exit 1
fi

if [[ ! -f "$INIT_MODEL_PATH" ]]; then
  echo "Missing init model at $INIT_MODEL_PATH" >&2
  exit 1
fi

exec "$PYTHON_BIN" train_gpt_mlx.py
