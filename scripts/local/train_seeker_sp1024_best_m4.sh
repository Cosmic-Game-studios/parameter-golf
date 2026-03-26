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

export RUN_ID="${RUN_ID:-local_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_compiled_longtail_projhi_attnhi_fp16_m4}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export NUM_LAYERS="${NUM_LAYERS:-10}"
export NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-10}"
export MODEL_DIM="${MODEL_DIM:-560}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export INIT_MODEL_PATH="${INIT_MODEL_PATH:-}"

# M4-safe local proxy geometry. This is for fast ranking, not official fidelity.
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-896}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-16384}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export MLX_MAX_MICROBATCH_TOKENS="${MLX_MAX_MICROBATCH_TOKENS:-8192}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32768}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-262144}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.88}"

export QK_GAIN_INIT="${QK_GAIN_INIT:-1.45}"
export ROPE_BASE="${ROPE_BASE:-10000.0}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-28.8}"

export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export WARMUP_STEPS="${WARMUP_STEPS:-24}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-2250}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-24}"
export MIN_LR_SCALE="${MIN_LR_SCALE:-0.14}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.00195}"
export MATRIX_LR="${MATRIX_LR:-0.0013}"
export SCALAR_LR="${SCALAR_LR:-0.0013}"
export TIED_EMBED_INIT_STD="${TIED_EMBED_INIT_STD:-0.0052}"

export MUON_MOMENTUM="${MUON_MOMENTUM:-0.97}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-6}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.9}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-900}"
export BETA1="${BETA1:-0.9}"
export BETA2="${BETA2:-0.96}"
export ADAM_EPS="${ADAM_EPS:-1e-08}"

export ITERATIONS="${ITERATIONS:-300}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}"

# Pure compile-friendly training path. Shipping policy remains post-hoc.
export LFQAT_START_STEP="${LFQAT_START_STEP:-0}"
export LFQAT_FULL_STEP="${LFQAT_FULL_STEP:-0}"
export LFQAT_MIN_PROB="${LFQAT_MIN_PROB:-1.0}"
export LFQAT_MAX_PROB="${LFQAT_MAX_PROB:-1.0}"
export LFQAT_KL_WEIGHT="${LFQAT_KL_WEIGHT:-0.0}"
export LFQAT_FISHER_WEIGHT="${LFQAT_FISHER_WEIGHT:-0.0}"
export LFQAT_TEMPERATURE="${LFQAT_TEMPERATURE:-2.0}"
export TRAIN_COMPRESSION_AWARE_WEIGHT="${TRAIN_COMPRESSION_AWARE_WEIGHT:-0.0}"
export TRAIN_GRAD_ONLY_NAME_PATTERNS="${TRAIN_GRAD_ONLY_NAME_PATTERNS:-}"
export TRAIN_GRAD_SKIP_NAME_PATTERNS="${TRAIN_GRAD_SKIP_NAME_PATTERNS:-}"

# Best measured fixed-settings local export policy:
# int8 everywhere, fp16 only on upper mlp.proj + attn.proj.
export QUANT_FORMAT="${QUANT_FORMAT:-int8_clean_per_row_v1}"
export INT4_BLOCK_SIZE="${INT4_BLOCK_SIZE:-64}"
export INT4_CLIP_PERCENTILE="${INT4_CLIP_PERCENTILE:-99.9}"
export TRAIN_QAT_BLOCK_SIZE="${TRAIN_QAT_BLOCK_SIZE:-64}"
export TARGET_EXPORT_NAME_PATTERNS="${TARGET_EXPORT_NAME_PATTERNS:-}"
export INT4_NAME_PATTERNS="${INT4_NAME_PATTERNS:-}"
export TRAIN_QAT_NAME_PATTERNS="${TRAIN_QAT_NAME_PATTERNS:-}"
export TRAIN_COMPRESSION_AWARE_NAME_PATTERNS="${TRAIN_COMPRESSION_AWARE_NAME_PATTERNS:-}"
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS="${INT8_KEEP_FLOAT_FP16_NAME_PATTERNS:-blocks.5.attn.proj.weight,blocks.5.mlp.proj.weight,blocks.6.attn.proj.weight,blocks.6.mlp.proj.weight,blocks.7.attn.proj.weight,blocks.7.mlp.proj.weight,blocks.8.attn.proj.weight,blocks.8.mlp.proj.weight,blocks.9.attn.proj.weight,blocks.9.mlp.proj.weight}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing dataset at $DATA_PATH" >&2
  echo "Fetch at least one local shard first:" >&2
  echo "  $PYTHON_BIN data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1" >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "Missing tokenizer at $TOKENIZER_PATH" >&2
  echo "Fetch it with:" >&2
  echo "  $PYTHON_BIN data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1" >&2
  exit 1
fi

exec "$PYTHON_BIN" train_gpt_mlx.py
