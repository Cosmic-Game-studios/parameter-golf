#!/usr/bin/env bash
set -euo pipefail
# Autoresearch Experiment 003: Bigger effective batch on crossskip40 donor
# Hypothesis: 2x batch (grad_accum=2) gives better gradient signal for continuation
# Anchor: crossskip40_fc_top4 shipped 1.98453131

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

MIN_FREE_MB="${MIN_FREE_MB:-2048}"
AVAILABLE_MB="$(df -Pm . | awk 'NR==2 {print $4}')"
if [[ -n "${AVAILABLE_MB}" && "${AVAILABLE_MB}" -lt "${MIN_FREE_MB}" ]]; then
  echo "error:not_enough_disk_space avail_mb:${AVAILABLE_MB} required_mb:${MIN_FREE_MB} cwd:${ROOT}" >&2
  exit 1
fi

export DATA_PATH="./data/local_u4k_unigram/datasets/fineweb10B_spu4096_local"
export TOKENIZER_PATH="./data/local_u4k_unigram/tokenizers/fineweb_4096_unigram.model"
export VOCAB_SIZE="4096"

export NUM_LAYERS="12"
export NUM_UNIQUE_LAYERS="10"
export MODEL_DIM="608"
export NUM_HEADS="8"
export NUM_KV_HEADS="2"
export MLP_MULT="2"
export TIE_EMBEDDINGS="1"

export INIT_MODEL_PATH="./logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_mlx_model.npz"
export RUN_ID="autoresearch_003_bigbatch80_crossskip40_m4"

export TRAIN_SEQ_LEN="1024"
# 2x effective batch via grad accumulation
export TRAIN_BATCH_TOKENS="16384"
export GRAD_ACCUM_STEPS="2"
export MLX_MAX_MICROBATCH_TOKENS="4096"
export VAL_BATCH_SIZE="131072"
export VAL_MAX_TOKENS="262144"
export GRAD_CLIP_NORM="0.9"

export QK_GAIN_INIT="1.45"
export ROPE_BASE="10000.0"
export LOGIT_SOFTCAP="30.0"

# Match donor LR (from real80m launcher)
export LR_SCHEDULE="cosine"
export WARMUP_STEPS="0"
export WARMDOWN_ITERS="1200"
export LR_WARMUP_ITERS="0"
export MIN_LR_SCALE="0.18"
export TIED_EMBED_LR="0.0008"
export MATRIX_LR="0.0007"
export SCALAR_LR="0.0007"
export TIED_EMBED_INIT_STD="0.0050"

export MUON_MOMENTUM="0.95"
export MUON_BACKEND_STEPS="5"
export MUON_MOMENTUM_WARMUP_START="0.85"
export MUON_MOMENTUM_WARMUP_STEPS="0"
export BETA1="0.9"
export BETA2="0.96"
export ADAM_EPS="1e-08"

export ITERATIONS="80"
export MAX_WALLCLOCK_SECONDS="4800"
export VAL_LOSS_EVERY="20"
export TRAIN_LOG_EVERY="10"

# No LFQAT
export LFQAT_START_STEP="0"
export LFQAT_FULL_STEP="0"
export LFQAT_MIN_PROB="1.0"
export LFQAT_MAX_PROB="1.0"
export LFQAT_KL_WEIGHT="0.0"
export LFQAT_FISHER_WEIGHT="0.0"
export LFQAT_TEMPERATURE="2.0"

export TRAIN_COMPRESSION_AWARE_WEIGHT="0.0"
export TRAIN_COMPRESSION_AWARE_BLOCK_SIZE="64"
export TRAIN_INT8_AWARE_WEIGHT="0.0"
export TRAIN_QER_AWARE_WEIGHT="0.0"
export TRAIN_QER_AWARE_NAME_PATTERNS=""
export TRAIN_QER_AWARE_RANK="0"
export TRAIN_GRAD_ONLY_NAME_PATTERNS=""
export TRAIN_GRAD_SKIP_NAME_PATTERNS=""

export QUANT_FORMAT="mixed_int4_int8_packed_v2"
export INT4_BLOCK_SIZE="64"
export INT4_CLIP_PERCENTILE="99.9"
export TRAIN_QAT_BLOCK_SIZE="64"
export TARGET_EXPORT_NAME_PATTERNS=""
export INT4_NAME_PATTERNS="blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight"
export TRAIN_QAT_NAME_PATTERNS=""
export TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=""
export TRAIN_INT8_AWARE_NAME_PATTERNS=""
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=""

export GLOBAL_BUS_ENABLED="1"
export GLOBAL_BUS_READ_LAYERS="top4"
export GLOBAL_BUS_WRITE_LAYERS="top4"
export GLOBAL_BUS_SUMMARY_MODE="mean_tail"
export GLOBAL_BUS_TAIL_WEIGHT="0.65"
export GLOBAL_BUS_WRITE_INIT="0.08"
export GLOBAL_BUS_DECAY_INIT="0.5"

export CROSS_SKIP_ROUTER_ENABLED="1"
export CROSS_SKIP_ROUTER_MATCH_INIT="4.0"
export CROSS_SKIP_ROUTER_OTHER_INIT="-4.0"
export ALLOW_INIT_MISSING_KEYS="0"

if [[ ! -d "$DATA_PATH" ]]; then echo "Missing dataset at $DATA_PATH" >&2; exit 1; fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then echo "Missing tokenizer at $TOKENIZER_PATH" >&2; exit 1; fi
if [[ ! -f "$INIT_MODEL_PATH" ]]; then echo "Missing init model at $INIT_MODEL_PATH" >&2; exit 1; fi

exec "$PYTHON_BIN" train_gpt_mlx.py
