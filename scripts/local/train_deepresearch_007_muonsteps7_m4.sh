#!/usr/bin/env bash
set -euo pipefail
# DeepResearch Exp 007: optimizer — Muon backend_steps=7 with best LR+batch
# Hypothesis: more Newton-Schulz iterations improve Muon at high LR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$ROOT/.venv/bin/python"
else PYTHON_BIN="python3"; fi

export DATA_PATH="./data/local_u4k_unigram/datasets/fineweb10B_spu4096_local"
export TOKENIZER_PATH="./data/local_u4k_unigram/tokenizers/fineweb_4096_unigram.model"
export VOCAB_SIZE="4096"
export NUM_LAYERS="12" NUM_UNIQUE_LAYERS="10" MODEL_DIM="608"
export NUM_HEADS="8" NUM_KV_HEADS="2" MLP_MULT="2" TIE_EMBEDDINGS="1"

export INIT_MODEL_PATH="./logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_mlx_model.npz"
export RUN_ID="${RUN_ID:-deepresearch_007_muonsteps7_m4}"

export TRAIN_SEQ_LEN="1024"
export TRAIN_BATCH_TOKENS="32768" GRAD_ACCUM_STEPS="4"
export MLX_MAX_MICROBATCH_TOKENS="4096"
export VAL_BATCH_SIZE="131072" VAL_MAX_TOKENS="262144"
export GRAD_CLIP_NORM="0.9"
export QK_GAIN_INIT="1.45" ROPE_BASE="10000.0" LOGIT_SOFTCAP="30.0"

# Best LR from DR-006
export LR_SCHEDULE="cosine"
export WARMUP_STEPS="0" WARMDOWN_ITERS="1200" LR_WARMUP_ITERS="0"
export MIN_LR_SCALE="0.18"
export TIED_EMBED_LR="0.0035"
export MATRIX_LR="0.0028"
export SCALAR_LR="0.0028"
export TIED_EMBED_INIT_STD="0.0050"

export MUON_MOMENTUM="0.95"
# THE CHANGE: more Newton-Schulz iterations
export MUON_BACKEND_STEPS="7"
export MUON_MOMENTUM_WARMUP_START="0.85" MUON_MOMENTUM_WARMUP_STEPS="0"
export BETA1="0.9" BETA2="0.96" ADAM_EPS="1e-08"

export ITERATIONS="80" MAX_WALLCLOCK_SECONDS="7200"
export VAL_LOSS_EVERY="20" TRAIN_LOG_EVERY="10"

export LFQAT_START_STEP="0" LFQAT_FULL_STEP="0"
export LFQAT_MIN_PROB="1.0" LFQAT_MAX_PROB="1.0"
export LFQAT_KL_WEIGHT="0.0" LFQAT_FISHER_WEIGHT="0.0" LFQAT_TEMPERATURE="2.0"
export TRAIN_COMPRESSION_AWARE_WEIGHT="0.0" TRAIN_COMPRESSION_AWARE_BLOCK_SIZE="64"
export TRAIN_INT8_AWARE_WEIGHT="0.0"
export TRAIN_QER_AWARE_WEIGHT="0.0" TRAIN_QER_AWARE_NAME_PATTERNS="" TRAIN_QER_AWARE_RANK="0"
export TRAIN_GRAD_ONLY_NAME_PATTERNS="" TRAIN_GRAD_SKIP_NAME_PATTERNS=""

export QUANT_FORMAT="mixed_int4_int8_packed_v2"
export INT4_BLOCK_SIZE="64" INT4_CLIP_PERCENTILE="99.9" TRAIN_QAT_BLOCK_SIZE="64"
export TARGET_EXPORT_NAME_PATTERNS=""
export INT4_NAME_PATTERNS="blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight"
export TRAIN_QAT_NAME_PATTERNS="" TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=""
export TRAIN_INT8_AWARE_NAME_PATTERNS="" INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=""

export GLOBAL_BUS_ENABLED="1" GLOBAL_BUS_READ_LAYERS="top4" GLOBAL_BUS_WRITE_LAYERS="top4"
export GLOBAL_BUS_SUMMARY_MODE="mean_tail" GLOBAL_BUS_TAIL_WEIGHT="0.65"
export GLOBAL_BUS_WRITE_INIT="0.08" GLOBAL_BUS_DECAY_INIT="0.5"
export CROSS_SKIP_ROUTER_ENABLED="1"
export CROSS_SKIP_ROUTER_MATCH_INIT="4.0" CROSS_SKIP_ROUTER_OTHER_INIT="-4.0"
export ALLOW_INIT_MISSING_KEYS="0"

if [[ ! -d "$DATA_PATH" ]]; then echo "Missing dataset" >&2; exit 1; fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then echo "Missing tokenizer" >&2; exit 1; fi
if [[ ! -f "$INIT_MODEL_PATH" ]]; then echo "Missing init model" >&2; exit 1; fi

exec "$PYTHON_BIN" train_gpt_mlx.py
