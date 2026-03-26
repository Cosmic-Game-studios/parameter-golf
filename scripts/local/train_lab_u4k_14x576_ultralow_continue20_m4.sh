#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

MIN_FREE_MB="${MIN_FREE_MB:-512}"
AVAILABLE_MB="$(df -Pm . | awk 'NR==2 {print $4}')"
if [[ -n "${AVAILABLE_MB}" && "${AVAILABLE_MB}" -lt "${MIN_FREE_MB}" ]]; then
  echo "error:not_enough_disk_space avail_mb:${AVAILABLE_MB} required_mb:${MIN_FREE_MB} cwd:${ROOT}" >&2
  exit 1
fi

export RUN_ID="${RUN_ID:-lab_u4k_14x576_kv2_ultralow_continue20_m4}"

export DATA_PATH="${DATA_PATH:-./data/local_u4k_unigram/datasets/fineweb10B_spu4096_local}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/local_u4k_unigram/tokenizers/fineweb_4096_unigram.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-4096}"

export NUM_LAYERS="${NUM_LAYERS:-14}"
export NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-14}"
export MODEL_DIM="${MODEL_DIM:-576}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"

export INIT_MODEL_PATH="${INIT_MODEL_PATH:-./logs/dense_u4k_14x576_kv2_boot_from_12x576_adapt150_mlx_model.npz}"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export MLX_MAX_MICROBATCH_TOKENS="${MLX_MAX_MICROBATCH_TOKENS:-8192}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8192}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-262144}"

export ITERATIONS="${ITERATIONS:-20}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export LR_SCHEDULE="${LR_SCHEDULE:-constant}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-0}"
export MIN_LR_SCALE="${MIN_LR_SCALE:-1.0}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.0020}"
export MATRIX_LR="${MATRIX_LR:-0.0015}"
export SCALAR_LR="${SCALAR_LR:-0.0015}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-20}"

export QUANT_FORMAT="${QUANT_FORMAT:-mixed_int4_int8_packed_v2}"
export INT4_NAME_PATTERNS="${INT4_NAME_PATTERNS:-blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.12.mlp.fc.weight,blocks.13.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight,blocks.12.mlp.proj.weight,blocks.13.mlp.proj.weight}"
export INT4_BLOCK_SIZE="${INT4_BLOCK_SIZE:-64}"

export TRAIN_COMPRESSION_AWARE_WEIGHT="${TRAIN_COMPRESSION_AWARE_WEIGHT:-0.0}"
export TRAIN_COMPRESSION_AWARE_NAME_PATTERNS="${TRAIN_COMPRESSION_AWARE_NAME_PATTERNS:-}"
export TRAIN_QAT_NAME_PATTERNS="${TRAIN_QAT_NAME_PATTERNS:-}"
export LFQAT_KL_WEIGHT="${LFQAT_KL_WEIGHT:-0.0}"
export LFQAT_FISHER_WEIGHT="${LFQAT_FISHER_WEIGHT:-0.0}"
export LFQAT_MIN_PROB="${LFQAT_MIN_PROB:-1.0}"
export LFQAT_MAX_PROB="${LFQAT_MAX_PROB:-1.0}"

python train_gpt_mlx.py
