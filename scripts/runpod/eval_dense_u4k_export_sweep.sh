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
BASE_CHECKPOINT_DEFAULT="./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_final_model.pt"
FC_HI="blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight"
PROJ_HI="blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight"
ATTN_HI="blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight"
FC_PROJ_HI="${FC_HI},${PROJ_HI}"

export DATA_PATH="${DATA_PATH:-$DATA_PATH_DEFAULT}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$TOKENIZER_PATH_DEFAULT}"
export DOCS_MANIFEST="${DOCS_MANIFEST:-$DOCS_MANIFEST_DEFAULT}"
export INIT_MODEL_PATH="${INIT_MODEL_PATH:-$BASE_CHECKPOINT_DEFAULT}"
export RUN_PREFIX="${RUN_PREFIX:-$(basename "${INIT_MODEL_PATH%.pt}")}"
export VOCAB_SIZE="${VOCAB_SIZE:-4096}"
export NUM_LAYERS="${NUM_LAYERS:-12}"
export NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-12}"
export MODEL_DIM="${MODEL_DIM:-608}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-60}"
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
if [[ ! -f "$INIT_MODEL_PATH" ]]; then
  echo "Missing INIT_MODEL_PATH checkpoint at $INIT_MODEL_PATH" >&2
  exit 1
fi

mkdir -p logs
SUMMARY="logs/${RUN_PREFIX}_export_sweep_summary.md"
printf '| Candidate | Compressed model bytes | Total counted bytes | Full-val loss / bpb |\n| --- | --- | --- | --- |\n' > "$SUMMARY"

run_eval() {
  local name="$1"
  local quant_format="$2"
  local int4_patterns="$3"
  local fp16_patterns="$4"
  local run_id="${RUN_PREFIX}_${name}"
  local log_file="logs/${run_id}.txt"

  rm -f final_model.pt final_model.int8.ptz "$log_file"

  env \
    RUN_ID="$run_id" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE="$VOCAB_SIZE" \
    NUM_LAYERS="$NUM_LAYERS" \
    NUM_UNIQUE_LAYERS="$NUM_UNIQUE_LAYERS" \
    MODEL_DIM="$MODEL_DIM" \
    NUM_HEADS="$NUM_HEADS" \
    NUM_KV_HEADS="$NUM_KV_HEADS" \
    MLP_MULT="$MLP_MULT" \
    TIE_EMBEDDINGS="$TIE_EMBEDDINGS" \
    TRAIN_BATCH_TOKENS="$TRAIN_BATCH_TOKENS" \
    VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
    ITERATIONS=0 \
    VAL_LOSS_EVERY=1 \
    TRAIN_LOG_EVERY=1 \
    MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
    WARMUP_STEPS=0 \
    INIT_MODEL_PATH="$INIT_MODEL_PATH" \
    LR_SCHEDULE=constant \
    TIED_EMBED_LR=0 \
    MATRIX_LR=0 \
    SCALAR_LR=0 \
    QUANT_FORMAT="$quant_format" \
    INT4_NAME_PATTERNS="$int4_patterns" \
    INT8_KEEP_FLOAT_FP16_NAME_PATTERNS="$fp16_patterns" \
    TRAIN_QAT_NAME_PATTERNS="__never__" \
    TRAIN_COMPRESSION_AWARE_WEIGHT=0 \
    TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
    LFQAT_KL_WEIGHT=0 \
    LFQAT_FISHER_WEIGHT=0 \
    LFQAT_START_STEP=0 \
    LFQAT_FULL_STEP=0 \
    LFQAT_MIN_PROB=0 \
    LFQAT_MAX_PROB=0 \
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py > "$log_file" 2>&1

  local compressed total final_line
  compressed="$(grep 'Serialized model int8+zlib:' "$log_file" | tail -n1 | awk '{print $4}')"
  total="$(grep 'Total submission size int8+zlib:' "$log_file" | tail -n1 | awk '{print $5}')"
  final_line="$(grep 'final_int8_zlib_roundtrip_exact' "$log_file" | tail -n1 | sed 's/^.*final_int8_zlib_roundtrip_exact //')"
  printf '| `%s` | `%s` | `%s` | `%s` |\n' "$name" "$compressed" "$total" "$final_line" >> "$SUMMARY"
  printf '%s\n' "--- ${run_id} ---"
  grep -E 'Serialized model int8\+zlib|Total submission size int8\+zlib|final_int8_zlib_roundtrip_exact' "$log_file"
  printf '\n'
}

run_eval "current_fcproj" "mixed_int4_int8_packed_v2" "$FC_PROJ_HI" ""
run_eval "fchi_only" "mixed_int4_int8_packed_v2" "$FC_HI" ""
run_eval "int8_all" "int8_clean_per_row_v1" "" ""
run_eval "int8_tok" "int8_clean_per_row_v1" "" "tok_emb.weight"
run_eval "int8_attnhi_fp16" "int8_clean_per_row_v1" "" "$ATTN_HI"
run_eval "int8_projhi_fp16" "int8_clean_per_row_v1" "" "$PROJ_HI"
run_eval "int8_attnhi_tok_fp16" "int8_clean_per_row_v1" "" "tok_emb.weight,$ATTN_HI"
run_eval "fchi_tok_fp16" "mixed_int4_int8_packed_v2" "$FC_HI" "tok_emb.weight"
run_eval "fcproj_tok_fp16" "mixed_int4_int8_packed_v2" "$FC_PROJ_HI" "tok_emb.weight"
run_eval "fcproj_attnhi_fp16" "mixed_int4_int8_packed_v2" "$FC_PROJ_HI" "$ATTN_HI"
run_eval "fcproj_tok_attnhi_fp16" "mixed_int4_int8_packed_v2" "$FC_PROJ_HI" "tok_emb.weight,$ATTN_HI"

printf 'Summary written to %s\n' "$SUMMARY"
