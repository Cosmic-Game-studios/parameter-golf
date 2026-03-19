#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  TORCHRUN_BIN="torchrun"
fi

export RUN_ID="${RUN_ID:-runpod_baseline_sp1024}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing dataset at $DATA_PATH" >&2
  exit 1
fi
if [[ "$DATA_PATH" == *"local_u4k_unigram"* ]]; then
  echo "Refusing local proxy dataset path: $DATA_PATH" >&2
  exit 1
fi
if [[ "$TOKENIZER_PATH" == *"local_u4k_unigram"* ]]; then
  echo "Refusing local proxy tokenizer path: $TOKENIZER_PATH" >&2
  exit 1
fi

exec "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py
