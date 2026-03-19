#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

SOURCE_VARIANT="${SOURCE_VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data/official_u4k_unigram}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-./data/tokenizer_specs_u4k_unigram.json}"
DATASET_DIR="$OUTPUT_ROOT/datasets/fineweb10B_spu4096_docs"
DOCS_MANIFEST="$OUTPUT_ROOT/docs_selected.source_manifest.json"

if [[ "${FORCE_REBUILD:-0}" != "1" && -d "$DATASET_DIR" ]]; then
  echo "Dataset already exists at $DATASET_DIR"
  exit 0
fi

"$PYTHON_BIN" data/cached_challenge_fineweb.py --variant "$SOURCE_VARIANT" --with-docs --train-shards "$TRAIN_SHARDS"
"$PYTHON_BIN" data/download_hf_docs_and_tokenize.py --output-root "$OUTPUT_ROOT" --tokenizer-config "$TOKENIZER_CONFIG"

if [[ ! -f "$DOCS_MANIFEST" ]]; then
  echo "Missing docs manifest at $DOCS_MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Missing rebuilt dataset at $DATASET_DIR" >&2
  exit 1
fi

echo "Built official U4K docs export at $DATASET_DIR"
