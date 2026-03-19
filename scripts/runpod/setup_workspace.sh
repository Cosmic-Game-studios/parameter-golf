#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

PYTHON_BIN="$ROOT/.venv/bin/python"
PIP_BIN="$ROOT/.venv/bin/pip"

"$PYTHON_BIN" -m pip install --upgrade pip
if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  "$PIP_BIN" install -r requirements.txt
fi

nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
import sentencepiece
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count()})
PY
