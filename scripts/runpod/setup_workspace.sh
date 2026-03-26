#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

requirements_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum requirements.txt | awk '{print $1}'
  else
    shasum -a 256 requirements.txt | awk '{print $1}'
  fi
}

venv_is_ready() {
  [[ -x "$PYTHON_BIN" ]] || return 1
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sentencepiece  # noqa: F401
import torch  # noqa: F401
PY
}

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

PYTHON_BIN="$ROOT/.venv/bin/python"
PIP_BIN="$ROOT/.venv/bin/pip"
STAMP_FILE="$ROOT/.venv/.requirements.sha256"
CURRENT_HASH="$(requirements_hash)"
STAMP_HASH=""

if [[ -f "$STAMP_FILE" ]]; then
  STAMP_HASH="$(tr -d '[:space:]' < "$STAMP_FILE")"
fi

if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
  echo "Skipping pip install because SKIP_PIP_INSTALL=1"
elif [[ "${FORCE_PIP_INSTALL:-0}" != "1" && "$STAMP_HASH" == "$CURRENT_HASH" ]] && venv_is_ready; then
  echo "Python environment already matches requirements.txt; skipping pip install"
else
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PIP_BIN" install -r requirements.txt
  printf '%s\n' "$CURRENT_HASH" > "$STAMP_FILE"
fi

nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
import sentencepiece
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count()})
PY
