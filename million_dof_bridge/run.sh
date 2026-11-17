#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON:-python3}"

if ! "$PYTHON_BIN" -c 'import torch, numpy, matplotlib' >/dev/null 2>&1; then
  echo "Dependencies are incomplete; creating local environment at $ROOT/.venv"
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$ROOT/.venv"
  fi
  PYTHON_BIN="$ROOT/.venv/bin/python"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r "$ROOT/requirements.txt"
fi

exec "$PYTHON_BIN" "$ROOT/run.py" "$@"
