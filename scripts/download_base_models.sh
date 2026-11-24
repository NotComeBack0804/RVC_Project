#!/usr/bin/env bash
# 下载公开基础模型（HuBERT、RMVPE），不会下载或附带任何他人音色模型。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PY="$(command -v python3.12)"
  else
    PY="$(command -v python3)"
  fi
fi

exec "$PY" "$ROOT/scripts/download_base_models.py" "$@"
