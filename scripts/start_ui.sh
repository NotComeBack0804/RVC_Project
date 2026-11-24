#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-rvc/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "请先运行 ./scripts/install_linux.sh --with-rvc-cpu"
  exit 1
fi
exec "$PY" -m rvc_realtime serve --host 127.0.0.1 --port "${PORT:-7870}"
