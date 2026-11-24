#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "请先运行 ./scripts/install_linux.sh"
  exit 1
fi

if [[ "${1:-}" == "--teardown" ]]; then
  exec "$ROOT/.venv/bin/python" -m rvc_realtime setup-virtual-audio --teardown
fi
exec "$ROOT/.venv/bin/python" -m rvc_realtime setup-virtual-audio "$@"
