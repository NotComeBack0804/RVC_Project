#!/usr/bin/env bash
# 在 Linux 上创建 Python 3.12 虚拟环境并安装依赖。默认只装轻量直通环境。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_RVC_CPU=0
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
  cat <<USAGE
用法: ./scripts/install_linux.sh [--with-rvc-cpu]

  默认安装轻量环境到 .venv（直通、虚拟音频、自检、基础模型下载）。
  --with-rvc-cpu  另外创建 .venv-rvc，并按官方 requirments_cpu_py312.txt
                  安装 CPU 推理依赖。不要在该环境中混入 CUDA 安装命令。
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --with-rvc-cpu) WITH_RVC_CPU=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $arg"; usage; exit 2 ;;
  esac
done

find_python() {
  if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.12 >/dev/null
    if command -v python3.12 >/dev/null 2>&1; then
      command -v python3.12
      return
    fi
    # uv 安装的解释器可能尚未出现在 PATH 上
    UV_PY="$(uv python find 3.12 2>/dev/null || true)"
    if [[ -n "$UV_PY" && -x "$UV_PY" ]]; then
      echo "$UV_PY"
      return
    fi
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "警告: 未找到 Python 3.12，回退到 $(python3 -V 2>&1)。官方 RVC CPU 依赖面向 3.12；轻量直通可在 3.10+ 运行。" >&2
    command -v python3
    return
  fi
  echo "未找到可用的 Python。请安装 Python 3.12，或先安装 uv。" >&2
  exit 1
}

ensure_rvc_repo() {
  local dest="$ROOT/third_party/Retrieval-based-Voice-Conversion-WebUI"
  if [[ -f "$dest/requirments_cpu_py312.txt" ]]; then
    return
  fi
  mkdir -p "$ROOT/third_party"
  echo "正在浅克隆官方 RVC 仓库到 third_party/ ..."
  git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git "$dest"
}

ensure_venv() {
  local venv_dir="$1"
  if [[ -x "$venv_dir/bin/python" ]]; then
    echo "复用已有虚拟环境: $venv_dir"
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PY" "$venv_dir"
  else
    "$PY" -m venv "$venv_dir"
  fi
}

install_reqs() {
  local venv_dir="$1"
  local reqs="$2"
  # 官方 requirments_*.txt 同时声明了多个 index。uv 默认只采用「首次出现该包的
  # 源」，会把 tqdm 等包卡在 PyTorch CPU 索引的旧版本上；这里对多源依赖文件
  # 使用 unsafe-best-match，效果接近 pip 的检索方式，并不安装 CUDA。
  local uv_index_args=()
  if grep -qE '^--(extra-)?index-url ' "$reqs"; then
    uv_index_args+=(--index-strategy unsafe-best-match)
  fi
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$venv_dir" "${uv_index_args[@]}" -r "$reqs"
  else
    if ! "$venv_dir/bin/python" -m pip --version >/dev/null 2>&1; then
      "$PY" -m ensurepip --upgrade || true
    fi
    "$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel
    "$venv_dir/bin/python" -m pip install -r "$reqs"
  fi
}

PY="$(find_python)"
echo "使用解释器: $PY ($("$PY" -V 2>&1))"

ensure_venv "$ROOT/.venv"
install_reqs "$ROOT/.venv" "$ROOT/requirements-lite.txt"

echo "轻量环境已就绪: $ROOT/.venv"
echo "激活: source $ROOT/.venv/bin/activate"

ensure_rvc_repo

if [[ "$WITH_RVC_CPU" -eq 1 ]]; then
  REQ="$ROOT/third_party/Retrieval-based-Voice-Conversion-WebUI/requirments_cpu_py312.txt"
  if [[ ! -f "$REQ" ]]; then
    echo "未找到官方 CPU 依赖文件。当前仓库应存在 requirments_cpu_py312.txt（文件名以仓库为准）。" >&2
    exit 1
  fi
  echo "正在准备独立的 RVC CPU 环境 .venv-rvc（不安装 CUDA）..."
  ensure_venv "$ROOT/.venv-rvc"
  install_reqs "$ROOT/.venv-rvc" "$REQ"
  echo "RVC CPU 环境已就绪: $ROOT/.venv-rvc"
  "$ROOT/.venv-rvc/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("警告: 当前环境检测到 CUDA。CPU 起步方案不应混入 CUDA 安装。")
PY
else
  echo "未安装完整 RVC CPU 推理依赖。需要变声时再运行:"
  echo "  ./scripts/install_linux.sh --with-rvc-cpu"
fi

echo
echo "接下来建议："
echo "  source .venv/bin/activate"
echo "  python -m rvc_realtime setup-virtual-audio"
echo "  python -m rvc_realtime self-test"
echo "  ./scripts/download_base_models.sh"
