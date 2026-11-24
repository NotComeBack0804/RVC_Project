from __future__ import annotations

import platform
import sys
from pathlib import Path

from rvc_realtime.paths import (
    hubert_status,
    indices_dir,
    list_user_indices,
    list_user_models,
    official_requirements_cpu,
    project_root,
    rmvpe_status,
    third_party_rvc,
    weights_dir,
)
from rvc_realtime.processor import probe_runtime


def gather_status() -> str:
    root = project_root()
    runtime = probe_runtime()
    models = list_user_models()
    indices = list_user_indices()
    lines = [
        "RVC 实时语音转换 — 环境状态",
        f"项目目录：{root}",
        f"Python：{sys.version.split()[0]} ({platform.system()} {platform.machine()})",
        f"可执行文件：{sys.executable}",
    ]
    try:
        import numpy

        lines.append(f"numpy：{numpy.__version__}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"numpy：未安装（{exc}）")
    try:
        import soundfile

        lines.append(f"soundfile：{soundfile.__version__}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"soundfile：未安装（{exc}）")

    lines.append(f"torch：{runtime.torch_version or '未安装'}  cuda={runtime.torch_cuda}")
    lines.append(f"官方 RVC 源码：{'是' if (third_party_rvc() / 'infer' / 'rtrvc.py').is_file() else '否'}  ({runtime.rvc_path})")
    req = official_requirements_cpu()
    lines.append(f"官方 CPU 依赖文件：{req if req.is_file() else '未找到（文件名以仓库为准，当前为 requirments_cpu_py312.txt）'}")
    lines.append(runtime.message)

    lines.append("基础模型（公开，仅用于特征与音高，不是他人音色）：")
    for name, ok in hubert_status().items():
        lines.append(f"  HuBERT {name}：{'就绪' if ok else '缺失'}")
    for name, ok in rmvpe_status().items():
        lines.append(f"  RMVPE {name}：{'就绪' if ok else '缺失（onnx 仅 Windows DirectML 需要）' if name.endswith('.onnx') else '缺失'}")

    lines.append(f"用户音色模型目录：{weights_dir()}")
    if models:
        for p in models:
            lines.append(f"  - {p.name}  ({p.stat().st_size} bytes)")
    else:
        lines.append("  （空，变声模式会明确报错并退出，不会崩溃）")
    lines.append(f"索引目录：{indices_dir()}")
    if indices:
        for p in indices:
            lines.append(f"  - {p.name}")
    else:
        lines.append("  （空，可选）")
    return "\n".join(lines)
