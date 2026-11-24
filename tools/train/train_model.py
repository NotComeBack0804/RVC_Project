#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练入口骨架：检查条件，不在用户未提供授权数据时开始训练。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rvc_realtime.paths import testdata_dir, third_party_rvc, weights_dir
from rvc_realtime.processor import probe_runtime


def main() -> int:
    prepared = testdata_dir() / "prepared"
    clips = list(prepared.glob("*.wav")) if prepared.is_dir() else []
    runtime = probe_runtime()
    print("这是训练入口骨架。在提供已授权参考录音并完成特征提取之前，不会开始训练。")
    print(f"已整理的片段：{len(clips)}")
    print(runtime.message)
    if not clips:
        print("没有可用的授权训练素材。请先准备录音并运行 prepare_dataset.py。")
        return 2
    if not runtime.rvc_ok:
        print("请先安装独立的 RVC CPU 环境：./scripts/install_linux.sh --with-rvc-cpu")
        return 3
    print("请使用官方 RVC WebUI 完成训练并导出模型：")
    print(f"  cd {third_party_rvc()}")
    print("  source ../../.venv-rvc/bin/activate   # Windows: ..\\..\\.venv-rvc\\Scripts\\Activate.ps1")
    print("  python webui.py")
    print(f"导出的 .pth 请复制到：{weights_dir()}")
    print("可选索引请放到 assets/indices/。")
    print("CPU 训练可能较慢。AMD RX 7800 XT 加速请在独立环境中稍后验证 DirectML，不要把这套 CPU 环境改成 CUDA。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
