#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征提取入口骨架：检查数据与环境，不在条件不足时开始提取。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rvc_realtime.paths import official_requirements_cpu, testdata_dir, third_party_rvc
from rvc_realtime.processor import probe_runtime


def main() -> int:
    prepared = testdata_dir() / "prepared"
    clips = list(prepared.glob("*.wav")) if prepared.is_dir() else []
    runtime = probe_runtime()
    print("这是训练流程中的特征提取骨架，不会在数据或环境不完整时擅自开始提取。")
    print(f"已整理的片段：{len(clips)} 个（目录 {prepared}）")
    print(runtime.message)
    if not clips:
        print("请先把授权录音放到 testdata/authorized_reference/，并运行：")
        print("  python tools/train/prepare_dataset.py --i-have-authorization")
        return 2
    if not runtime.rvc_ok:
        print("完整 RVC CPU 环境尚未就绪。请先：")
        print("  ./scripts/install_linux.sh --with-rvc-cpu")
        print("  ./scripts/download_base_models.sh --with-pretrained")
        return 3
    rvc = third_party_rvc()
    print("环境看起来可用。请在官方 RVC WebUI 中完成音高与特征提取，或在其仓库根目录按其当前文档操作：")
    print(f"  {rvc}")
    print(f"官方 CPU 依赖文件：{official_requirements_cpu()}")
    print("不要使用未经授权的参考音频。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
