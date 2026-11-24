# -*- coding: utf-8 -*-
"""便携运行时静默初始化：ffmpeg / 原生库 PATH，以及官方 assets 目录连接。"""
from __future__ import annotations

def _boot() -> None:
    try:
        import os
        import sys
        from pathlib import Path

        # runtime\\sitecustomize.py is on sys.path via python312._pth "."
        runtime = Path(__file__).resolve().parent
        if str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        root = str(runtime.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        os.chdir(root)
        from rvc_realtime.windows_bootstrap import configure_runtime_path, ensure_rvc_asset_links

        configure_runtime_path()
        ensure_rvc_asset_links()
    except Exception:
        # Never block the interpreter from starting.
        pass


_boot()
