"""在官方仓库根目录以模块方式执行脚本。

必须用 ``python -m train.train`` 这类方式启动，而不能 ``exec`` 成 ``__main__`` 源码：
官方 ``train.py`` 会 ``multiprocessing`` spawn，``run`` 需要能按模块名被 pickle。
同时避免 ``python train/preprocess.py`` 时 ``sys.path[0]`` 变成 ``train/``，
把 ``import train`` 解析成正在运行的 ``train.py``。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from rvc_realtime.paths import third_party_rvc


def script_to_module(rel: str) -> str:
    rel = rel.replace("\\", "/").lstrip("./")
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法: python -m rvc_realtime.run_official train/preprocess.py ...", file=sys.stderr)
        return 2
    rel = argv[0]
    rvc = third_party_rvc().resolve()
    script = (rvc / rel).resolve()
    if not script.is_file():
        print(f"找不到官方脚本：{script}", file=sys.stderr)
        return 2
    module = script_to_module(rel)
    os.chdir(rvc)
    pythonpath = os.pathsep.join(
        [str(rvc)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    )
    os.environ["PYTHONPATH"] = pythonpath
    os.execv(sys.executable, [sys.executable, "-m", module, *argv[1:]])
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
