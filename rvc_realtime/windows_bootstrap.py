# -*- coding: utf-8 -*-
"""Portable Windows 包的启动辅助：PATH、官方资产目录连接、底模检查。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rvc_realtime.paths import (
    HUBERT_FILES,
    RMVPE_ONNX,
    RMVPE_PT,
    assets_dir,
    hubert_dir,
    project_root,
    rmvpe_dir,
    third_party_rvc,
)


ASSET_LINKS = ("hubert_base", "rmvpe", "pretrained", "pretrained_v2", "weights", "indices")


def runtime_dir() -> Path | None:
    """便携包中 runtime\\python.exe 所在目录；源码安装返回 None。"""
    exe = Path(sys.executable).resolve()
    if exe.name.lower() == "python.exe" and exe.parent.name.lower() == "runtime":
        return exe.parent
    # also detect embeddable layout
    if (exe.parent / "python312._pth").is_file():
        return exe.parent
    return None


def configure_runtime_path() -> None:
    runtime = runtime_dir()
    if runtime is None:
        return
    extras = [
        runtime,
        runtime / "Scripts",
        runtime / "ffmpeg",
        runtime / "Lib" / "site-packages" / "torch" / "lib",
        runtime / "Lib" / "site-packages" / "onnxruntime" / "capi",
        runtime / "Lib" / "site-packages" / "torch_directml",
    ]
    existing = [p for p in extras if p.is_dir()]
    if not existing:
        return
    prefix = os.pathsep.join(str(p) for p in existing)
    path = os.environ.get("PATH", "")
    if prefix not in path:
        os.environ["PATH"] = prefix + os.pathsep + path
    adder = getattr(os, "add_dll_directory", None)
    if adder is not None:
        for p in existing:
            try:
                adder(str(p))
            except OSError:
                pass


def _same_dir(link: Path, target: Path) -> bool:
    try:
        return link.resolve() == target.resolve()
    except OSError:
        return False


def _make_junction(link: Path, target: Path) -> None:
    if not target.is_dir():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if _same_dir(link, target):
            return
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            return
    if sys.platform != "win32":
        try:
            link.symlink_to(os.path.relpath(target, start=link.parent), target_is_directory=True)
        except OSError:
            pass
        return
    # Directory junction does not require Administrator / Developer Mode.
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_rvc_asset_links() -> None:
    """让官方 RVC 的相对路径 assets/rmvpe 等指向本项目 assets。"""
    rvc = third_party_rvc()
    rvc_assets = rvc / "assets"
    rvc_assets.mkdir(parents=True, exist_ok=True)
    root_assets = assets_dir()
    for name in ASSET_LINKS:
        target = root_assets / name
        if not target.is_dir():
            continue
        _make_junction(rvc_assets / name, target)


def missing_base_models() -> list[str]:
    missing: list[str] = []
    for name in HUBERT_FILES:
        path = hubert_dir() / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(project_root())))
    for name in (RMVPE_PT, RMVPE_ONNX):
        path = rmvpe_dir() / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(project_root())))
    return missing


def maybe_download_base_models() -> int:
    missing = missing_base_models()
    if not missing:
        return 0
    print("缺少以下公开底模，正在自动下载（不会下载他人音色）：")
    for item in missing:
        print(f"  - {item}")
    script = project_root() / "scripts" / "download_base_models.py"
    args = [sys.executable, str(script), "--with-onnx"]
    # 若训练底模也不在，一并补上默认 v2 40k 所需文件所在的官方全集
    pre_g = assets_dir() / "pretrained_v2" / "f0G40k.pth"
    pre_d = assets_dir() / "pretrained_v2" / "f0D40k.pth"
    if not pre_g.is_file() or not pre_d.is_file():
        args.append("--with-pretrained")
    return subprocess.call(args)


def prepare_portable() -> int:
    configure_runtime_path()
    ensure_rvc_asset_links()
    return maybe_download_base_models()


def main() -> int:
    code = prepare_portable()
    missing = missing_base_models()
    if missing:
        print("底模仍缺失，无法启动。请检查网络后重试，或重新解压完整的模型包。", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1 if code == 0 else code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
