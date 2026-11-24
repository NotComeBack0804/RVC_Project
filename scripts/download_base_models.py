#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载公开基础模型（HuBERT、RMVPE）。不会下载他人音色模型。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rvc_realtime.paths import HUBERT_FILES, RMVPE_ONNX, RMVPE_PT, assets_dir, hubert_dir, rmvpe_dir


REPO = "lj1995/VoiceConversionWebUI"
REVISION = "main"


def _download_hf(repo: str, revision: str, include: str | None, filename: str | None, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "需要 huggingface_hub。请先运行 ./scripts/install_linux.sh 或 "
            "pip install huggingface_hub"
        ) from exc

    if filename:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_dir=str(local_dir),
        )
        print(f"已下载 {filename} -> {path}")
        return
    snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=[include] if include else None,
        local_dir=str(local_dir),
    )
    print(f"已下载 {include} -> {local_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 HuBERT / RMVPE 公开基础模型")
    parser.add_argument("--with-onnx", action="store_true", help="额外下载 rmvpe.onnx（Windows DirectML 的 RMVPE 必需；Windows 上默认开启）")
    parser.add_argument("--no-onnx", action="store_true", help="不要下载 rmvpe.onnx")
    parser.add_argument("--with-pretrained", action="store_true", help="额外下载官方训练底模（体积较大，仅在你准备训练自己的授权模型时使用）")
    args = parser.parse_args()
    want_onnx = (args.with_onnx or sys.platform == "win32") and not args.no_onnx

    print("只下载公开基础模型，不会下载或附带任何他人音色（.pth）。")
    print(f"目标目录：{assets_dir()}")

    # 按官方 README：hubert_base/* 放到 assets/ 下，形成 assets/hubert_base/
    _download_hf(REPO, REVISION, include="hubert_base/*", filename=None, local_dir=assets_dir())
    _download_hf(REPO, REVISION, include=None, filename="rmvpe.pt", local_dir=rmvpe_dir())

    if want_onnx:
        _download_hf(REPO, REVISION, include=None, filename="rmvpe.onnx", local_dir=rmvpe_dir())

    if args.with_pretrained:
        _download_hf(REPO, REVISION, include="pretrained/*", filename=None, local_dir=assets_dir())
        _download_hf(REPO, REVISION, include="pretrained_v2/*", filename=None, local_dir=assets_dir())
        # mute 样本是官方训练 filelist 需要的静音条，不是音色模型
        from rvc_realtime.paths import project_root, third_party_rvc
        dl = project_root() / ".model-downloads"
        _download_hf(REPO, REVISION, include=None, filename="mute.zip", local_dir=dl)
        zip_path = dl / "mute.zip"
        logs_dir = third_party_rvc() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if zip_path.is_file():
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(logs_dir)
            print(f"已解压 mute 样本到 {logs_dir}")
        print("训练底模已下载。请只使用已授权的参考录音进行训练。")
        # 官方训练从仓库根目录读取 assets/pretrained*
        rvc_assets = third_party_rvc() / "assets"
        rvc_assets.mkdir(parents=True, exist_ok=True)
        for name in ("pretrained", "pretrained_v2"):
            src = assets_dir() / name
            dest = rvc_assets / name
            if src.is_dir() and not dest.exists():
                dest.symlink_to(src.resolve())
                print(f"已链接 {dest} -> {src}")

    missing = []
    for name in HUBERT_FILES:
        path = hubert_dir() / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    rmvpe = rmvpe_dir() / RMVPE_PT
    if not rmvpe.is_file() or rmvpe.stat().st_size == 0:
        missing.append(str(rmvpe))
    if want_onnx:
        onnx = rmvpe_dir() / RMVPE_ONNX
        if not onnx.is_file() or onnx.stat().st_size == 0:
            missing.append(str(onnx))
    if missing:
        print("以下文件仍缺失：", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("基础模型已就绪。用户自己的 .pth 请放到 assets/weights/，索引放到 assets/indices/。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
