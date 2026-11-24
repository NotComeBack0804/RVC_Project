#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整理已授权参考录音：切分静音、写出待提取片段。没有授权数据和确认开关时不会处理。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rvc_realtime.audio_io import load_wav, save_wav
from rvc_realtime.paths import testdata_dir


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


def list_audio(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def split_silence(data, sample_rate: int, thresh: float = 0.01, min_sec: float = 1.0, max_sec: float = 12.0, pad_ms: int = 80):
    import numpy as np

    mono = data.mean(axis=1) if data.ndim == 2 else data
    energy = np.abs(mono)
    voiced = energy > thresh
    if not voiced.any():
        return []
    pad = int(sample_rate * pad_ms / 1000)
    min_len = int(min_sec * sample_rate)
    max_len = int(max_sec * sample_rate)
    clips = []
    start = None
    for i, flag in enumerate(voiced):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            if end - start >= min_len:
                a = max(0, start - pad)
                b = min(len(mono), end + pad)
                while b - a > max_len:
                    clips.append(data[a : a + max_len])
                    a += max_len
                if b - a >= min_len:
                    clips.append(data[a:b])
            start = None
    if start is not None and len(mono) - start >= min_len:
        a = max(0, start - pad)
        clips.append(data[a : min(len(mono), a + max_len)])
    return clips


def main() -> int:
    parser = argparse.ArgumentParser(description="整理已授权参考录音")
    parser.add_argument("--i-have-authorization", action="store_true", help="确认这些录音是本人声音或已获得明确授权")
    parser.add_argument("--input-dir", default=str(testdata_dir() / "authorized_reference"))
    parser.add_argument("--output-dir", default=str(testdata_dir() / "prepared"))
    args = parser.parse_args()

    if not args.i_have_authorization:
        print(
            "拒绝处理：请先确认录音来源合法，并加上 --i-have-authorization。\n"
            "只使用你本人的声音，或已获得明确授权的声音。",
            file=sys.stderr,
        )
        return 2

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = list_audio(input_dir) if input_dir.is_dir() else []
    if not files:
        print(
            f"未在 {input_dir} 找到参考录音。请把已授权的 WAV/FLAC 放到该目录后再运行。\n"
            "在提供授权录音之前，不会开始训练。"
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in files:
        try:
            data, sr = load_wav(src)
        except Exception as exc:  # noqa: BLE001
            print(f"跳过 {src.name}：{exc}")
            continue
        clips = split_silence(data, sr)
        if not clips:
            print(f"{src.name}：没有切出足够长的有效语音，请检查音量或静音阈值。")
            continue
        for i, clip in enumerate(clips, start=1):
            dest = output_dir / f"{src.stem}_{i:03d}.wav"
            save_wav(dest, clip, sr)
            written += 1
        print(f"{src.name} -> {len(clips)} 个片段")
    print(f"共写出 {written} 个片段到 {output_dir}")
    print("下一步：在完整 RVC CPU 环境中运行 python tools/train/extract_features.py 查看提取说明。")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
