"""输出响度：与输入对齐，并用峰值限制防止削波。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rvc_realtime.audio_io import load_wav, match_channels, save_wav


def rms(frames: np.ndarray) -> float:
    data = np.asarray(frames, dtype=np.float32)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))


def peak(frames: np.ndarray) -> float:
    data = np.asarray(frames, dtype=np.float32)
    if data.size == 0:
        return 0.0
    return float(np.max(np.abs(data)))


def limit_peak(frames: np.ndarray, peak_limit: float = 0.95) -> np.ndarray:
    out = np.asarray(frames, dtype=np.float32)
    current = peak(out)
    if current > peak_limit > 0:
        out = out * (peak_limit / current)
    return np.clip(out, -1.0, 1.0)


class GainSmoother:
    def __init__(self, attack: float = 0.35, release: float = 0.12):
        self.attack = attack
        self.release = release
        self.value: float | None = None

    def step(self, target: float) -> float:
        if self.value is None:
            self.value = target
            return target
        coef = self.attack if target > self.value else self.release
        self.value = self.value + (target - self.value) * coef
        return self.value


def compute_makeup_gain(
    input_rms: float,
    output_rms: float,
    extra_gain: float = 1.0,
    min_gain: float = 0.25,
    max_gain: float = 12.0,
) -> float:
    gain = extra_gain
    if output_rms > 1e-6 and input_rms > 1e-8:
        gain *= input_rms / output_rms
    return float(np.clip(gain, min_gain, max_gain))


def apply_chunk_loudness(
    output_frames: np.ndarray,
    input_frames: np.ndarray,
    mode: str = "match-input",
    extra_gain: float = 1.0,
    peak_limit: float = 0.95,
    smoother: GainSmoother | None = None,
) -> tuple[np.ndarray, float]:
    out = np.asarray(output_frames, dtype=np.float32)
    if mode == "off":
        gain = float(np.clip(extra_gain, 0.05, 12.0))
        if smoother is not None:
            gain = smoother.step(gain)
        return limit_peak(out * gain, peak_limit), gain
    gain = compute_makeup_gain(rms(input_frames), rms(out), extra_gain=extra_gain)
    if smoother is not None:
        gain = smoother.step(gain)
    return limit_peak(out * gain, peak_limit), gain


def finalize_file_loudness(
    source: Path,
    dest: Path,
    mode: str = "match-input",
    extra_gain: float = 1.0,
    peak_limit: float = 0.95,
) -> dict:
    src, ssr = load_wav(source)
    out, osr = load_wav(dest)
    if osr != ssr:
        from rvc_realtime.audio_io import resample

        src = resample(src, ssr, osr)
    src = match_channels(src, out.shape[1] if out.ndim == 2 else 1)
    before = {"rms": rms(out), "peak": peak(out)}
    if mode == "off" and extra_gain == 1.0:
        after = before
        return {"gain": 1.0, "before": before, "after": after, "changed": False}
    n = min(src.shape[0], out.shape[0])
    gain = compute_makeup_gain(rms(src[:n]), rms(out[:n]), extra_gain=extra_gain)
    adjusted = limit_peak(out * gain, peak_limit)
    save_wav(dest, adjusted, osr)
    after = {"rms": rms(adjusted), "peak": peak(adjusted)}
    return {"gain": gain, "before": before, "after": after, "changed": True}
