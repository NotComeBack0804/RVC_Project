from __future__ import annotations

from pathlib import Path

import numpy as np

from rvc_realtime.audio_io import load_wav, save_wav


def generate_test_tone(
    path: Path,
    sample_rate: int = 48000,
    duration: float = 4.0,
    frequencies: tuple[float, ...] = (440.0, 880.0),
) -> Path:
    """生成带淡入淡出的分段测试音，便于在录到的音频里做频率核对。"""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    audio = np.zeros(n, dtype=np.float32)
    seg = n // max(1, len(frequencies))
    for i, freq in enumerate(frequencies):
        start = i * seg
        end = n if i == len(frequencies) - 1 else (i + 1) * seg
        sl = slice(start, end)
        audio[sl] = 0.25 * np.sin(2.0 * np.pi * freq * t[sl])
    fade = min(int(0.02 * sample_rate), n // 8)
    if fade > 0:
        audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(path, audio.reshape(-1, 1), sample_rate)
    return path


def analyze_wav(path: Path) -> dict:
    data, sr = load_wav(path)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    n = mono.shape[0]
    duration = n / float(sr) if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if n else 0.0
    peak = float(np.max(np.abs(mono))) if n else 0.0
    result = {
        "path": str(path),
        "sample_rate": sr,
        "channels": int(data.shape[1]) if data.ndim == 2 else 1,
        "frames": int(n),
        "duration_sec": duration,
        "rms": rms,
        "peak": peak,
        "has_sound": rms > 1e-4 and peak > 1e-3,
        "dominant_freq_hz": None,
    }
    if n >= sr // 10:
        window = mono * np.hanning(n)
        spec = np.fft.rfft(window)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        # 忽略接近直流的分量
        min_bin = int(np.searchsorted(freqs, 80.0))
        if mag[min_bin:].size:
            idx = int(np.argmax(mag[min_bin:]) + min_bin)
            result["dominant_freq_hz"] = float(freqs[idx])
    return result


def freq_near(actual: float | None, expected: float, tol_hz: float = 30.0) -> bool:
    if actual is None:
        return False
    return abs(actual - expected) <= tol_hz


def compare_passthrough(source: Path, recorded: Path) -> dict:
    src = analyze_wav(source)
    rec = analyze_wav(recorded)
    duration_ok = rec["duration_sec"] >= max(0.3, src["duration_sec"] * 0.5)
    sound_ok = rec["has_sound"]
    return {
        "source": src,
        "recorded": rec,
        "duration_ok": duration_ok,
        "sound_ok": sound_ok,
        "passed": bool(duration_ok and sound_ok),
    }
