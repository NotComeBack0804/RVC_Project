from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from rvc_realtime.errors import AudioIOError


def _as_float32(mono_or_2d: np.ndarray) -> np.ndarray:
    data = np.asarray(mono_or_2d)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
        peak = np.max(np.abs(data)) if data.size else 0
        if peak > 1.5:
            data = data / 32768.0
    return np.clip(data, -1.0, 1.0)


def _to_int16_bytes(frames: np.ndarray) -> bytes:
    clipped = np.clip(frames, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return data, int(sr)
    except Exception as exc:  # noqa: BLE001
        # 回退到标准库 wave，只支持 PCM WAV。
        import wave

        try:
            with wave.open(str(path), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            if sw != 2:
                raise AudioIOError(f"不支持的采样宽度 {sw}，请使用 16-bit PCM WAV：{path}")
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                data = data.reshape(-1, ch)
            else:
                data = data.reshape(-1, 1)
            return data, sr
        except AudioIOError:
            raise
        except Exception as exc2:  # noqa: BLE001
            raise AudioIOError(f"无法读取音频文件 {path}：{exc} / {exc2}") from exc2


def save_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _as_float32(data)
    try:
        import soundfile as sf

        sf.write(str(path), frames, sample_rate, subtype="PCM_16")
        return
    except Exception:
        import wave

        pcm = (np.clip(frames, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(frames.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())


def resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    frames = _as_float32(data)
    if src_sr == dst_sr:
        return frames
    if frames.size == 0:
        return frames
    duration = frames.shape[0] / float(src_sr)
    dst_n = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, 1.0, frames.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, dst_n, endpoint=False)
    channels = []
    for ch in range(frames.shape[1]):
        channels.append(np.interp(dst_x, src_x, frames[:, ch]).astype(np.float32))
    return np.stack(channels, axis=1)


def match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    frames = _as_float32(data)
    if frames.shape[1] == channels:
        return frames
    if channels == 1:
        return np.mean(frames, axis=1, keepdims=True).astype(np.float32)
    if frames.shape[1] == 1:
        return np.repeat(frames, channels, axis=1)
    if frames.shape[1] > channels:
        return frames[:, :channels]
    pad = np.zeros((frames.shape[0], channels - frames.shape[1]), dtype=np.float32)
    return np.concatenate([frames, pad], axis=1)


class WavReader:
    def __init__(self, path: Path, sample_rate: int, channels: int):
        data, sr = load_wav(path)
        data = resample(data, sr, sample_rate)
        self.data = match_channels(data, channels)
        self.pos = 0
        self.sample_rate = sample_rate
        self.channels = channels
        self.source_sr = sr
        self.path = path

    def read(self, frames: int) -> Optional[np.ndarray]:
        if self.pos >= self.data.shape[0]:
            return None
        end = min(self.pos + frames, self.data.shape[0])
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def close(self) -> None:
        return


class WavWriter:
    def __init__(self, path: Path, sample_rate: int, channels: int):
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunks: list[np.ndarray] = []

    def write(self, frames: np.ndarray) -> None:
        self.chunks.append(match_channels(frames, self.channels))

    def close(self) -> None:
        if self.chunks:
            data = np.concatenate(self.chunks, axis=0)
        else:
            data = np.zeros((0, self.channels), dtype=np.float32)
        save_wav(self.path, data, self.sample_rate)


def _pulse_cmd_available(name: str) -> bool:
    from shutil import which

    return which(name) is not None


class PulseReader:
    """从 PulseAudio 源（包括虚拟麦克风）读取 PCM。"""

    def __init__(self, source: str, sample_rate: int, channels: int, latency_ms: int):
        if not _pulse_cmd_available("parec"):
            raise AudioIOError("未找到 parec。请安装 pulseaudio-utils。")
        self.source = source
        self.sample_rate = sample_rate
        self.channels = channels
        cmd = [
            "parec",
            "--raw",
            "--format=s16le",
            f"--rate={sample_rate}",
            f"--channels={channels}",
            f"--latency-msec={max(10, latency_ms)}",
            "-d",
            source,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioIOError(f"无法启动 parec 读取设备 {source}：{exc}") from exc
        time.sleep(0.15)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode("utf-8", "ignore") if self.proc.stderr else ""
            raise AudioIOError(f"parec 无法打开源 {source}。{err.strip()}")

    def read(self, frames: int) -> Optional[np.ndarray]:
        if self.proc.stdout is None:
            return None
        nbytes = frames * self.channels * 2
        raw = self.proc.stdout.read(nbytes)
        if not raw:
            return None
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if data.size % self.channels:
            data = data[: data.size - (data.size % self.channels)]
        return data.reshape(-1, self.channels)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class PulseWriter:
    """向 PulseAudio 沉（包括虚拟线缆输入）写入 PCM。"""

    def __init__(self, sink: str, sample_rate: int, channels: int, latency_ms: int):
        if not _pulse_cmd_available("pacat"):
            raise AudioIOError("未找到 pacat。请安装 pulseaudio-utils。")
        self.sink = sink
        self.sample_rate = sample_rate
        self.channels = channels
        cmd = [
            "pacat",
            "--raw",
            "--format=s16le",
            f"--rate={sample_rate}",
            f"--channels={channels}",
            f"--latency-msec={max(10, latency_ms)}",
            "-d",
            sink,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioIOError(f"无法启动 pacat 写入设备 {sink}：{exc}") from exc
        time.sleep(0.15)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode("utf-8", "ignore") if self.proc.stderr else ""
            raise AudioIOError(f"pacat 无法打开沉 {sink}。{err.strip()}")

    def write(self, frames: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise AudioIOError("pacat 标准输入已关闭。")
        payload = _to_int16_bytes(match_channels(frames, self.channels))
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = ""
            if self.proc.stderr is not None:
                try:
                    err = self.proc.stderr.read().decode("utf-8", "ignore")
                except Exception:
                    err = ""
            raise AudioIOError(f"写入 PulseAudio 设备 {self.sink} 失败。{err.strip()}") from exc

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
