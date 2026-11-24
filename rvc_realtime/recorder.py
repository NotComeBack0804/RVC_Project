"""录制系统正在播放的声音（扬声器 / 环回）。麦克风只是次要备选。无环回时可导入 WAV。"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from rvc_realtime.audio_io import PulseReader, save_wav
from rvc_realtime.devices import CaptureChoice, default_loopback_label, resolve_capture
from rvc_realtime.errors import AudioIOError
from rvc_realtime.paths import testdata_dir
from rvc_realtime.pulse import pulse_available


class SpeakerRecorder:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._saved = False
        self.running = False
        self.error: Optional[str] = None
        self.path: Optional[Path] = None
        self.sample_rate = 48000
        self.channels = 1
        self.started_at: Optional[float] = None
        self._ended_at: Optional[float] = None
        self.last_peak = 0.0
        self.last_rms = 0.0
        self.device = ""
        self.max_seconds: Optional[float] = None
        self.auto_stopped = False
        self.capture: Optional[CaptureChoice] = None

    def status_text(self) -> str:
        if self.error and not self.running:
            return f"录音失败：{self.error}"
        dur = self.duration()
        if self.running:
            state = "正在录制扬声器"
        elif self.auto_stopped:
            state = "已到目标时长，已保存"
        else:
            state = "已停止"
        target = f" / {self.max_seconds:.0f}s" if self.max_seconds else ""
        dest = str(self.path) if self.path else "（尚未保存）"
        return (
            f"{state}  设备：{self.device or '未选择'}  "
            f"时长：{dur:.1f}s{target}  电平 RMS={self.last_rms:.3f}  峰值={self.last_peak:.3f}\n"
            f"文件：{dest}"
        )

    def duration(self) -> float:
        if self.started_at is None:
            return 0.0
        end = time.time() if self.running else (self._ended_at or time.time())
        return max(0.0, end - self.started_at)

    def start(
        self,
        device: str,
        sample_rate: int = 48000,
        channels: int = 1,
        output_dir: Optional[Path] = None,
        max_seconds: Optional[float] = None,
    ) -> None:
        if self.running:
            raise AudioIOError("已经在录音。请先停止。")
        capture = resolve_capture(device)
        if capture is None:
            raise AudioIOError(
                "没有可用的扬声器环回。Windows 请使用 WASAPI 环回或启用「立体声混音」；"
                "Linux 请选择 speakers.monitor 或等价的播放设备监听源。也可以导入已有的短 WAV。"
            )
        dest_dir = Path(output_dir) if output_dir else (testdata_dir() / "authorized_reference")
        dest_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = dest_dir / f"speaker_{stamp}.wav"
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = capture.label
        self.capture = capture
        self.error = None
        self._chunks = []
        self._saved = False
        self._stop.clear()
        self.last_peak = 0.0
        self.last_rms = 0.0
        self.started_at = time.time()
        self._ended_at = None
        self.max_seconds = max_seconds
        self.auto_stopped = False
        self.running = True
        self._thread = threading.Thread(target=self._loop, args=(capture,), daemon=True)
        self._thread.start()

    def _open_reader(self, capture: CaptureChoice):
        if capture.backend == "pulse" or (capture.backend != "sounddevice" and pulse_available() and not capture.wasapi_loopback):
            if pulse_available():
                reader = PulseReader(capture.name, self.sample_rate, self.channels, 80)
                return ("pulse", reader, None)
        try:
            import sounddevice as sd

            q: queue.Queue = queue.Queue()

            def cb(indata, frames, time_info, status):
                q.put(indata.copy())

            kwargs = {
                "samplerate": self.sample_rate,
                "dtype": "float32",
                "callback": cb,
                "blocksize": int(self.sample_rate * 0.1),
            }
            if capture.device_index is not None:
                kwargs["device"] = capture.device_index
            else:
                kwargs["device"] = _resolve_sd_device(capture.name, input=not capture.wasapi_loopback)
            if capture.wasapi_loopback:
                if not hasattr(sd, "WasapiSettings"):
                    raise AudioIOError(
                        "当前 sounddevice 不支持 WASAPI 环回。请启用 Windows「立体声混音」，或升级 sounddevice 后再试。"
                    )
                kwargs["extra_settings"] = sd.WasapiSettings(loopback=True)
                kwargs["channels"] = max(self.channels, 2)
            else:
                kwargs["channels"] = self.channels
            stream = sd.InputStream(**kwargs)
            stream.start()
            return ("sd", stream, q)
        except AudioIOError:
            raise
        except Exception as exc:
            if pulse_available() and not capture.wasapi_loopback:
                reader = PulseReader(capture.name, self.sample_rate, self.channels, 80)
                return ("pulse", reader, None)
            raise AudioIOError(f"无法打开录音设备「{capture.label}」：{exc}") from exc

    def _loop(self, capture: CaptureChoice) -> None:
        handle = None
        try:
            kind, obj, q = self._open_reader(capture)
            handle = (kind, obj)
            while not self._stop.is_set():
                if self.max_seconds and self.started_at and (time.time() - self.started_at) >= self.max_seconds:
                    self.auto_stopped = True
                    break
                if kind == "sd":
                    try:
                        chunk = q.get(timeout=0.3)
                    except Exception:
                        continue
                    data = np.asarray(chunk, dtype=np.float32)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    if data.shape[1] > 1:
                        data = data.mean(axis=1, keepdims=True)
                else:
                    data = obj.read(int(self.sample_rate * 0.1))
                    if data is None or data.size == 0:
                        continue
                    data = np.asarray(data, dtype=np.float32)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                with self._lock:
                    self._chunks.append(data)
                    self.last_peak = float(np.max(np.abs(data)))
                    self.last_rms = float(np.sqrt(np.mean(np.square(data))))
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
        finally:
            if handle:
                kind, obj = handle
                try:
                    if kind == "sd":
                        obj.stop()
                        obj.close()
                    else:
                        obj.close()
                except Exception:
                    pass
            self.running = False
            if self._ended_at is None:
                self._ended_at = time.time()

    def stop(self) -> Optional[Path]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.running = False
        if self._ended_at is None:
            self._ended_at = time.time()
        return self._finalize_save()

    def _finalize_save(self) -> Optional[Path]:
        if self._saved:
            return self.path
        with self._lock:
            chunks = list(self._chunks)
        if not chunks:
            if not self.error:
                self.error = (
                    "没有录到音频。请确认选择的是扬声器环回，并且电脑正在播放要录的发言。"
                    "也可以导入已有的短 WAV。"
                )
            return None
        data = np.concatenate(chunks, axis=0)
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        if float(np.sqrt(np.mean(np.square(data)))) < 1e-4:
            self.error = (
                "录到的信号接近静音。请确认电脑正在播放要录的发言，"
                "录音时先不要开实时变声，也不要从 CABLE 录（以免把变声结果再录进去）。"
                "当前环境若没有可捕获的系统音频，请改用「导入已有短 WAV」。"
            )
        if self.path:
            save_wav(self.path, data, self.sample_rate)
            self._saved = True
        return self.path


# 兼容旧名称，避免外部 import 失败；行为已改为扬声器环回优先。
MicRecorder = SpeakerRecorder


def _resolve_sd_device(name: str, input: bool):
    import sounddevice as sd

    devices = sd.query_devices()
    lowered = name.lower()
    for i, dev in enumerate(devices):
        ok = int(dev.get("max_input_channels" if input else "max_output_channels") or 0) > 0
        if ok and (name == dev.get("name") or lowered in str(dev.get("name", "")).lower()):
            return i
    try:
        return int(name)
    except ValueError:
        return name


def import_authorized_wav(src: str | Path, output_dir: Optional[Path] = None) -> Path:
    from rvc_realtime.audio_io import load_wav, match_channels, resample

    src_path = Path(src).expanduser()
    if not src_path.is_file():
        raise AudioIOError(f"找不到要导入的音频：{src_path}")
    dest_dir = Path(output_dir) if output_dir else (testdata_dir() / "authorized_reference")
    dest_dir.mkdir(parents=True, exist_ok=True)
    data, sr = load_wav(src_path)
    data = match_channels(resample(data, sr, 48000), 1)
    dest = dest_dir / f"imported_{src_path.stem}_{time.strftime('%Y%m%d-%H%M%S')}.wav"
    save_wav(dest, data, 48000)
    return dest


def list_input_devices() -> list[str]:
    from rvc_realtime.devices import loopback_labels

    return loopback_labels()


def default_record_device() -> Optional[str]:
    return default_loopback_label()
