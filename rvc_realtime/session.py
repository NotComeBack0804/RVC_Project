"""实时变声会话：可开始/停止，支持直通与变声，以及文件模拟实时。"""
from __future__ import annotations

import threading
import time
from typing import Optional

from rvc_realtime.engine import open_reader, open_writer, run_engine
from rvc_realtime.errors import AudioIOError, UsageError
from rvc_realtime.processor import build_processor


class RealtimeSession:
    def __init__(self):
        self.running = False
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = ""
        self.frames = 0

    def start(
        self,
        mode: str,
        input_spec: str,
        output_spec: str,
        model: Optional[str] = None,
        index: Optional[str] = None,
        buffer_ms: int = 150,
        sample_rate: int = 48000,
        channels: int = 1,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        gain_db: float = 0.0,
        loudness_mode: str = "match-input",
        f0_method: str = "rmvpe",
        duration: Optional[float] = None,
        simulate_realtime: bool = False,
    ) -> None:
        if self.running:
            raise AudioIOError("实时会话已经在运行。请先停止。")
        if not input_spec or not output_spec:
            raise UsageError("请选择输入设备（麦克风）和输出设备（CABLE Input 或 Linux 虚拟线缆）。")
        self.error = None
        self.stats = "正在启动…"
        self.frames = 0
        self._stop.clear()
        self.running = True
        args = dict(
            mode=mode,
            input_spec=input_spec,
            output_spec=output_spec,
            model=model,
            index=index,
            buffer_ms=buffer_ms,
            sample_rate=sample_rate,
            channels=channels,
            f0_up_key=f0_up_key,
            index_rate=index_rate,
            gain_db=gain_db,
            loudness_mode=loudness_mode,
            f0_method=f0_method,
            duration=duration,
            simulate_realtime=simulate_realtime,
        )
        self._thread = threading.Thread(target=self._loop, kwargs=args, daemon=True)
        self._thread.start()

    def _loop(self, **kwargs):
        reader = writer = processor = None
        started = time.perf_counter()
        latencies = []
        try:
            processor = build_processor(
                mode=kwargs["mode"],
                model=kwargs.get("model"),
                index=kwargs.get("index"),
                sample_rate=kwargs["sample_rate"],
                channels=kwargs["channels"],
                f0_up_key=kwargs["f0_up_key"],
                index_rate=kwargs["index_rate"],
                f0_method=kwargs["f0_method"],
                loudness_mode=kwargs["loudness_mode"],
                gain_db=kwargs["gain_db"],
            )
            reader, in_label = open_reader(
                kwargs["input_spec"], kwargs["sample_rate"], kwargs["channels"], kwargs["buffer_ms"]
            )
            writer, out_label = open_writer(
                kwargs["output_spec"], kwargs["sample_rate"], kwargs["channels"], kwargs["buffer_ms"]
            )
            buffer_frames = max(1, int(kwargs["sample_rate"] * kwargs["buffer_ms"] / 1000))
            self.stats = f"运行中：{kwargs['mode']}  {in_label} → {out_label}  缓冲 {kwargs['buffer_ms']} ms"
            while not self._stop.is_set():
                chunk = reader.read(buffer_frames)
                if chunk is None or chunk.size == 0:
                    break
                t0 = time.perf_counter()
                out = processor.process(chunk)
                proc_s = time.perf_counter() - t0
                latencies.append(proc_s * 1000)
                if kwargs.get("simulate_realtime"):
                    remain = (chunk.shape[0] / float(kwargs["sample_rate"])) - proc_s
                    if remain > 0:
                        time.sleep(remain)
                writer.write(out)
                self.frames += int(chunk.shape[0])
                if kwargs.get("duration") and self.frames / kwargs["sample_rate"] >= kwargs["duration"]:
                    break
            elapsed = time.perf_counter() - started
            audio_s = self.frames / float(kwargs["sample_rate"]) if kwargs["sample_rate"] else 0
            avg = sum(latencies) / len(latencies) if latencies else 0
            extra = ""
            if hasattr(processor, "index_stats"):
                extra = "  " + processor.index_stats.summary()
            self.stats = (
                f"已停止。模式 {kwargs['mode']}，音频 {audio_s:.2f}s，墙钟 {elapsed:.2f}s，"
                f"平均块处理 {avg:.0f} ms，缓冲 {kwargs['buffer_ms']} ms。{extra}"
            )
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.stats = f"实时会话失败：{exc}"
        finally:
            for obj in (reader, writer, processor):
                if obj is not None and hasattr(obj, "close"):
                    try:
                        obj.close()
                    except Exception:
                        pass
            self.running = False

    def stop(self) -> str:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        self.running = False
        return self.stats or "已停止。"


def run_simulated_realtime(input_wav: str, output_wav: str, model: str, index: Optional[str], buffer_ms: int = 200, **kwargs):
    """无麦克风时的模拟实时验收：按块喂入文件并等待块时长。"""
    return run_engine(
        mode="convert",
        input_spec=input_wav,
        output_spec=output_wav,
        model=model,
        index=index,
        buffer_ms=buffer_ms,
        simulate_realtime=True,
        **kwargs,
    )
