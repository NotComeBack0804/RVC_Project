from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from rvc_realtime.audio_io import PulseReader, PulseWriter, WavReader, WavWriter
from rvc_realtime.errors import AudioIOError, UsageError
from rvc_realtime.processor import build_processor


@dataclass
class RunResult:
    mode: str
    frames: int
    sample_rate: int
    elapsed_sec: float
    rtf: float
    estimated_buffer_latency_ms: float
    input_label: str
    output_label: str
    notes: list[str] = field(default_factory=list)
    chunk_latency_ms: list[float] = field(default_factory=list)
    index_stats: dict | None = None
    loudness: dict | None = None
    simulate_realtime: bool = False


def _is_probably_file(value: str) -> bool:
    lower = value.lower()
    if lower.endswith((".wav", ".flac", ".ogg", ".mp3")):
        return True
    path = Path(value)
    return path.exists() and path.is_file()


def open_reader(spec: str, sample_rate: int, channels: int, buffer_ms: int):
    if _is_probably_file(spec):
        path = Path(spec).expanduser()
        if not path.is_file():
            raise AudioIOError(f"找不到输入文件：{path}")
        return WavReader(path, sample_rate, channels), f"file:{path}"
    return PulseReader(spec, sample_rate, channels, buffer_ms), f"pulse:{spec}"


def open_writer(spec: str, sample_rate: int, channels: int, buffer_ms: int):
    if _is_probably_file(spec) or Path(spec).suffix:
        path = Path(spec).expanduser()
        if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
            # 允许用户写 output.wav；若没有后缀则视为设备名
            if path.suffix == "":
                return PulseWriter(spec, sample_rate, channels, buffer_ms), f"pulse:{spec}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return WavWriter(path, sample_rate, channels), f"file:{path}"
    return PulseWriter(spec, sample_rate, channels, buffer_ms), f"pulse:{spec}"


def run_engine(
    mode: str,
    input_spec: str,
    output_spec: str,
    sample_rate: int = 48000,
    channels: int = 1,
    buffer_ms: int = 100,
    duration: Optional[float] = None,
    model: Optional[str] = None,
    index: Optional[str] = None,
    f0_up_key: int = 0,
    formant: float = 0.0,
    index_rate: float = 0.5,
    f0_method: str = "rmvpe",
    loudness_mode: str = "match-input",
    gain_db: float = 0.0,
    peak_limit: float = 0.95,
    simulate_realtime: bool = False,
) -> RunResult:
    if buffer_ms < 20:
        raise UsageError("缓冲区过小（至少 20 ms）。建议先从 100–200 ms 开始。")
    processor = build_processor(
        mode=mode,
        model=model,
        index=index,
        sample_rate=sample_rate,
        channels=channels,
        f0_up_key=f0_up_key,
        formant=formant,
        index_rate=index_rate,
        f0_method=f0_method,
        loudness_mode=loudness_mode,
        gain_db=gain_db,
        peak_limit=peak_limit,
    )
    reader = writer = None
    notes: list[str] = []
    compute = getattr(processor, "compute", None)
    if compute is not None:
        notes.append(compute.message)
    chunk_latency_ms: list[float] = []
    buffer_frames = max(1, int(sample_rate * buffer_ms / 1000))
    started = time.perf_counter()
    total_frames = 0
    in_label = input_spec
    out_label = output_spec
    try:
        reader, in_label = open_reader(input_spec, sample_rate, channels, buffer_ms)
        writer, out_label = open_writer(output_spec, sample_rate, channels, buffer_ms)
        while True:
            chunk = reader.read(buffer_frames)
            if chunk is None or chunk.size == 0:
                break
            t0 = time.perf_counter()
            out = processor.process(chunk)
            proc_s = time.perf_counter() - t0
            chunk_latency_ms.append(proc_s * 1000.0)
            if simulate_realtime:
                remain = (chunk.shape[0] / float(sample_rate)) - proc_s
                if remain > 0:
                    time.sleep(remain)
            writer.write(out)
            total_frames += int(chunk.shape[0])
            if duration is not None and total_frames / sample_rate >= duration:
                break
    finally:
        for obj in (reader, writer, processor):
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    notes.append(f"关闭 {type(obj).__name__} 时出现问题。")
    loudness_info = None
    if mode == "convert" and loudness_mode != "off":
        from pathlib import Path as _Path
        from rvc_realtime.loudness import finalize_file_loudness

        src = _Path(input_spec)
        dst = _Path(output_spec)
        if src.is_file() and dst.is_file() and dst.stat().st_size > 44:
            try:
                loudness_info = finalize_file_loudness(
                    src, dst, mode=loudness_mode, extra_gain=10 ** (gain_db / 20.0), peak_limit=peak_limit
                )
                notes.append(
                    f"文件响度已对齐：增益 {loudness_info['gain']:.2f}x，"
                    f"RMS {loudness_info['before']['rms']:.4f} → {loudness_info['after']['rms']:.4f}，"
                    f"峰值 {loudness_info['after']['peak']:.3f}。"
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"文件响度对齐未完成：{exc}")
    elapsed = max(1e-6, time.perf_counter() - started)
    audio_sec = total_frames / float(sample_rate) if sample_rate else 0.0
    rtf = (elapsed / audio_sec) if audio_sec else float("inf")
    # 处理 RTF 不含模拟实时的 sleep
    proc_only = sum(chunk_latency_ms) / 1000.0 if chunk_latency_ms else elapsed
    proc_rtf = (proc_only / audio_sec) if audio_sec else float("inf")
    est = float(buffer_ms * 2)
    if chunk_latency_ms:
        avg = sum(chunk_latency_ms) / len(chunk_latency_ms)
        p95 = sorted(chunk_latency_ms)[max(0, int(len(chunk_latency_ms) * 0.95) - 1)]
        notes.append(
            f"分块处理延迟：平均 {avg:.0f} ms，约 95% 分位 {p95:.0f} ms，"
            f"块时长 {buffer_ms} ms，处理 RTF={proc_rtf:.3f}。"
        )
        if avg > buffer_ms and mode == "convert":
            notes.append("平均处理时间超过块时长，实时通话可能出现断续。请增大 --buffer-ms。")
    if proc_rtf > 1.05 and mode == "convert" and not simulate_realtime:
        notes.append(
            f"处理慢于实时（RTF={rtf:.2f}）。请增大 --buffer-ms，或确认计算设备是否已走到 DirectML（见上方设备说明）。"
        )
    if est > 250 and not simulate_realtime:
        notes.append(f"估计缓冲延迟约 {est:.0f} ms，可能在通话中感到明显迟滞。建议先保证稳定，再逐步减小缓冲。")
    if simulate_realtime:
        notes.append("已按块时长等待（模拟实时喂入）。报告中的处理 RTF 不含等待时间。")
    index_stats = getattr(processor, "index_stats", None)
    index_dict = index_stats.as_dict() if index_stats is not None and hasattr(index_stats, "as_dict") else None
    if index_stats is not None and hasattr(index_stats, "summary"):
        notes.append(index_stats.summary())
    return RunResult(
        mode=mode,
        frames=total_frames,
        sample_rate=sample_rate,
        elapsed_sec=elapsed,
        rtf=proc_rtf if simulate_realtime else rtf,
        estimated_buffer_latency_ms=est,
        input_label=in_label,
        output_label=out_label,
        notes=notes,
        chunk_latency_ms=chunk_latency_ms,
        index_stats=index_dict,
        loudness=loudness_info,
        simulate_realtime=simulate_realtime,
    )


def format_run_result(result: RunResult) -> str:
    audio_sec = result.frames / float(result.sample_rate) if result.sample_rate else 0.0
    lines = [
        f"模式：{result.mode}",
        f"输入：{result.input_label}",
        f"输出：{result.output_label}",
        f"音频时长：{audio_sec:.3f} s",
        f"处理耗时：{result.elapsed_sec:.3f} s",
        f"实时率 RTF：{result.rtf:.3f}（小于 1 表示快于实时）",
        f"估计缓冲延迟：约 {result.estimated_buffer_latency_ms:.0f} ms（输入+输出缓冲；不含驱动与聊天软件）",
        "延迟目标约 100–200 ms；明显超过约 250 ms 时，通话可能感到迟滞。",
        f"模拟实时：{'是' if result.simulate_realtime else '否'}",
    ]
    for note in result.notes:
        lines.append(f"注意：{note}")
    return "\n".join(lines)
