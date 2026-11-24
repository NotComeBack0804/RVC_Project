from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rvc_realtime import pulse
from rvc_realtime.audio_io import PulseReader, WavWriter, save_wav
from rvc_realtime.engine import run_engine
from rvc_realtime.errors import AppError, ModelNotFoundError
from rvc_realtime.paths import project_root, testdata_dir
from rvc_realtime.processor import build_processor
from rvc_realtime.signal import analyze_wav, compare_passthrough, generate_test_tone
from rvc_realtime.status import gather_status


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())


def _record_background(source: str, output: Path, sample_rate: int, channels: int, buffer_ms: int) -> PulseReader:
    return PulseReader(source, sample_rate, channels, buffer_ms)


def run_selftest(skip_pulse: bool = False) -> dict[str, Any]:
    root = project_root()
    generated = testdata_dir() / "generated"
    recorded = testdata_dir() / "recorded"
    generated.mkdir(parents=True, exist_ok=True)
    recorded.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "time": _now_iso(),
        "project": str(root),
        "python": sys.version,
        "steps": [],
        "passed": True,
    }

    def add(name: str, ok: bool, **kwargs):
        step = {"name": name, "ok": ok, **kwargs}
        report["steps"].append(step)
        if not ok:
            report["passed"] = False
        return step

    add("status_snapshot", True, text=gather_status())

    tone = generated / "test_tone.wav"
    generate_test_tone(tone, sample_rate=48000, duration=4.0, frequencies=(440.0, 880.0))
    info = analyze_wav(tone)
    add("generate_test_tone", info["has_sound"], analysis=info, path=str(tone))

    out_file = recorded / "passthrough_file.wav"
    try:
        result = run_engine(
            mode="passthrough",
            input_spec=str(tone),
            output_spec=str(out_file),
            sample_rate=48000,
            channels=1,
            buffer_ms=100,
        )
        cmp = compare_passthrough(tone, out_file)
        add(
            "passthrough_file",
            cmp["passed"],
            rtf=result.rtf,
            elapsed_sec=result.elapsed_sec,
            estimated_buffer_latency_ms=result.estimated_buffer_latency_ms,
            comparison=cmp,
        )
    except Exception as exc:  # noqa: BLE001
        add("passthrough_file", False, error=str(exc))

    # 缺模型时变声路径必须明确报错、不崩溃
    try:
        build_processor(mode="convert", model=None, index=None, sample_rate=48000, channels=1)
        add("convert_missing_model", False, error="变声路径在缺少模型时没有报错。")
    except ModelNotFoundError as exc:
        msg = exc.message
        add(
            "convert_missing_model",
            "缺少" in msg or "未找到" in msg or "没有 .pth" in msg,
            exit_code=exc.exit_code,
            message=msg,
        )
    except AppError as exc:
        add("convert_missing_model", False, error=f"抛出了其他 AppError：{exc}")
    except Exception as exc:  # noqa: BLE001
        add("convert_missing_model", False, error=f"非预期异常：{type(exc).__name__}: {exc}")

    if skip_pulse:
        add("pulse_routing", True, skipped=True, reason="按参数跳过 PulseAudio 测试")
    elif not pulse.pulse_available():
        add("pulse_routing", False, error="未找到 pactl，无法测试虚拟音频。")
    else:
        try:
            routing = pulse.setup_virtual_audio(sample_rate=48000, channels=1)
            add("setup_virtual_audio", True, routing=routing)
            source = routing["cable_out"]
            rec_path = recorded / "passthrough_pulse.wav"
            rec_seconds = 5.0
            reader = _record_background(source, rec_path, 48000, 1, 80)
            chunks = []
            # 先开始录音，再把测试音写入虚拟线缆输入
            import threading

            stop = {"flag": False}

            def _consume():
                deadline = time.time() + rec_seconds + 0.5
                while time.time() < deadline and not stop["flag"]:
                    frame = reader.read(int(48000 * 0.08))
                    if frame is None:
                        break
                    chunks.append(frame)

            t = threading.Thread(target=_consume, daemon=True)
            t.start()
            time.sleep(0.25)
            result = run_engine(
                mode="passthrough",
                input_spec=str(tone),
                output_spec=routing["cable_in"],
                sample_rate=48000,
                channels=1,
                buffer_ms=80,
            )
            time.sleep(0.4)
            stop["flag"] = True
            t.join(timeout=3)
            reader.close()
            import numpy as np

            if chunks:
                data = np.concatenate(chunks, axis=0)
            else:
                data = np.zeros((0, 1), dtype=np.float32)
            save_wav(rec_path, data, 48000)
            rec_info = analyze_wav(rec_path) if rec_path.is_file() and rec_path.stat().st_size > 44 else {
                "has_sound": False,
                "error": "录音文件为空",
            }
            ok = bool(rec_info.get("has_sound"))
            add(
                "passthrough_pulse",
                ok,
                routing=routing,
                recorded=str(rec_path),
                analysis=rec_info,
                engine={
                    "rtf": result.rtf,
                    "elapsed_sec": result.elapsed_sec,
                    "frames": result.frames,
                },
                note="本环境通常没有真实麦克风；此测试把 WAV 送入程序，再经虚拟线缆录回。",
            )
        except Exception as exc:  # noqa: BLE001
            add("passthrough_pulse", False, error=str(exc))

    report_path = recorded / "selftest_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"自检时间：{report.get('time')}",
        f"总体结果：{'通过' if report.get('passed') else '未全部通过'}",
        f"报告文件：{report.get('report_path', '')}",
        "",
    ]
    for step in report.get("steps", []):
        flag = "OK" if step.get("ok") else "FAIL"
        if step.get("skipped"):
            flag = "SKIP"
        lines.append(f"[{flag}] {step.get('name')}")
        if step.get("error"):
            lines.append(f"      {step['error']}")
        if step.get("message"):
            preview = step["message"].strip().splitlines()[0]
            lines.append(f"      {preview}")
        if step.get("analysis") and isinstance(step["analysis"], dict):
            a = step["analysis"]
            if "rms" in a:
                lines.append(
                    f"      时长={a.get('duration_sec'):.3f}s  RMS={a.get('rms'):.4f}  "
                    f"峰值={a.get('peak'):.4f}  主频={a.get('dominant_freq_hz')}"
                )
        if step.get("comparison"):
            c = step["comparison"]
            rec = c.get("recorded", {})
            lines.append(
                f"      录音时长={rec.get('duration_sec'):.3f}s  有声={rec.get('has_sound')}  RMS={rec.get('rms'):.4f}"
            )
    return "\n".join(lines)
