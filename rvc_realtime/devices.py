from __future__ import annotations

from dataclasses import dataclass, field
from shutil import which
from typing import Optional

from rvc_realtime import pulse
from rvc_realtime.errors import AudioIOError


@dataclass
class DeviceInfo:
    kind: str  # input / output
    name: str
    backend: str
    extra: str = ""


@dataclass
class CaptureChoice:
    """可供录音的设备。主路径是扬声器环回，麦克风只作为次要选项。"""

    label: str
    name: str
    backend: str  # pulse / sounddevice
    category: str  # loopback / mix / mic / other
    wasapi_loopback: bool = False
    device_index: Optional[int] = None
    is_cable: bool = False
    is_default_output: bool = False
    extra: str = ""


# 环回 / 系统音频关键词（Windows 立体声混音、WASAPI loopback 文案、Linux sink monitor）
_LOOPBACK_HINTS = (
    "loopback",
    "stereo mix",
    "stereomix",
    "立体声混音",
    "立体声混合",
    "what u hear",
    "wave out mix",
    "waveout mix",
    "系统音频",
    "system audio",
    "环回",
    "monitor of",
)
_CABLE_HINTS = ("cable input", "cable output", "vb-audio", "vb cable", "rvc_cable", "rvc cable")
_MIC_HINTS = ("microphone", "麦克风", "headset mic", "mic ", " mic", "micphone", "array")
_SPEAKER_HINTS = ("speakers", "speaker", "扬声器", "headphones", "耳机", "headphone", "realtek", "output")


def _norm(name: str) -> str:
    return (name or "").strip().lower().replace("（", "(").replace("）", ")")


def is_cable_name(name: str) -> bool:
    n = _norm(name)
    return any(h in n for h in _CABLE_HINTS) or n.startswith("cable ") or "rvc_cable" in n


def is_monitor_name(name: str) -> bool:
    n = _norm(name)
    return n.endswith(".monitor") or ".monitor" in n or "monitor of" in n


def is_loopback_name(name: str) -> bool:
    n = _norm(name)
    if is_monitor_name(name):
        return True
    return any(h in n for h in _LOOPBACK_HINTS)


def is_mic_name(name: str) -> bool:
    n = _norm(name)
    if is_loopback_name(name) or is_cable_name(name):
        return False
    return any(h in n for h in _MIC_HINTS)


def list_sounddevice() -> tuple[list[DeviceInfo], Optional[str]]:
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        return [], f"sounddevice 不可用（{exc}）。Linux 将改用 PulseAudio；Windows 请安装 PortAudio 后重试。"
    try:
        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        return [], f"sounddevice 无法枚举设备：{exc}"
    result: list[DeviceInfo] = []
    for index, dev in enumerate(devices):
        name = str(dev.get("name", f"#{index}"))
        host = ""
        try:
            host = sd.query_hostapis()[int(dev["hostapi"])]["name"]
        except Exception:
            host = str(dev.get("hostapi", ""))
        extra = (
            f"index={index}, hostapi={host}, in={dev.get('max_input_channels')}, "
            f"out={dev.get('max_output_channels')}, sr={dev.get('default_samplerate')}"
        )
        if int(dev.get("max_input_channels") or 0) > 0:
            result.append(DeviceInfo("input", name, "sounddevice", extra))
        if int(dev.get("max_output_channels") or 0) > 0:
            result.append(DeviceInfo("output", name, "sounddevice", extra))
    return result, None


def list_pulse() -> tuple[list[DeviceInfo], Optional[str]]:
    if not pulse.pulse_available():
        return [], "未找到 pactl，跳过 PulseAudio 设备枚举。"
    result: list[DeviceInfo] = []
    try:
        for src in pulse.list_short("source"):
            result.append(DeviceInfo("input", src.name, "pulse", f"index={src.index}, {src.format}, {src.state}"))
        for sink in pulse.list_short("sink"):
            result.append(DeviceInfo("output", sink.name, "pulse", f"index={sink.index}, {sink.format}, {sink.state}"))
    except AudioIOError as exc:
        return [], exc.message
    return result, None


def list_all_devices() -> tuple[list[DeviceInfo], list[str]]:
    notes: list[str] = []
    devices: list[DeviceInfo] = []
    sd_devs, sd_note = list_sounddevice()
    if sd_note:
        notes.append(sd_note)
    devices.extend(sd_devs)
    pulse_devs, pulse_note = list_pulse()
    if pulse_note:
        notes.append(pulse_note)
    devices.extend(pulse_devs)
    if not devices and not notes:
        notes.append("没有发现可用音频设备。")
    return devices, notes


def _host_name(sd, dev) -> str:
    try:
        return str(sd.query_hostapis()[int(dev["hostapi"])]["name"])
    except Exception:
        return str(dev.get("hostapi", ""))


def _default_output_name() -> Optional[str]:
    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="output")
        return str(dev.get("name") or "") or None
    except Exception:
        return None


def list_capture_choices() -> list[CaptureChoice]:
    """录音设备列表：扬声器环回 / 系统音频在前，麦克风在后。"""
    choices: list[CaptureChoice] = []
    seen: set[tuple[str, str, bool]] = set()

    def add(choice: CaptureChoice) -> None:
        key = (choice.backend, choice.name, choice.wasapi_loopback)
        if key in seen:
            return
        seen.add(key)
        choices.append(choice)

    # Linux Pulse：sink 的 .monitor 就是正在播放的系统声音
    if pulse.pulse_available():
        try:
            for src in pulse.list_short("source"):
                name = src.name
                if is_monitor_name(name):
                    add(
                        CaptureChoice(
                            label=f"扬声器环回 · {name}",
                            name=name,
                            backend="pulse",
                            category="loopback",
                            is_cable=is_cable_name(name),
                            is_default_output=name in {"speakers.monitor", "RVC_Cable_In.monitor"},
                            extra=f"pulse source {src.index}",
                        )
                    )
                else:
                    add(
                        CaptureChoice(
                            label=f"麦克风 · {name}",
                            name=name,
                            backend="pulse",
                            category="mic",
                            is_cable=is_cable_name(name),
                            extra=f"pulse source {src.index}",
                        )
                    )
        except AudioIOError:
            pass

    try:
        import sounddevice as sd

        devices = list(sd.query_devices())
        default_out = _default_output_name()
        for index, dev in enumerate(devices):
            name = str(dev.get("name", f"#{index}"))
            host = _host_name(sd, dev)
            in_ch = int(dev.get("max_input_channels") or 0)
            out_ch = int(dev.get("max_output_channels") or 0)
            wasapi = "wasapi" in host.lower()
            # Windows：把 WASAPI 播放设备做成 loopback 输入
            if wasapi and out_ch > 0:
                add(
                    CaptureChoice(
                        label=f"扬声器环回 · {name}",
                        name=name,
                        backend="sounddevice",
                        category="loopback",
                        wasapi_loopback=True,
                        device_index=index,
                        is_cable=is_cable_name(name),
                        is_default_output=bool(default_out) and name == default_out,
                        extra=f"WASAPI loopback, index={index}, out={out_ch}",
                    )
                )
            if in_ch <= 0:
                continue
            if is_loopback_name(name):
                add(
                    CaptureChoice(
                        label=f"系统音频 · {name}",
                        name=name,
                        backend="sounddevice",
                        category="mix",
                        device_index=index,
                        is_cable=is_cable_name(name),
                        extra=f"index={index}, hostapi={host}",
                    )
                )
            else:
                cat = "mic" if is_mic_name(name) or not is_cable_name(name) else "other"
                prefix = "麦克风 · " if cat == "mic" else "输入 · "
                add(
                    CaptureChoice(
                        label=f"{prefix}{name}",
                        name=name,
                        backend="sounddevice",
                        category=cat,
                        device_index=index,
                        is_cable=is_cable_name(name),
                        extra=f"index={index}, hostapi={host}",
                    )
                )
    except Exception:
        pass

    return _sort_capture(choices)


def _sort_capture(choices: list[CaptureChoice]) -> list[CaptureChoice]:
    def score(c: CaptureChoice) -> tuple:
        # 数字越小越靠前。CABLE 环回不要当默认，以免把变声结果再录进去。
        if c.category in {"loopback", "mix"} and not c.is_cable:
            if c.name in {"speakers.monitor"} or c.is_default_output:
                group = 0
            elif "speakers" in _norm(c.name) or "扬声器" in c.name:
                group = 1
            else:
                group = 2
        elif c.category in {"loopback", "mix"} and c.is_cable:
            group = 5
        elif c.category == "mic" and not c.is_cable:
            group = 6
        else:
            group = 7
        return (group, c.label)

    return sorted(choices, key=score)


def default_loopback_label(choices: Optional[list[CaptureChoice]] = None) -> Optional[str]:
    items = choices if choices is not None else list_capture_choices()
    for c in items:
        if c.category in {"loopback", "mix"} and not c.is_cable:
            return c.label
    return None


def loopback_labels(choices: Optional[list[CaptureChoice]] = None) -> list[str]:
    items = choices if choices is not None else list_capture_choices()
    labels = [c.label for c in items if c.category in {"loopback", "mix"}]
    return labels or ["（未找到扬声器环回，请导入已有 WAV）"]


def mic_labels(choices: Optional[list[CaptureChoice]] = None) -> list[str]:
    items = choices if choices is not None else list_capture_choices()
    labels = [c.label for c in items if c.category == "mic" and not c.is_cable]
    return labels or ["（未找到麦克风）"]


def all_capture_labels(choices: Optional[list[CaptureChoice]] = None) -> list[str]:
    items = choices if choices is not None else list_capture_choices()
    return [c.label for c in items] or ["（未找到录音设备，请导入已有 WAV）"]


def resolve_capture(label_or_name: str, choices: Optional[list[CaptureChoice]] = None) -> Optional[CaptureChoice]:
    if not label_or_name or label_or_name.startswith("（"):
        return None
    items = choices if choices is not None else list_capture_choices()
    for c in items:
        if c.label == label_or_name or c.name == label_or_name:
            return c
    # 容忍只写了设备原名
    for c in items:
        if c.name in label_or_name or label_or_name.endswith(c.name):
            return c
    # 退化：把字符串当成 Pulse / 设备名
    if label_or_name in {"speakers.monitor"} or is_monitor_name(label_or_name) or is_loopback_name(label_or_name):
        return CaptureChoice(
            label=label_or_name,
            name=label_or_name,
            backend="pulse" if pulse.pulse_available() else "sounddevice",
            category="loopback",
        )
    return CaptureChoice(
        label=label_or_name,
        name=label_or_name,
        backend="pulse" if pulse.pulse_available() else "sounddevice",
        category="other",
    )


def preferred_output_names(outputs: list[str]) -> Optional[str]:
    prefer = [
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE Input",
        "RVC_Cable_In",
        "speakers",
    ]
    lower = {n.lower(): n for n in outputs}
    for name in prefer:
        if name in outputs:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    for n in outputs:
        if "cable input" in n.lower():
            return n
    return outputs[0] if outputs else None


def preferred_mic_names(inputs: list[str]) -> Optional[str]:
    for n in inputs:
        if is_loopback_name(n) or is_monitor_name(n) or is_cable_name(n):
            continue
        if n.startswith("（"):
            continue
        return n
    return inputs[0] if inputs and not inputs[0].startswith("（") else None


def format_device_list(devices: list[DeviceInfo], notes: list[str]) -> str:
    lines = ["可用音频设备："]
    if not devices:
        lines.append("  （空）")
    for dev in devices:
        kind = "输入" if dev.kind == "input" else "输出"
        extra = f"  [{dev.extra}]" if dev.extra else ""
        lines.append(f"  - [{kind}/{dev.backend}] {dev.name}{extra}")
    captures = list_capture_choices()
    if captures:
        lines.append("录音主路径（扬声器 / 系统音频 / 环回）优先：")
        for c in captures:
            if c.category in {"loopback", "mix"}:
                lines.append(f"  - {c.label}")
    if notes:
        lines.append("说明：")
        for note in notes:
            lines.append(f"  - {note}")
    if which("pactl"):
        lines.append("")
        lines.append(pulse.describe_routing())
    lines.append("录音请优先选择扬声器环回（Windows：WASAPI 环回或立体声混音；Linux：speakers.monitor）。")
    lines.append("Windows 上实时变声路由通常为：")
    lines.append("  - 程序输出：CABLE Input")
    lines.append("  - 聊天/游戏麦克风：CABLE Output")
    lines.append("录音时先不要开变声输出，也不要从 CABLE 录，以免把变声结果再录进去。")
    return "\n".join(lines)
