from __future__ import annotations

import subprocess
from dataclasses import dataclass
from shutil import which
from typing import Optional

from rvc_realtime.errors import AudioIOError

CABLE_IN_SINK = "RVC_Cable_In"
CABLE_OUT_SOURCE = "RVC_Cable_Out"
TEST_SRC_SINK = "RVC_TestSrc"

CABLE_IN_DESC = "RVC Cable Input"
CABLE_OUT_DESC = "RVC Cable Output"
TEST_SRC_DESC = "RVC Test Source"


@dataclass
class PulseDevice:
    index: str
    name: str
    module: str
    format: str
    state: str
    kind: str  # sink / source


def pulse_available() -> bool:
    return which("pactl") is not None


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def require_pulse() -> None:
    if not pulse_available():
        raise AudioIOError(
            "未找到 pactl，无法配置 Linux 虚拟音频设备。"
            "请安装 PulseAudio 或 PipeWire 的 Pulse 兼容层（pulseaudio-utils / pipewire-pulse）。"
        )


def list_short(kind: str) -> list[PulseDevice]:
    require_pulse()
    cmd = ["pactl", "list", "short", "sinks" if kind == "sink" else "sources"]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioIOError(result.stderr.strip() or f"pactl list {kind}s 失败")
    devices = []
    for line in result.stdout.splitlines():
        parts = line.split("	")
        if len(parts) < 2:
            continue
        devices.append(
            PulseDevice(
                index=parts[0],
                name=parts[1],
                module=parts[2] if len(parts) > 2 else "",
                format=parts[3] if len(parts) > 3 else "",
                state=parts[4] if len(parts) > 4 else "",
                kind=kind,
            )
        )
    return devices


def has_sink(name: str) -> bool:
    return any(d.name == name for d in list_short("sink"))


def has_source(name: str) -> bool:
    return any(d.name == name for d in list_short("source"))


def _module_ids_for(needle: str) -> list[str]:
    result = _run(["pactl", "list", "short", "modules"])
    if result.returncode != 0:
        return []
    ids = []
    for line in result.stdout.splitlines():
        if needle in line:
            ids.append(line.split("	", 1)[0])
    return ids


def _load_module(args: list[str]) -> str:
    result = _run(["pactl", "load-module", *args])
    if result.returncode != 0:
        raise AudioIOError(result.stderr.strip() or f"加载模块失败：{args}")
    return result.stdout.strip()


def _escape_prop(value: str) -> str:
    return value.replace(" ", r"\ ")


def _try_load_named_cables(sample_rate: int, channels: int) -> tuple[dict[str, str], Optional[str]]:
    """尝试创建命名虚拟线缆。成功返回 (created, None)；失败返回 ({}, reason)。"""
    created: dict[str, str] = {}
    try:
        if not has_sink(CABLE_IN_SINK):
            args = [
                "module-null-sink",
                f"sink_name={CABLE_IN_SINK}",
                f"rate={sample_rate}",
                "sink_properties=device.description=RVC_Cable_Input",
            ]
            if channels:
                args.insert(3, f"channels={channels}")
            created["cable_in_module"] = _load_module(args)
        if not has_source(CABLE_OUT_SOURCE) and (has_sink(CABLE_IN_SINK) or has_source(f"{CABLE_IN_SINK}.monitor")):
            created["cable_out_module"] = _load_module(
                [
                    "module-remap-source",
                    f"master={CABLE_IN_SINK}.monitor",
                    f"source_name={CABLE_OUT_SOURCE}",
                    "source_properties=device.description=RVC_Cable_Output",
                ]
            )
        if not has_sink(TEST_SRC_SINK):
            args = [
                "module-null-sink",
                f"sink_name={TEST_SRC_SINK}",
                f"rate={sample_rate}",
                "sink_properties=device.description=RVC_Test_Source",
            ]
            if channels:
                args.insert(3, f"channels={channels}")
            created["test_src_module"] = _load_module(args)
    except AudioIOError as exc:
        return created, exc.message
    return created, None


def _fallback_routing() -> Optional[dict[str, str]]:
    """当无法加载模块时，复用已有的空沉（例如本开发盒子预置的 speakers）。"""
    sinks = {d.name for d in list_short("sink")}
    sources = {d.name for d in list_short("source")}
    # 优先使用已有的命名线缆
    if CABLE_IN_SINK in sinks:
        cable_out = CABLE_OUT_SOURCE if CABLE_OUT_SOURCE in sources else f"{CABLE_IN_SINK}.monitor"
        if cable_out in sources or f"{CABLE_IN_SINK}.monitor" in sources:
            return {
                "cable_in": CABLE_IN_SINK,
                "cable_out": cable_out if cable_out in sources else f"{CABLE_IN_SINK}.monitor",
                "cable_in_monitor": f"{CABLE_IN_SINK}.monitor",
                "test_src": TEST_SRC_SINK if TEST_SRC_SINK in sinks else CABLE_IN_SINK,
                "test_src_monitor": (
                    f"{TEST_SRC_SINK}.monitor" if TEST_SRC_SINK in sinks else f"{CABLE_IN_SINK}.monitor"
                ),
                "mode": "named",
            }
    # 常见预置：speakers 空沉 + speakers.monitor
    for sink_name in ("speakers", "auto_null", "null", "dummy"):
        if sink_name in sinks and f"{sink_name}.monitor" in sources:
            return {
                "cable_in": sink_name,
                "cable_out": f"{sink_name}.monitor",
                "cable_in_monitor": f"{sink_name}.monitor",
                "test_src": sink_name,
                "test_src_monitor": f"{sink_name}.monitor",
                "mode": "fallback-existing-null-sink",
            }
    # 任意一个带 monitor 的空沉
    for sink_name in sorted(sinks):
        mon = f"{sink_name}.monitor"
        if mon in sources:
            return {
                "cable_in": sink_name,
                "cable_out": mon,
                "cable_in_monitor": mon,
                "test_src": sink_name,
                "test_src_monitor": mon,
                "mode": "fallback-existing-null-sink",
            }
    return None


def setup_virtual_audio(sample_rate: int = 48000, channels: int = 1) -> dict[str, str]:
    """创建等效于 VB-CABLE 的 Linux 虚拟线缆；若服务器禁止加载模块则回退到已有空沉。"""
    require_pulse()
    created, load_error = _try_load_named_cables(sample_rate, channels)
    if not load_error and has_sink(CABLE_IN_SINK) and (
        has_source(CABLE_OUT_SOURCE) or has_source(f"{CABLE_IN_SINK}.monitor")
    ):
        info = {
            "cable_in": CABLE_IN_SINK,
            "cable_out": CABLE_OUT_SOURCE if has_source(CABLE_OUT_SOURCE) else f"{CABLE_IN_SINK}.monitor",
            "cable_in_monitor": f"{CABLE_IN_SINK}.monitor",
            "test_src": TEST_SRC_SINK if has_sink(TEST_SRC_SINK) else CABLE_IN_SINK,
            "test_src_monitor": f"{TEST_SRC_SINK}.monitor" if has_sink(TEST_SRC_SINK) else f"{CABLE_IN_SINK}.monitor",
            "created_modules": ",".join(f"{k}={v}" for k, v in created.items()) or "already-present",
            "mode": "named",
        }
        return info

    fallback = _fallback_routing()
    if fallback:
        fallback["created_modules"] = "none"
        if load_error:
            fallback["load_module_error"] = load_error
            fallback["note"] = (
                "当前 PulseAudio 禁止运行时加载模块（常见于使用 "
                "--disallow-module-loading 的环境）。"
                f"已改用已有空沉 {fallback['cable_in']}，"
                f"其监听源 {fallback['cable_out']} 可作为虚拟麦克风。"
                "在普通 Linux 桌面上，本命令会创建 RVC_Cable_In / RVC_Cable_Out。"
            )
        return fallback

    detail = load_error or "未知原因"
    raise AudioIOError(
        "无法创建虚拟线缆，也没有找到可复用的空沉（带 .monitor 的 sink）。"
        f" pactl load-module 失败原因：{detail}"
    )


def teardown_virtual_audio() -> list[str]:
    require_pulse()
    removed = []
    needles = [CABLE_OUT_SOURCE, CABLE_IN_SINK, TEST_SRC_SINK, CABLE_IN_DESC, CABLE_OUT_DESC, TEST_SRC_DESC]
    seen = set()
    for needle in needles:
        for mid in _module_ids_for(needle):
            if mid in seen:
                continue
            seen.add(mid)
            result = _run(["pactl", "unload-module", mid])
            if result.returncode == 0:
                removed.append(mid)
    return removed


def default_cable_out() -> str:
    if has_source(CABLE_OUT_SOURCE):
        return CABLE_OUT_SOURCE
    return f"{CABLE_IN_SINK}.monitor"


def describe_routing() -> str:
    return (
        "Linux 虚拟线缆（等价于 Windows VB-CABLE）：\n"
        f"  程序输出设备：{CABLE_IN_SINK}（沉，相当于 CABLE Input）\n"
        f"  聊天软件麦克风：{CABLE_OUT_SOURCE}（源，相当于 CABLE Output）\n"
        f"  也可使用监听源：{CABLE_IN_SINK}.monitor\n"
        f"  测试音源沉：{TEST_SRC_SINK}，程序可从 {TEST_SRC_SINK}.monitor 读取\n"
    )
