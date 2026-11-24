"""选择计算设备：Windows 上优先 DirectML，其它情况回退 CPU。从不混入 CUDA。"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


ENV_COMPUTE = "RVC_COMPUTE_DEVICE"
ENV_TRAIN = "RVC_TRAIN_DEVICE"


@dataclass(frozen=True)
class ComputeInfo:
    kind: str  # "directml" | "cpu"
    torch_device: Any
    label: str
    short_label: str
    adapter_name: str
    device_str: str
    requested: str
    fallback_reason: str
    rmvpe_needs_onnx: bool
    message: str

    @property
    def is_directml(self) -> bool:
        return self.kind == "directml"


def requested_compute(override: Optional[str] = None) -> str:
    raw = (override if override is not None else os.environ.get(ENV_COMPUTE, "auto")) or "auto"
    val = raw.strip().lower()
    if val in {"dml", "directml"}:
        return "directml"
    if val in {"cpu"}:
        return "cpu"
    return "auto"


def _cpu_device():
    import torch

    return torch.device("cpu")


def _cpu_info(requested: str, reason: str = "") -> ComputeInfo:
    if reason:
        message = f"计算设备：CPU（已回退）。{reason}"
    elif requested == "cpu":
        message = "计算设备：CPU（已按 RVC_COMPUTE_DEVICE=cpu 或参数指定）。"
    elif sys.platform != "win32":
        message = (
            "计算设备：CPU。"
            "官方 RVC 的 DirectML（torch-directml / onnxruntime-directml，设备名 privateuseone:0）只在 Windows 上安装和启用；"
            "Linux 使用 CPU 是预期回退，不是故障。"
        )
        reason = "当前系统不是 Windows，官方 DirectML 仅在 Windows 上可用。"
    else:
        message = "计算设备：CPU。"
    return ComputeInfo(
        kind="cpu",
        torch_device=_cpu_device(),
        label="CPU",
        short_label="CPU",
        adapter_name="",
        device_str="cpu",
        requested=requested,
        fallback_reason=reason,
        rmvpe_needs_onnx=False,
        message=message,
    )


def probe_directml() -> tuple[bool, Any, str, str]:
    """探测 torch-directml 是否能真正在适配器上做一次张量运算。"""
    try:
        import torch
        import torch_directml
    except Exception as exc:  # noqa: BLE001
        return False, None, "", f"未安装或无法导入 torch-directml（{exc}）。"

    try:
        index = int(torch_directml.default_device())
        device = torch_directml.device(index)
        try:
            name = str(torch_directml.device_name(index))
        except Exception:
            name = str(device)
        probe = torch.ones(1, dtype=torch.float32).to(device)
        _ = (probe + 1).cpu()
        if not name:
            name = str(device)
        return True, device, name, ""
    except Exception as exc:  # noqa: BLE001
        return False, None, "", f"已安装 torch-directml，但 DirectML 适配器探测失败：{exc}。"


def probe_onnxruntime_directml() -> tuple[bool, str]:
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        return False, f"未安装 onnxruntime-directml（{exc}）。"
    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        return False, f"无法查询 onnxruntime 执行提供程序：{exc}。"
    if "DmlExecutionProvider" not in providers:
        return (
            False,
            "当前 onnxruntime 不含 DmlExecutionProvider"
            f"（providers={providers}）。Windows 上请安装官方依赖里的 onnxruntime-directml。",
        )
    return True, ""


def select_compute(override: Optional[str] = None) -> ComputeInfo:
    """按官方 RVC 规则选择设备：无可用 CUDA 时，Windows 使用 DirectML，否则 CPU。本项目不安装、不选择 CUDA。"""
    requested = requested_compute(override)
    if requested == "cpu":
        return _cpu_info("cpu")

    want_dml = requested == "directml" or sys.platform == "win32"
    if not want_dml:
        return _cpu_info(requested)

    ok, device, name, err = probe_directml()
    if not ok:
        hint = (
            "请确认已用 .\\scripts\\install_windows.ps1 -WithRvcCpu 安装官方 requirments_cpu_py312.txt，"
            "并且本机已安装支持 DirectX 12 的 AMD 显卡驱动。"
        )
        return _cpu_info(requested, (err + " " + hint).strip())

    ort_ok, ort_err = probe_onnxruntime_directml()
    if not ort_ok:
        hint = "音高提取 RMVPE 在 DirectML 上必须走 rmvpe.onnx + DmlExecutionProvider，因此整体回退 CPU。"
        return _cpu_info(requested, f"{ort_err} {hint}".strip())

    label = f"DirectML（{name}，{device}）" if name else f"DirectML（{device}）"
    message = (
        f"计算设备：{label}。"
        "推理与训练将默认使用显卡；音高提取 RMVPE 使用 rmvpe.onnx 和 onnxruntime-directml。"
        "可用 RVC_COMPUTE_DEVICE=cpu 强制回退 CPU。"
    )
    return ComputeInfo(
        kind="directml",
        torch_device=device,
        label=label,
        short_label="DirectML",
        adapter_name=name,
        device_str=str(device),
        requested=requested,
        fallback_reason="",
        rmvpe_needs_onnx=True,
        message=message,
    )


def format_compute_status(info: Optional[ComputeInfo] = None) -> str:
    info = info or select_compute()
    lines = [info.message]
    lines.append(f"  类型：{info.short_label}")
    lines.append(f"  设备名：{info.device_str}")
    if info.adapter_name:
        lines.append(f"  适配器：{info.adapter_name}")
    if info.fallback_reason:
        lines.append(f"  回退原因：{info.fallback_reason}")
    lines.append(f"  RMVPE 需要 rmvpe.onnx：{'是' if info.rmvpe_needs_onnx else '否（使用 rmvpe.pt）'}")
    return "\n".join(lines)


def apply_compute_override(value: Optional[str]) -> None:
    if not value:
        return
    os.environ[ENV_COMPUTE] = requested_compute(value)


def configure_training_env(env: dict, info: Optional[ComputeInfo] = None) -> dict:
    """给官方训练子进程设置设备环境变量。不改变 Linux CPU 路径。"""
    info = info or select_compute()
    env[ENV_COMPUTE] = info.kind if info.requested == "auto" else info.requested
    if info.is_directml:
        env[ENV_TRAIN] = "directml"
    else:
        env.pop(ENV_TRAIN, None)
        env[ENV_COMPUTE] = "cpu"
    return env
