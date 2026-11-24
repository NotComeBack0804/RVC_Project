from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from rvc_realtime.audio_io import match_channels, resample
from rvc_realtime.loudness import GainSmoother, apply_chunk_loudness
from rvc_realtime.rvc_infer_safe import IndexStats, install_safe_infer
from rvc_realtime.errors import (
    BaseModelNotFoundError,
    MissingDependencyError,
    ModelNotFoundError,
)
from rvc_realtime.compute import ComputeInfo, select_compute
from rvc_realtime.paths import (
    list_user_indices,
    list_user_models,
    missing_hubert,
    missing_rmvpe_onnx,
    missing_rmvpe_pt,
    project_root,
    third_party_rvc,
    weights_dir,
)


@dataclass
class RuntimeStatus:
    torch_ok: bool
    torch_version: str
    torch_cuda: bool
    compute: str
    compute_device: str
    rvc_ok: bool
    rvc_path: str
    message: str


def probe_runtime() -> RuntimeStatus:
    torch_ok = False
    torch_version = ""
    torch_cuda = False
    rvc_ok = False
    rvc_path = str(third_party_rvc())
    notes = []
    compute_label = "未知"
    compute_device = "cpu"
    try:
        import torch

        torch_ok = True
        torch_version = str(torch.__version__)
        torch_cuda = bool(torch.cuda.is_available())
        compute = select_compute()
        compute_label = compute.label
        compute_device = compute.device_str
        notes.append(compute.message)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"未安装 torch（{exc}）。完整推理环境尚未就绪。")

    rvc_root = third_party_rvc()
    if not (rvc_root / "infer" / "rtrvc.py").is_file():
        notes.append(f"未找到官方 RVC 仓库：{rvc_root}")
    elif torch_ok:
        rvc_ok = True
    else:
        notes.append("已找到官方 RVC 源码，但缺少完整推理依赖。")

    if torch_cuda:
        notes.append("检测到 CUDA 可用，但本项目的 Windows/Linux 安装路径不使用 CUDA，仍按 DirectML 或 CPU 选择设备。")

    return RuntimeStatus(
        torch_ok=torch_ok,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        compute=compute_label,
        compute_device=compute_device,
        rvc_ok=rvc_ok,
        rvc_path=rvc_path,
        message="；".join(notes) if notes else "RVC 推理运行时可用。",
    )


def resolve_model(model: Optional[str]) -> Optional[Path]:
    if model:
        path = Path(model).expanduser()
        if not path.is_absolute():
            candidate = project_root() / path
            path = candidate if candidate.exists() else path
        if path.is_file():
            return path
        return None
    models = list_user_models()
    if len(models) == 1:
        return models[0]
    return None


def resolve_index(index: Optional[str], model: Optional[Path] = None) -> Optional[Path]:
    if index:
        path = Path(index).expanduser()
        if not path.is_absolute():
            candidate = project_root() / path
            path = candidate if candidate.exists() else path
        return path if path.is_file() else None
    if model is not None:
        sibling = model.with_suffix(".index")
        if sibling.is_file():
            return sibling
        same_name = Path(str(model) + ".index")
        if same_name.is_file():
            return same_name
        # assets/indices/<stem>.index
        from rvc_realtime.paths import indices_dir

        alt = indices_dir() / (model.stem + ".index")
        if alt.is_file():
            return alt
    indices = list_user_indices()
    if len(indices) == 1:
        return indices[0]
    return None



@contextmanager
def official_rvc_cwd():
    """官方推理代码使用相对路径（如 ./i18n、assets/rmvpe/rmvpe.pt），需在其仓库根目录执行。"""
    import os

    root = third_party_rvc()
    prev = Path.cwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


class PassthroughProcessor:
    name = "passthrough"

    def __init__(self, channels: int):
        self.channels = channels

    def process(self, frames: np.ndarray) -> np.ndarray:
        return match_channels(frames, self.channels)

    def close(self) -> None:
        return


class OfficialRvcProcessor:
    """加载标准 RVC .pth，并按块做语音转换。"""

    name = "convert"

    def __init__(
        self,
        model_path: Path,
        index_path: Optional[Path],
        sample_rate: int,
        channels: int,
        f0_up_key: int = 0,
        formant: float = 0.0,
        index_rate: float = 0.0,
        f0_method: str = "rmvpe",
        extra_sec: float = 0.8,
        loudness_mode: str = "match-input",
        gain_db: float = 0.0,
        peak_limit: float = 0.95,
    ):
        import sys
        import torch

        rvc_root = third_party_rvc()
        root_s = str(rvc_root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

        model_path = model_path.resolve()
        if index_path is not None:
            index_path = index_path.resolve()

        compute = select_compute()
        class InferConfig:
            def __init__(self, info: ComputeInfo):
                self.device = info.torch_device
                self.is_half = False  # 官方 DirectML 路径使用 fp32

        self.compute = compute
        self.config = InferConfig(compute)
        print(compute.message, flush=True)
        self.sample_rate = sample_rate
        self.channels = channels
        self.f0_method = f0_method
        with official_rvc_cwd():
            from infer import rtrvc as rvc_for_realtime

            self.rvc = rvc_for_realtime.RVC(
                f0_up_key,
                formant,
                str(model_path),
                str(index_path) if index_path else "",
                index_rate if index_path else 0.0,
                self.config,
                None,
            )
        if getattr(self.rvc, "net_g", None) is None:
            raise MissingDependencyError(
                f"RVC 模型未能成功加载：{model_path}。请确认这是已授权的标准 RVC .pth 文件。"
            )
        self.tgt_sr = int(getattr(self.rvc, "tgt_sr", 40000))
        self.extra_sec = extra_sec
        extra = int(extra_sec * sample_rate)
        self.context = np.zeros((extra, 1), dtype=np.float32)
        self.loudness_mode = loudness_mode
        self.extra_gain = float(10 ** (gain_db / 20.0))
        self.peak_limit = peak_limit
        self.gain_smoother = GainSmoother()
        self.last_gain = 1.0
        self.index_stats = IndexStats()
        install_safe_infer(self.rvc, self.index_stats)

    def process(self, frames: np.ndarray) -> np.ndarray:
        import torch

        mono = match_channels(frames, 1)
        # 在设备采样率上累积上下文，再重采样到 16 kHz 供官方实时推理使用。
        self.context = np.concatenate([self.context, mono], axis=0)
        keep = int((self.extra_sec + len(mono) / self.sample_rate) * self.sample_rate) + 16
        if self.context.shape[0] > keep:
            self.context = self.context[-keep:]
        x_16k = resample(self.context, self.sample_rate, 16000).reshape(-1)
        input_wav = torch.from_numpy(x_16k).float()
        block_frame_16k = max(160, int(round(mono.shape[0] * 16000 / self.sample_rate)))
        # skip_head / return_length 以 10ms  hop（tgt_sr/100，对应 16k 的 160）计。
        total_hop = max(1, input_wav.shape[0] // 160)
        return_length = max(1, int(round((mono.shape[0] / self.sample_rate) * 100)))
        skip_head = max(0, total_hop - return_length)
        with official_rvc_cwd():
            inferred = self.rvc.infer(
                input_wav,
                block_frame_16k,
                skip_head,
                return_length,
                self.f0_method,
            )
        if hasattr(inferred, "detach"):
            inferred = inferred.detach().cpu().numpy()
        inferred = np.asarray(inferred, dtype=np.float32).reshape(-1, 1)
        out = resample(inferred, self.tgt_sr, self.sample_rate)
        # 取末尾与输入块等长的部分，避免上下文把输出拉长。
        if out.shape[0] >= mono.shape[0]:
            out = out[-mono.shape[0] :]
        elif out.shape[0] < mono.shape[0]:
            pad = np.zeros((mono.shape[0] - out.shape[0], 1), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        out, gain = apply_chunk_loudness(
            out,
            mono,
            mode=self.loudness_mode,
            extra_gain=self.extra_gain,
            peak_limit=self.peak_limit,
            smoother=self.gain_smoother,
        )
        self.last_gain = gain
        return match_channels(out, self.channels)

    def close(self) -> None:
        print(self.index_stats.summary(), flush=True)
        return


def _missing_model_message() -> str:
    existing = list_user_models()
    extra = ""
    if existing:
        extra = "在 assets/weights/ 中找到多个模型，请用 --model 明确指定其一：\n" + "\n".join(
            f"  - {p}" for p in existing
        )
    else:
        extra = f"当前 {weights_dir()} 中没有 .pth 文件。"
    return (
        "缺少已授权的 RVC 音色模型，变声路径已安全退出（未崩溃，也未加载任何他人音色）。\n"
        "实时变声只能用于你本人的声音，或已获得明确授权的声音。\n"
        "请将授权后训练得到的 .pth 放到 assets/weights/，可选的检索索引放到 assets/indices/，或使用 --model / --index。\n"
        f"{extra}\n"
        "若只想验证音频链路，请改用直通模式：\n"
        "  python -m rvc_realtime run --mode passthrough ..."
    )


def _missing_base_message() -> str:
    parts = []
    hubert = missing_hubert()
    if hubert:
        parts.append("缺少 HuBERT 基础模型：" + ", ".join(hubert))
    if missing_rmvpe_pt():
        parts.append("缺少 RMVPE 音高模型：assets/rmvpe/rmvpe.pt")
    compute = select_compute()
    if compute.rmvpe_needs_onnx and missing_rmvpe_onnx():
        parts.append(
            "缺少 RMVPE ONNX 模型：assets/rmvpe/rmvpe.onnx。"
            "Windows DirectML 音高提取必须使用该文件（官方路径：rmvpe.pt 同目录下的 rmvpe.onnx，"
            "由 onnxruntime-directml 的 DmlExecutionProvider 执行）。"
        )
    if compute.rmvpe_needs_onnx:
        parts.append(
            r"请运行：.\scripts\download_base_models.ps1   （Windows 会默认下载 rmvpe.onnx）"
        )
    else:
        parts.append(r"请运行：./scripts/download_base_models.sh  或  .\scripts\download_base_models.ps1")
    return "\n".join(parts)



def build_processor(
    mode: str,
    model: Optional[str],
    index: Optional[str],
    sample_rate: int,
    channels: int,
    f0_up_key: int = 0,
    formant: float = 0.0,
    index_rate: float = 0.5,
    f0_method: str = "rmvpe",
    loudness_mode: str = "match-input",
    gain_db: float = 0.0,
    peak_limit: float = 0.95,
):
    if mode == "passthrough":
        return PassthroughProcessor(channels=channels)
    if mode != "convert":
        raise ModelNotFoundError(f"未知模式：{mode}（可选 passthrough / convert）")

    model_path = resolve_model(model)
    if model_path is None:
        raise ModelNotFoundError(_missing_model_message())

    missing_base = []
    if missing_hubert():
        missing_base.append("hubert")
    if missing_rmvpe_pt() and f0_method == "rmvpe":
        missing_base.append("rmvpe")
    compute = select_compute()
    if compute.rmvpe_needs_onnx and missing_rmvpe_onnx() and f0_method == "rmvpe":
        missing_base.append("rmvpe.onnx")
    if missing_base:
        raise BaseModelNotFoundError(_missing_base_message())

    runtime = probe_runtime()
    if not runtime.rvc_ok:
        raise MissingDependencyError(
            "已找到授权音色模型，但完整 RVC 推理环境尚未就绪，因此不会开始变声。\n"
            f"{runtime.message}\n"
            "请在独立环境中安装官方 CPU/DirectML 依赖（不要混入 CUDA）：\n"
            "  Linux:   ./scripts/install_linux.sh --with-rvc-cpu\n"
            "  Windows: .\\scripts\\install_windows.ps1 -WithRvcCpu\n"
            "Windows 上该依赖会安装 torch-directml 与 onnxruntime-directml，默认走显卡。\n"
            f"模型路径：{model_path}"
        )

    index_path = resolve_index(index, model_path)
    return OfficialRvcProcessor(
        model_path=model_path,
        index_path=index_path,
        sample_rate=sample_rate,
        channels=channels,
        f0_up_key=f0_up_key,
        formant=formant,
        index_rate=index_rate if index_path else 0.0,
        f0_method=f0_method,
        loudness_mode=loudness_mode,
        gain_db=gain_db,
        peak_limit=peak_limit,
    )
