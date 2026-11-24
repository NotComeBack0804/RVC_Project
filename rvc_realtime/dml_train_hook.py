"""确保官方 train/train.py 带有 DirectML 放置挂钩。

官方脚本本身只有 CUDA 或 CPU。本仓库的 hook 在 RVC_TRAIN_DEVICE=directml
时把模型与 batch 放到 torch-directml 的 privateuseone 设备。官方目录通常
被 gitignore，因此训练前会检查并在缺失时写入同一段挂钩。
"""
from __future__ import annotations

from pathlib import Path

from rvc_realtime.paths import third_party_rvc

MARKER = "TRAIN_DML_DEVICE = _rvc_directml_train_device()"

HOOK_BLOCK = '''import torch

from configs.config import get_training_dtype

# Hook used by the parent rvc-realtime-voice project on Windows + AMD.
# Official train.py is CUDA-or-CPU; when RVC_TRAIN_DEVICE=directml is set,
# models and batches are moved to torch-directml's privateuseone device.
# This is not a CUDA path and does nothing unless that env var is set.
def _rvc_directml_train_device():
    if os.environ.get("RVC_TRAIN_DEVICE", "").strip().lower() not in {"dml", "directml"}:
        return None
    try:
        import torch_directml

        device = torch_directml.device(torch_directml.default_device())
        probe = torch.ones(1, dtype=torch.float32).to(device)
        _ = (probe + 1).cpu()
        try:
            name = torch_directml.device_name(torch_directml.default_device())
        except Exception:
            name = str(device)
        print("训练设备：DirectML（%s / %s）" % (name, device), flush=True)
        logger.info("Training on DirectML device: %s (%s)", name, device)
        return device
    except Exception as exc:
        print("DirectML 训练设备不可用，回退 CPU：%s" % exc, flush=True)
        logger.exception("DirectML training device unavailable, falling back to CPU")
        return None


TRAIN_DML_DEVICE = _rvc_directml_train_device()


def _has_train_accelerator():
    return torch.cuda.is_available() or TRAIN_DML_DEVICE is not None


def _to_train_device(tensor, rank=0, non_blocking=False):
    if torch.cuda.is_available():
        return tensor.cuda(rank, non_blocking=non_blocking)
    if TRAIN_DML_DEVICE is not None:
        return tensor.to(TRAIN_DML_DEVICE)
    return tensor
'''

IMPORT_OLD = """import torch

from configs.config import get_training_dtype
"""

NETS_OLD = """    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
"""
NETS_NEW = """    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    elif TRAIN_DML_DEVICE is not None:
        net_g = net_g.to(TRAIN_DML_DEVICE)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    elif TRAIN_DML_DEVICE is not None:
        net_d = net_d.to(TRAIN_DML_DEVICE)
"""

CACHE_OLD = """                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
"""
CACHE_NEW = """                # Load on CUDA or DirectML
                if _has_train_accelerator():
                    phone = _to_train_device(phone, rank, True)
                    phone_lengths = _to_train_device(phone_lengths, rank, True)
                    if hps.if_f0 == 1:
                        pitch = _to_train_device(pitch, rank, True)
                        pitchf = _to_train_device(pitchf, rank, True)
                    sid = _to_train_device(sid, rank, True)
                    spec = _to_train_device(spec, rank, True)
                    spec_lengths = _to_train_device(spec_lengths, rank, True)
                    wave = _to_train_device(wave, rank, True)
                    wave_lengths = _to_train_device(wave_lengths, rank, True)
"""

STEP_OLD = """        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
"""
STEP_NEW = """        ## Load on CUDA or DirectML
        if (hps.if_cache_data_in_gpu == False) and _has_train_accelerator():
            phone = _to_train_device(phone, rank, True)
            phone_lengths = _to_train_device(phone_lengths, rank, True)
            if hps.if_f0 == 1:
                pitch = _to_train_device(pitch, rank, True)
                pitchf = _to_train_device(pitchf, rank, True)
            sid = _to_train_device(sid, rank, True)
            spec = _to_train_device(spec, rank, True)
            spec_lengths = _to_train_device(spec_lengths, rank, True)
            wave = _to_train_device(wave, rank, True)
            # wave_lengths = _to_train_device(wave_lengths, rank, True)
"""

PIN_OLD = "        pin_memory=True,\n"
PIN_NEW = "        pin_memory=torch.cuda.is_available(),\n"


def official_train_script() -> Path:
    return third_party_rvc() / "train" / "train.py"


def hook_present(text: str | None = None) -> bool:
    if text is None:
        path = official_train_script()
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8")
    return MARKER in text


def apply_official_train_hook() -> str:
    """若官方 train.py 还没有挂钩则写入。返回说明文字。"""
    path = official_train_script()
    if not path.is_file():
        return f"未找到官方训练脚本：{path}"
    text = path.read_text(encoding="utf-8")
    if hook_present(text):
        return "官方训练脚本已包含 DirectML 挂钩。"
    missing = []
    if IMPORT_OLD not in text:
        missing.append("import torch / get_training_dtype")
    if NETS_OLD not in text:
        missing.append("net_g/net_d .cuda 放置")
    if CACHE_OLD not in text:
        missing.append("cache 批次 .cuda 放置")
    if STEP_OLD not in text:
        missing.append("训练步进 .cuda 放置")
    if PIN_OLD not in text:
        missing.append("pin_memory=True")
    if missing:
        return (
            "官方 train.py 与预期不一致，无法自动加入 DirectML 挂钩（"
            + "、".join(missing)
            + "）。模型训练循环将停留在官方的 CUDA/CPU 路径；音高和特征提取仍可走 DirectML。"
        )
    text = text.replace(IMPORT_OLD, HOOK_BLOCK, 1)
    text = text.replace(PIN_OLD, PIN_NEW, 1)
    text = text.replace(NETS_OLD, NETS_NEW, 1)
    text = text.replace(CACHE_OLD, CACHE_NEW, 1)
    text = text.replace(STEP_OLD, STEP_NEW, 1)
    path.write_text(text, encoding="utf-8")
    return f"已向官方训练脚本写入 DirectML 挂钩：{path}"
