from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def assets_dir() -> Path:
    return project_root() / "assets"


def weights_dir() -> Path:
    return assets_dir() / "weights"


def indices_dir() -> Path:
    return assets_dir() / "indices"


def hubert_dir() -> Path:
    return assets_dir() / "hubert_base"


def rmvpe_dir() -> Path:
    return assets_dir() / "rmvpe"


def testdata_dir() -> Path:
    return project_root() / "testdata"


def third_party_rvc() -> Path:
    return project_root() / "third_party" / "Retrieval-based-Voice-Conversion-WebUI"


def official_requirements_cpu() -> Path:
    return third_party_rvc() / "requirments_cpu_py312.txt"


HUBERT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
)
RMVPE_PT = "rmvpe.pt"
RMVPE_ONNX = "rmvpe.onnx"


def list_user_models() -> list[Path]:
    if not weights_dir().is_dir():
        return []
    return sorted(p for p in weights_dir().glob("*.pth") if p.is_file())


def list_user_indices() -> list[Path]:
    if not indices_dir().is_dir():
        return []
    return sorted(p for p in indices_dir().glob("*.index") if p.is_file())


def hubert_status() -> dict[str, bool]:
    d = hubert_dir()
    return {name: (d / name).is_file() and (d / name).stat().st_size > 0 for name in HUBERT_FILES}


def rmvpe_status() -> dict[str, bool]:
    d = rmvpe_dir()
    return {
        RMVPE_PT: (d / RMVPE_PT).is_file() and (d / RMVPE_PT).stat().st_size > 0,
        RMVPE_ONNX: (d / RMVPE_ONNX).is_file() and (d / RMVPE_ONNX).stat().st_size > 0,
    }


def missing_hubert() -> list[str]:
    return [name for name, ok in hubert_status().items() if not ok]


def missing_rmvpe_pt() -> bool:
    return not rmvpe_status()[RMVPE_PT]


def missing_rmvpe_onnx() -> bool:
    return not rmvpe_status()[RMVPE_ONNX]
