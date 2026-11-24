"""调用官方 RVC 训练脚本：切分 → 提取 → 训练 → 导出 .pth / 索引。"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from rvc_realtime.compute import configure_training_env, select_compute
from rvc_realtime.dml_train_hook import apply_official_train_hook
from rvc_realtime.errors import AppError, MissingDependencyError, UsageError
from rvc_realtime.paths import (
    assets_dir,
    indices_dir,
    missing_rmvpe_onnx,
    project_root,
    third_party_rvc,
    weights_dir,
)

ProgressCb = Callable[[str], None]

_current_proc: subprocess.Popen | None = None
_stop_flag = threading.Event()


class TrainingError(AppError):
    exit_code = 1


def request_stop() -> None:
    _stop_flag.set()
    proc = _current_proc
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass


def _python() -> str:
    return sys.executable


def _notify(cb: Optional[ProgressCb], message: str) -> None:
    print(message, flush=True)
    if cb:
        cb(message)


def _official_args(script_rel: str, script_args: list[str]) -> list[str]:
    return [_python(), "-m", "rvc_realtime.run_official", script_rel, *script_args]


def _run_step(cwd: Path, args: list[str], log_path: Path, title: str, cb: Optional[ProgressCb]) -> None:
    global _current_proc
    if _stop_flag.is_set():
        raise TrainingError("训练已被中断。")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _notify(cb, f"开始：{title}\n命令：{' '.join(args)}")
    env = os.environ.copy()
    env["RVC_CUDA_GRAPH"] = "0"
    env["RVC_AUDIO_FORCE_CPU"] = "1"
    configure_training_env(env)
    root = str(project_root())
    env["PYTHONPATH"] = os.pathsep.join([root, str(cwd)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    with open(log_path, "a", encoding="utf8") as logf:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        _current_proc = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            logf.write(line + "\n")
            logf.flush()
            if line:
                _notify(cb, line)
        code = proc.wait()
        _current_proc = None
        if _stop_flag.is_set():
            raise TrainingError("训练已被中断。")
        if code != 0:
            raise TrainingError(f"{title} 失败，退出码 {code}。日志：{log_path}")
    _notify(cb, f"完成：{title}")


def _ensure_training_assets(version: str, sr: str) -> tuple[str, str]:
    rvc = third_party_rvc()
    path_str = "" if version == "v1" else "_v2"
    g = rvc / "assets" / f"pretrained{path_str}" / f"f0G{sr}.pth"
    d = rvc / "assets" / f"pretrained{path_str}" / f"f0D{sr}.pth"
    # fallback to project assets
    if not g.is_file():
        g = assets_dir() / f"pretrained{path_str}" / f"f0G{sr}.pth"
    if not d.is_file():
        d = assets_dir() / f"pretrained{path_str}" / f"f0D{sr}.pth"
    mute = rvc / "logs" / "mute" / "0_gt_wavs" / f"mute{sr}.wav"
    missing = []
    if not g.is_file():
        missing.append(str(g))
    if not d.is_file():
        missing.append(str(d))
    if not mute.is_file():
        missing.append(str(mute))
    if missing:
        raise MissingDependencyError(
            "训练所需的官方底模或 mute 样本缺失：\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n请运行：./scripts/download_base_models.sh --with-pretrained"
        )
    return str(g), str(d)



def _reset_exp_features(exp_dir: Path) -> None:
    for name in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature256", "3_feature768"):
        d = exp_dir / name
        if d.is_dir():
            shutil.rmtree(d)


def _stage_audio_dir(src_files: list[Path], exp_dir: Path) -> Path:
    """只把音频放进训练输入目录，避免官方预处理把 README 等文件当音频。"""
    staged = exp_dir / "_authorized_input"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        dest = staged / src.name
        if dest.exists():
            dest.unlink()
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)
    return staged


def _copy_config(rvc: Path, exp_dir: Path, version: str, sr: str) -> None:
    # 官方逻辑：40k 使用 v1/40k.json；其他采样率按版本选择
    if version == "v1" or sr == "40k":
        rel = f"v1/{sr}.json"
    else:
        rel = f"v2/{sr}.json"
    src = rvc / "configs" / rel
    if not src.is_file():
        raise TrainingError(f"找不到训练配置：{src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    dest = exp_dir / "config.json"
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _build_filelist(rvc: Path, exp_dir: Path, version: str, sr: str, use_f0: bool = True) -> int:
    gt = exp_dir / "0_gt_wavs"
    feat = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    f0_dir = exp_dir / "2a_f0"
    f0nsf = exp_dir / "2b-f0nsf"
    names = {p.stem for p in gt.glob("*.wav")} & {p.stem for p in feat.glob("*.npy")}
    if use_f0:
        names &= {p.name.replace(".wav.npy", "") for p in f0_dir.glob("*.wav.npy")}
        names &= {p.name.replace(".wav.npy", "") for p in f0nsf.glob("*.wav.npy")}
    if not names:
        raise TrainingError("没有可用于训练的有效音频，请先确认切分和特征提取都已完成。")
    fea_dim = 256 if version == "v1" else 768
    lines = []
    for name in sorted(names):
        if use_f0:
            lines.append(
                f"{gt}/{name}.wav|{feat}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf}/{name}.wav.npy|0"
            )
        else:
            lines.append(f"{gt}/{name}.wav|{feat}/{name}.npy|0")
    mute_root = rvc / "logs" / "mute"
    if use_f0:
        mute_line = (
            f"{mute_root}/0_gt_wavs/mute{sr}.wav|{mute_root}/3_feature{fea_dim}/mute.npy|"
            f"{mute_root}/2a_f0/mute.wav.npy|{mute_root}/2b-f0nsf/mute.wav.npy|0"
        )
    else:
        mute_line = f"{mute_root}/0_gt_wavs/mute{sr}.wav|{mute_root}/3_feature{fea_dim}/mute.npy|0"
    lines.extend([mute_line, mute_line])
    (exp_dir / "filelist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(names)


def estimate_message(audio_dir: Path, epochs: int) -> str:
    wavs = list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.flac"))
    compute = select_compute()
    return (
        f"将使用 {len(wavs)} 个音频文件，训练 {epochs} 轮。"
        f"{compute.message}"
        "请保持授权确认，可以随时中断。"
    )


def run_training(
    audio_dir: str,
    name: str,
    authorized: bool,
    version: str = "v2",
    sample_rate: str = "40k",
    total_epoch: int = 200,
    batch_size: int = 1,
    f0_method: str = "rmvpe",
    slice_sec: float = 3.7,
    progress: Optional[ProgressCb] = None,
) -> dict:
    _stop_flag.clear()
    if not authorized:
        raise UsageError("训练被拒绝：请先确认录音是本人声音或已获得明确授权（--i-have-authorization 或界面勾选）。")
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_")
    if not safe:
        raise UsageError("模型名只能包含字母、数字、下划线和短横线。")
    src = Path(audio_dir).expanduser().resolve()
    if not src.is_dir():
        raise UsageError(f"找不到录音目录：{src}")
    audio_files = [p for p in src.iterdir() if p.suffix.lower() in {".wav", ".flac", ".ogg"}]
    if not audio_files:
        raise UsageError(f"{src} 中没有 WAV/FLAC 录音。请先录音或导入授权音频。")

    rvc = third_party_rvc()
    pre_g, pre_d = _ensure_training_assets(version, sample_rate)
    sr_hz = {"32k": 32000, "40k": 40000, "48k": 48000}[sample_rate]
    exp_dir = rvc / "logs" / safe
    exp_dir.mkdir(parents=True, exist_ok=True)
    _reset_exp_features(exp_dir)
    staged = _stage_audio_dir(audio_files, exp_dir)
    py = _python()
    started = time.time()
    _notify(progress, estimate_message(src, total_epoch))
    _notify(progress, f"实验目录：{exp_dir}")
    _notify(progress, f"训练输入（仅音频）：{staged}")

    _run_step(
        rvc,
        _official_args("train/preprocess.py", [str(staged), str(sr_hz), "1", str(exp_dir), "True", str(float(slice_sec))]),
        exp_dir / "preprocess.log",
        "数据切分",
        progress,
    )
    compute = select_compute()
    _notify(progress, compute.message)
    _notify(progress, apply_official_train_hook())
    if compute.is_directml and missing_rmvpe_onnx() and f0_method == "rmvpe":
        raise MissingDependencyError(
            "Windows DirectML 音高提取需要 assets/rmvpe/rmvpe.onnx。"
            "请运行：scripts/download_base_models.ps1（Windows 会默认下载 rmvpe.onnx）"
        )
    if compute.is_directml and f0_method == "rmvpe":
        f0_args = ["dml", str(exp_dir)]
        hubert_device = compute.device_str  # e.g. privateuseone:0
    else:
        if compute.is_directml and f0_method != "rmvpe":
            _notify(
                progress,
                f"当前音高方法是 {f0_method}，官方 DirectML 提取脚本只支持 rmvpe，这一步回退 CPU。",
            )
        f0_args = ["cpu", str(exp_dir), "1", f0_method]
        hubert_device = "cpu"
    _run_step(
        rvc,
        _official_args("train/dataset/extract_f0.py", f0_args),
        exp_dir / "extract_f0_feature.log",
        "音高提取",
        progress,
    )
    _run_step(
        rvc,
        _official_args(
            "train/dataset/extract_hubert_feature.py",
            [hubert_device, "1", "0", str(exp_dir), version, "False"],
        ),
        exp_dir / "extract_f0_feature.log",
        "特征提取",
        progress,
    )
    n_clips = _build_filelist(rvc, exp_dir, version, sample_rate, use_f0=True)
    _copy_config(rvc, exp_dir, version, sample_rate)
    _notify(
        progress,
        f"可用训练片段：{n_clips}。开始模型训练（{total_epoch} 轮，batch={batch_size}，设备：{compute.label}）。",
    )
    _run_step(
        rvc,
        _official_args(
            "train/train.py",
            [
                "-e", safe,
                "-sr", sample_rate,
                "-f0", "1",
                "-bs", str(batch_size),
                "-te", str(total_epoch),
                "-se", str(max(1, min(total_epoch, 10))),
                "-pg", pre_g,
                "-pd", pre_d,
                "-l", "1",
                "-c", "0",
                "-sw", "1",
                "-v", version,
            ],
        ),
        exp_dir / "train.log",
        "模型训练",
        progress,
    )
    _run_step(
        rvc,
        _official_args("train/train_index.py", [safe, version, str(indices_dir()), "1", "single"]),
        exp_dir / "train_index.log",
        "索引训练",
        progress,
    )

    pth = weights_dir() / f"{safe}.pth"
    # savee 写入官方 assets/weights，已符号链接到本项目
    if not pth.is_file():
        # 兜底：从 logs 里找最新 G_ 权重再提示
        raise TrainingError(f"训练结束但未找到导出模型：{pth}")
    index_candidates = sorted(exp_dir.glob("added_*.index")) + sorted(indices_dir().glob(f"*{safe}*.index"))
    index_path = None
    for cand in index_candidates:
        dest = indices_dir() / cand.name
        if cand.resolve() != dest.resolve():
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            try:
                dest.symlink_to(cand.resolve())
            except OSError:
                shutil.copy2(cand, dest)
        alias = indices_dir() / f"added_{safe}.index"
        if alias.resolve() != Path(cand).resolve():
            if alias.exists() or alias.is_symlink():
                alias.unlink()
            try:
                alias.symlink_to(Path(cand).resolve())
            except OSError:
                shutil.copy2(cand, alias)
        index_path = alias if alias.exists() else dest
        break
    elapsed = time.time() - started
    msg = (
        f"训练完成，用时 {elapsed/60:.1f} 分钟。\n"
        f"模型：{pth}\n"
        + (f"索引：{index_path}\n" if index_path else "未找到索引文件。\n")
        + "请只在本机使用，不要把录音或音色模型上传到云端。"
    )
    _notify(progress, msg)
    return {"ok": True, "pth": str(pth), "index": str(index_path) if index_path else "", "message": msg, "clips": n_clips, "elapsed_sec": elapsed}
