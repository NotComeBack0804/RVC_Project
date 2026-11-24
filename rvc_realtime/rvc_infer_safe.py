"""对官方 RVC.infer 做安全索引融合：逐帧处理 faiss 的 -1，而不是整块丢弃。"""
from __future__ import annotations

from time import time as ttime
from typing import Any

import numpy as np


class IndexStats:
    def __init__(self):
        self.chunks = 0
        self.frames = 0
        self.used = 0
        self.skipped = 0
        self.disabled_chunks = 0

    def as_dict(self) -> dict:
        total = max(1, self.frames)
        return {
            "chunks": self.chunks,
            "frames": self.frames,
            "used": self.used,
            "skipped": self.skipped,
            "disabled_chunks": self.disabled_chunks,
            "used_ratio": self.used / total,
            "skipped_ratio": self.skipped / total,
        }

    def summary(self) -> str:
        d = self.as_dict()
        if self.frames == 0:
            return "未使用检索索引（index_rate=0 或没有可用索引）。"
        return (
            f"索引融合：{self.used}/{self.frames} 帧成功"
            f"（{d['used_ratio']*100:.1f}%），"
            f"跳过 {self.skipped} 帧（{d['skipped_ratio']*100:.1f}%）；"
            f"共 {self.chunks} 块。"
        )


def blend_index(
    feats,
    skip_head: int,
    index,
    big_npy,
    index_rate: float,
    stats: IndexStats | None = None,
    device=None,
):
    import torch

    if index is None or index_rate == 0 or big_npy is None:
        if stats is not None:
            stats.disabled_chunks += 1
            stats.chunks += 1
        return feats
    start = skip_head // 2
    npy = feats[0][start:].detach().cpu().numpy().astype("float32")
    if npy.size == 0:
        return feats
    k = 8
    score, ix = index.search(npy, k)
    blended = npy.copy()
    used = skipped = 0
    for i in range(npy.shape[0]):
        valid = ix[i] >= 0
        if not np.any(valid):
            skipped += 1
            continue
        ids = ix[i][valid]
        s = np.maximum(score[i][valid], 1e-8)
        weight = np.square(1.0 / s)
        weight /= weight.sum()
        blended[i] = np.sum(big_npy[ids] * weight[:, None], axis=0)
        used += 1
    if used:
        mixed = blended * float(index_rate) + (1.0 - float(index_rate)) * npy
        mixed_t = torch.from_numpy(mixed).to(device=feats.device, dtype=feats.dtype)
        feats = feats.clone()
        feats[0][start:] = mixed_t
    if stats is not None:
        stats.chunks += 1
        stats.frames += int(npy.shape[0])
        stats.used += used
        stats.skipped += skipped
    return feats


def install_safe_infer(rvc, stats: IndexStats) -> None:
    """替换实例的 infer，保留官方特征/音高/合成，只改索引融合。"""
    import torch
    import torch.nn.functional as F
    from infer.hubert import extract_hubert_features
    from tools.cuda_graph import run_cuda_graph

    def infer(input_wav, block_frame_16k, skip_head, return_length, f0method):
        report_status = rvc.infer_count < 3 or rvc.infer_count % 100 == 0
        rvc.infer_count += 1
        t1 = ttime()
        with torch.no_grad():
            if rvc.config.is_half:
                feats = input_wav.to(device=rvc.device).half().view(1, -1)
            else:
                feats = input_wav.to(device=rvc.device).float().view(1, -1)
            padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=rvc.device)
            feats = extract_hubert_features(
                rvc.model,
                feats,
                rvc.version,
                padding_mask=padding_mask,
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
        t2 = ttime()
        try:
            if hasattr(rvc, "index") and rvc.index_rate != 0:
                feats = blend_index(
                    feats,
                    skip_head,
                    rvc.index,
                    getattr(rvc, "big_npy", None),
                    rvc.index_rate,
                    stats=stats,
                    device=rvc.device,
                )
            else:
                if stats is not None:
                    stats.disabled_chunks += 1
                    stats.chunks += 1
                if report_status:
                    print("索引检索未启用", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"索引检索失败：{exc}", flush=True)
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        factor = pow(2, rvc.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        if rvc.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf = rvc.get_f0(
                input_wav[-f0_extractor_frame:],
                rvc.f0_up_key - rvc.formant_shift,
                f0method,
            )
            shift = block_frame_16k // 160
            rvc.cache_pitch[:-shift] = rvc.cache_pitch[shift:].clone()
            rvc.cache_pitchf[:-shift] = rvc.cache_pitchf[shift:].clone()
            rvc.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            rvc.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = rvc.cache_pitch[None, -p_len:]
            cache_pitchf = rvc.cache_pitchf[None, -p_len:] * return_length2 / return_length
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len_tensor = torch.LongTensor([p_len]).to(rvc.device)
        sid = torch.LongTensor([0]).to(rvc.device)
        skip_head_value = int(skip_head)
        return_length_value = int(return_length)
        return_length2_value = int(return_length2)
        with torch.no_grad():
            if rvc.if_f0 == 1:
                infered_audio = run_cuda_graph(
                    rvc.net_g,
                    "rvc-realtime-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, coarse, continuous, speaker: rvc.net_g.infer(
                        phone,
                        lengths,
                        coarse,
                        continuous,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                )
            else:
                infered_audio = run_cuda_graph(
                    rvc.net_g,
                    "rvc-realtime-no-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, speaker: rvc.net_g.infer(
                        phone,
                        lengths,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    sid,
                )
        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * rvc.tgt_sr // 100))
        if upp_res != rvc.tgt_sr // 100:
            if upp_res not in rvc.resample_kernel:
                from torchaudio.transforms import Resample

                rvc.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=rvc.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(rvc.device)
            infered_audio = rvc.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
        t5 = ttime()
        if report_status:
            print(
                "耗时：特征=%.3f秒，索引=%.3f秒，音高=%.3f秒，模型=%.3f秒"
                % (t2 - t1, t3 - t2, t4 - t3, t5 - t4),
                flush=True,
            )
        return infered_audio.squeeze()

    rvc.infer = infer
