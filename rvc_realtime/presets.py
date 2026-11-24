"""游戏短音频变身的预设参数。

面向鹅鸭杀、狼人杀这类发言局：只有 1–2 分钟干净说话（不是唱歌、不是 10–30 分钟长语料）。
默认值写在这里，主界面直接使用，用户不必再填一轮参数。
"""
from __future__ import annotations

# ---------- 短语料训练 ----------
# version=v2、sr=40k：官方 RVC 最常见的 f0 语音组合，底模也是 f0G40k/f0D40k。
# batch_size=1：短数据再加大 batch 也凑不满；CPU 与 DirectML 上都更稳，减少内存抖动。
# f0_method=rmvpe：说话语音的音高提取比 pm 准，游戏发言以对白为主。
# slice_sec=3.0：官方默认切分约 3.7s。1–2 分钟语料用 3.0s 能多切出几段，方便学习音色；
# 再短（<2s）音高/特征会不稳，所以不继续往下压。
# epochs=60：官方一键训练常用 200 轮，那是按 10–30 分钟语料来的。1–2 分钟若跑 200 轮，
# 很容易把那几句发言背下来，换一句就飘。轮数太少（例如 10–20）则往往还停在底模附近，
# 音色学不够。60 轮是在「能从短干净语音里贴上音色」和「少背词」之间的折中，可在高级设置改。
GAME_TRAIN = {
    "name": "game_voice",
    "version": "v2",
    "sample_rate": "40k",
    "total_epoch": 60,
    "batch_size": 1,
    "f0_method": "rmvpe",
    "slice_sec": 3.0,
}

# ---------- 游戏实时变声 ----------
# buffer_ms=160：先保证不爆音、不断续，再尽量压延迟。落在任务要求的 120–180 ms 一档里，
# 比 200 ms 更接近发言游戏的回合感，又比 100–120 ms 更不易卡顿。DirectML 加速后可再尝试略减缓冲。可在高级设置调节。
# f0_up_key=0：默认不变调，避免发言游戏里声音飘高飘低。
# index_rate=0.5：官方常用融合比。短数据索引样本少，不再默认调得更高，以免检索伪影。
# gain_db=0、loudness_mode=match-input：用响度对齐避免输出明显偏小，同时限制峰值，不必再靠默认加增益。
GAME_REALTIME = {
    "buffer_ms": 160,
    "f0_up_key": 0,
    "index_rate": 0.5,
    "gain_db": 0.0,
    "loudness_mode": "match-input",
    "peak_limit": 0.95,
}

GAME_DURATIONS = {
    "1 分钟": 60,
    "2 分钟": 120,
}

DEFAULT_MODEL_NAME = GAME_TRAIN["name"]


def game_train_estimate(audio_count: int | None = None) -> str:
    n = f"{audio_count} 个音频文件，" if audio_count is not None else ""
    prefix = (
        f"按「游戏短音频」预设训练：{n}"
        f"{GAME_TRAIN['total_epoch']} 轮，RVC {GAME_TRAIN['version']} / {GAME_TRAIN['sample_rate']}，"
        f"切分 {GAME_TRAIN['slice_sec']} 秒，batch={GAME_TRAIN['batch_size']}，音高提取 {GAME_TRAIN['f0_method']}。"
        "面向 1–2 分钟干净发言，轮数比官方 200 轮低，减轻背词，仍尽量学到音色。"
    )
    try:
        from rvc_realtime.compute import select_compute

        compute = select_compute()
        extra = compute.message
        if compute.is_directml:
            extra += "显卡加速后通常比 CPU 快，仍可以中断。"
        else:
            extra += "CPU 上回退时往往需要数十分钟到约两小时，可以中断。"
        return prefix + extra
    except Exception:
        return prefix + "CPU 上往往需要数十分钟到约两小时，可以中断。"
