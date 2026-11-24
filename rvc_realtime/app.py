"""本地界面：游戏短音频变身。默认录系统扬声器，一键训练，一键实时变声。不上传云端。"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from rvc_realtime.devices import (
    all_capture_labels,
    default_loopback_label,
    list_all_devices,
    loopback_labels,
    mic_labels,
    preferred_mic_names,
    preferred_output_names,
)
from rvc_realtime.errors import AppError
from rvc_realtime.paths import indices_dir, project_root, testdata_dir, weights_dir
from rvc_realtime.presets import (
    DEFAULT_MODEL_NAME,
    GAME_DURATIONS,
    GAME_REALTIME,
    GAME_TRAIN,
    game_train_estimate,
)
from rvc_realtime.pulse import pulse_available, setup_virtual_audio
from rvc_realtime.recorder import SpeakerRecorder, import_authorized_wav
from rvc_realtime.session import RealtimeSession
from rvc_realtime.train_pipeline import request_stop, run_training

recorder = SpeakerRecorder()
session = RealtimeSession()

AUTH_LABEL = "这是已授权的声音（本人或已获授权）"

HERO_HTML = """
<div id="game-hero">
  <div class="kicker">本地音色工具</div>
  <h1>游戏短音频变身</h1>
  <p>面向鹅鸭杀、狼人杀这类发言局：录下电脑正在播放的 1 或 2 分钟纯净发言，一键做成音色，再实时变声。录音、训练和变声都在本机进行。</p>
</div>
"""

STEP1_HTML = """
<div class="step-card-title">
  <div class="step-head"><span class="step-num">1</span><p class="step-title">录制扬声器</p></div>
  <p class="step-help">录的是电脑正在播放的声音，不是麦克风。请先在游戏或播放器里放出那 1–2 分钟纯净发言，再开始录。录音时先不要开实时变声，也不要从 CABLE 录，以免把变声结果再录进去。没有环回设备时，可以导入已有短 WAV。</p>
</div>
"""

STEP2_HTML = """
<div class="step-card-title">
  <div class="step-head"><span class="step-num">2</span><p class="step-title">一键训练</p></div>
  <p class="step-help">参数已按 1–2 分钟干净发言预设好，一般不用改。CPU 上可能需要数十分钟到约两小时，可以中断。</p>
</div>
"""

STEP3_HTML = """
<div class="step-card-title">
  <div class="step-head"><span class="step-num">3</span><p class="step-title">一键实时变声</p></div>
  <p class="step-help">Windows：本程序输出选 CABLE Input，游戏或聊天软件的麦克风选 CABLE Output。请戴耳机。可随时把游戏麦克风切回真实麦克风。</p>
</div>
"""

FOOT_HTML = """
<div id="game-foot">录音主路径是系统扬声器 / 环回。麦克风、轮数、音高和缓冲等项都在下面的高级设置里。</div>
"""


def _outputs() -> list[str]:
    devices, _ = list_all_devices()
    names = [d.name for d in devices if d.kind == "output"]
    return names or ["（无可用输出设备）"]


def _models() -> list[str]:
    return [p.name for p in sorted(weights_dir().glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)]


def _index_for(model_name: str) -> str:
    if not model_name:
        return ""
    stem = Path(model_name).stem
    for cand in (
        indices_dir() / f"{stem}.index",
        indices_dir() / f"added_{stem}.index",
        indices_dir() / f"{stem}_added.index",
    ):
        if cand.is_file() or cand.is_symlink():
            return str(cand)
    found = list(indices_dir().glob(f"*{stem}*.index"))
    return str(found[0]) if found else ""


def _default_model() -> str | None:
    models = _models()
    for name in (f"{DEFAULT_MODEL_NAME}.pth", "game_voice.pth"):
        if name in models:
            return name
    usable = [m for m in models if "smoke" not in m.lower()]
    return (usable[0] if usable else (models[0] if models else None))


def _record_choices() -> tuple[list[str], str | None]:
    labels = loopback_labels()
    default = default_loopback_label()
    if default and default in labels:
        return labels, default
    return labels, (labels[0] if labels else None)


def _empty_record_placeholder(label: str | None) -> bool:
    return not label or str(label).startswith("（")


def start_record(device, duration_label, authorized):
    if not authorized:
        yield "请先勾选：这是已授权的声音。"
        return
    if _empty_record_placeholder(device):
        yield (
            "没有可用的扬声器环回。Windows 请选 WASAPI 环回或「立体声混音」；"
            "Linux 请选 speakers.monitor。也可以导入已有短 WAV。"
        )
        return
    seconds = GAME_DURATIONS.get(duration_label, 60)
    try:
        recorder.start(device, max_seconds=float(seconds))
    except AppError as exc:
        yield exc.message
        return
    while recorder.running:
        yield recorder.status_text()
        time.sleep(0.4)
    path = recorder.stop()
    extra = f"\n已保存到：{path}" if path else ""
    if recorder.auto_stopped:
        extra = "\n已到目标时长，自动停止。" + extra
    yield recorder.status_text() + extra


def stop_record():
    path = recorder.stop()
    extra = f"\n已保存到：{path}" if path else ""
    return recorder.status_text() + extra


def import_wav(file_obj, authorized):
    if not authorized:
        return "请先勾选授权，再导入录音。"
    if file_obj is None:
        return "请先选择一个 WAV/FLAC 文件。"
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        dest = import_authorized_wav(path)
        return f"已导入到：{dest}\n将使用目录：{dest.parent}"
    except AppError as exc:
        return exc.message


def start_train(name, audio_dir, epochs, batch, authorized, version, sr, slice_sec):
    if not authorized:
        yield "训练被拒绝：请勾选「这是已授权的声音」。"
        return
    audio_dir = (audio_dir or "").strip() or str(testdata_dir() / "authorized_reference")
    name = (name or "").strip() or DEFAULT_MODEL_NAME
    q: queue.Queue = queue.Queue()

    def cb(msg: str):
        q.put(msg)

    def worker():
        try:
            result = run_training(
                audio_dir=audio_dir,
                name=name,
                authorized=True,
                version=version,
                sample_rate=sr,
                total_epoch=int(epochs),
                batch_size=int(batch),
                f0_method=GAME_TRAIN["f0_method"],
                slice_sec=float(slice_sec),
                progress=cb,
            )
            q.put(("DONE", result["message"]))
        except Exception as exc:  # noqa: BLE001
            q.put(("ERR", str(exc)))

    threading.Thread(target=worker, daemon=True).start()
    lines: list[str] = [game_train_estimate()]
    yield "\n".join(lines)
    while True:
        item = q.get()
        if isinstance(item, tuple):
            _kind, msg = item
            lines.append(msg)
            yield "\n".join(lines[-80:])
            return
        lines.append(item)
        yield "\n".join(lines[-80:])


def stop_train():
    request_stop()
    return "已请求中断训练。当前步骤结束后会停止。"


def start_rt(model, index, in_dev, out_dev, pitch, buffer_ms, gain_db, index_rate, passthrough):
    actual_mode = "passthrough" if passthrough else "convert"
    if actual_mode == "convert" and not model:
        return "还没有可加载的音色模型。请先完成训练，或在高级设置中选择已有模型。"
    try:
        session.start(
            mode=actual_mode,
            input_spec=in_dev,
            output_spec=out_dev,
            model=None if actual_mode == "passthrough" else str(weights_dir() / model),
            index=index or _index_for(model or "") or None,
            buffer_ms=int(buffer_ms),
            f0_up_key=int(pitch),
            index_rate=float(index_rate),
            gain_db=float(gain_db),
            loudness_mode=GAME_REALTIME["loudness_mode"],
        )
        return session.stats
    except AppError as exc:
        return exc.message


def stop_rt():
    return session.stop()


def rt_status():
    if session.error:
        return session.stats + "\n" + session.error
    return session.stats or "尚未启动。"


def setup_audio():
    if not pulse_available():
        return "当前不是 Linux PulseAudio 环境。Windows 请先安装 VB-CABLE，然后刷新设备列表。"
    try:
        info = setup_virtual_audio()
        return "虚拟音频：" + "；".join(f"{k}={v}" for k, v in info.items())
    except AppError as exc:
        return exc.message


def simulate_rt(model, index, buffer_ms, gain_db, index_rate):
    from rvc_realtime.engine import format_run_result
    from rvc_realtime.session import run_simulated_realtime

    src = testdata_dir() / "generated" / "speech_jfk.wav"
    dest = testdata_dir() / "recorded" / "simulate_realtime.wav"
    if not src.is_file():
        return "找不到测试音频 testdata/generated/speech_jfk.wav"
    if not model:
        return "请先选择模型。"
    try:
        result = run_simulated_realtime(
            str(src),
            str(dest),
            model=str(weights_dir() / model),
            index=index or None,
            buffer_ms=int(buffer_ms),
            gain_db=float(gain_db),
            index_rate=float(index_rate),
        )
        return format_run_result(result) + f"\n输出：{dest}"
    except AppError as exc:
        return exc.message


def refresh_devices():
    import gradio as gr

    rec_choices, rec_val = _record_choices()
    adv_capture = all_capture_labels()
    mics = mic_labels()
    outs = _outputs()
    models = _models()
    mic_val = preferred_mic_names(mics)
    out_val = preferred_output_names(outs)
    model_val = _default_model()
    return (
        gr.update(choices=rec_choices, value=rec_val),
        gr.update(choices=adv_capture, value=rec_val if rec_val in adv_capture else (adv_capture[0] if adv_capture else None)),
        gr.update(choices=mics, value=mic_val),
        gr.update(choices=outs, value=out_val),
        gr.update(choices=models, value=model_val),
        _index_for(model_val or ""),
        "设备已刷新。录音默认扬声器环回；实时输出请优先选 CABLE Input。",
    )


def launch_app(host: str = "127.0.0.1", port: int = 7870, share: bool = False) -> None:
    import gradio as gr

    if share:
        raise RuntimeError("禁止开启 Gradio share：录音和音色必须留在本机，不能上传云端。")

    rec_choices, rec_default = _record_choices()
    adv_capture = all_capture_labels()
    mics = mic_labels()
    outs = _outputs()
    models = _models()
    mic_default = preferred_mic_names(mics)
    out_default = preferred_output_names(outs)
    model_default = _default_model()
    css_path = str(Path(__file__).with_name("ui.css"))
    audio_dir_default = str(testdata_dir() / "authorized_reference")

    with gr.Blocks(
        title="游戏短音频变身",
        css=css_path,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(HERO_HTML)
        with gr.Group(elem_id="auth-row"):
            auth = gr.Checkbox(label=AUTH_LABEL, value=False)
        with gr.Group(elem_id="step-wrap"):
            gr.HTML(STEP1_HTML)
            rec_duration = gr.Radio(
                choices=list(GAME_DURATIONS.keys()),
                value="1 分钟",
                label="目标时长",
            )
            rec_device = gr.Dropdown(
                choices=rec_choices,
                value=rec_default,
                label="录音设备（扬声器 / 系统音频 / 环回）",
            )
            rec_status = gr.Textbox(label="录音状态", lines=4, elem_id="status-record")
            with gr.Row():
                rec_start = gr.Button("开始录制扬声器", variant="primary", elem_id="btn-record")
                rec_stop = gr.Button("停止并保存", variant="stop", elem_id="btn-stop-record")
            wav_file = gr.File(label="或导入已有短 WAV / FLAC")
            rec_import = gr.Button("导入并用作训练素材", variant="secondary")

        with gr.Group(elem_id="step-wrap"):
            gr.HTML(STEP2_HTML)
            train_name = gr.Textbox(label="模型名", value=DEFAULT_MODEL_NAME)
            train_log = gr.Textbox(label="训练进度", lines=8, elem_id="status-train")
            with gr.Row():
                train_start = gr.Button("一键训练", variant="primary", elem_id="btn-train")
                train_stop = gr.Button("中断训练", variant="stop", elem_id="btn-stop-train")

        with gr.Group(elem_id="step-wrap"):
            gr.HTML(STEP3_HTML)
            rt_model = gr.Dropdown(choices=models, value=model_default, label="音色模型")
            rt_status_box = gr.Textbox(label="实时状态", lines=4, elem_id="status-live")
            with gr.Row():
                rt_start = gr.Button("一键开始实时变声", variant="primary", elem_id="btn-live")
                rt_stop = gr.Button("停止变声", variant="stop", elem_id="btn-stop-live")

        with gr.Accordion("高级设置", open=False, elem_id="advanced-wrap"):
            gr.Markdown(
                "一般不必改。录音主路径已默认扬声器环回；麦克风只放在这里。"
                "直通用于排查路由，不要挡在主按钮前面。"
            )
            train_dir = gr.Textbox(label="训练音频目录", value=audio_dir_default)
            with gr.Row():
                train_epochs = gr.Number(label="训练轮数", value=GAME_TRAIN["total_epoch"], precision=0)
                train_batch = gr.Number(label="批大小", value=GAME_TRAIN["batch_size"], precision=0)
                train_slice = gr.Number(label="切分秒数", value=GAME_TRAIN["slice_sec"])
            with gr.Row():
                train_ver = gr.Dropdown(choices=["v2", "v1"], value=GAME_TRAIN["version"], label="版本")
                train_sr = gr.Dropdown(choices=["40k", "48k", "32k"], value=GAME_TRAIN["sample_rate"], label="采样率")
            rec_device_adv = gr.Dropdown(
                choices=adv_capture,
                value=rec_default if rec_default in adv_capture else (adv_capture[0] if adv_capture else None),
                label="全部录音设备（含麦克风）",
            )
            with gr.Row():
                rt_in = gr.Dropdown(choices=mics, value=mic_default, label="实时输入（麦克风）")
                rt_out = gr.Dropdown(choices=outs, value=out_default, label="实时输出（CABLE Input）")
            rt_index = gr.Textbox(label="索引路径", value=_index_for(model_default or ""))
            with gr.Row():
                rt_pitch = gr.Number(label="音高（半音）", value=GAME_REALTIME["f0_up_key"], precision=0)
                rt_buffer = gr.Number(label="缓冲 ms", value=GAME_REALTIME["buffer_ms"], precision=0)
                rt_gain = gr.Number(label="额外增益 dB", value=GAME_REALTIME["gain_db"])
                rt_irate = gr.Number(label="索引比例", value=GAME_REALTIME["index_rate"])
            rt_pass = gr.Checkbox(label="直通（只验证路由，不变声）", value=False)
            with gr.Row():
                rt_refresh = gr.Button("刷新设备 / 模型", variant="secondary")
                rt_setup = gr.Button("配置 Linux 虚拟音频", variant="secondary")
                rt_poll = gr.Button("刷新实时状态", variant="secondary")
                rt_sim = gr.Button("模拟实时（测试 WAV）", variant="secondary")

        gr.HTML(FOOT_HTML)

        def _sync_rec_from_adv(dev):
            return dev

        def _sync_rec_to_adv(dev):
            return dev

        rec_start.click(start_record, [rec_device, rec_duration, auth], [rec_status])
        rec_stop.click(stop_record, [], [rec_status])
        rec_import.click(import_wav, [wav_file, auth], [rec_status])
        rec_device_adv.change(_sync_rec_from_adv, [rec_device_adv], [rec_device])
        rec_device.change(_sync_rec_to_adv, [rec_device], [rec_device_adv])
        train_start.click(
            start_train,
            [train_name, train_dir, train_epochs, train_batch, auth, train_ver, train_sr, train_slice],
            [train_log],
        )
        train_stop.click(stop_train, [], [train_log])
        rt_model.change(lambda m: _index_for(m or ""), [rt_model], [rt_index])
        rt_refresh.click(
            refresh_devices,
            [],
            [rec_device, rec_device_adv, rt_in, rt_out, rt_model, rt_index, rt_status_box],
        )
        rt_setup.click(setup_audio, [], [rt_status_box])
        rt_start.click(
            start_rt,
            [rt_model, rt_index, rt_in, rt_out, rt_pitch, rt_buffer, rt_gain, rt_irate, rt_pass],
            [rt_status_box],
        )
        rt_stop.click(stop_rt, [], [rt_status_box])
        rt_poll.click(rt_status, [], [rt_status_box])
        rt_sim.click(simulate_rt, [rt_model, rt_index, rt_buffer, rt_gain, rt_irate], [rt_status_box])

    demo.queue()
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=False,
        show_error=True,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    launch_app()
