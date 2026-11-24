"""本地界面：游戏短音频变身。默认录系统扬声器，一键训练，一键实时变声。不上传云端。"""
from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path

from rvc_realtime.devices import (
    all_capture_labels,
    default_loopback_label,
    list_all_devices,
    preferred_mic_names,
    preferred_output_names,
    mic_labels,
)
from rvc_realtime.compute import select_compute
from rvc_realtime.errors import AppError
from rvc_realtime.paths import indices_dir, testdata_dir, weights_dir
from rvc_realtime.pretrained import (
    import_pretrained_pair,
    pair_choices,
    placement_help,
    resolve_pretrained,
)
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

BRAND_HTML = """
<div id="app-brand">
  <div class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
  <div class="brand-text">
    <h1>游戏短音频变身</h1>
    <p>本地工作台 · 默认录系统扬声器</p>
  </div>
</div>
"""

RAIL_HTML = """
<nav id="app-rail-nav">
  <input type="radio" name="app-step" id="step-record" checked>
  <input type="radio" name="app-step" id="step-train">
  <input type="radio" name="app-step" id="step-live">
  <input type="checkbox" id="settings-toggle">
  <label class="rail-step" for="step-record" id="rail-label-record">
    <span class="rail-num">1</span>
    <div><strong>录制扬声器</strong><em>系统正在播放的声音</em></div>
  </label>
  <label class="rail-step" for="step-train" id="rail-label-train">
    <span class="rail-num">2</span>
    <div><strong>一键训练</strong><em>做成 game_voice</em></div>
  </label>
  <label class="rail-step" for="step-live" id="rail-label-live">
    <span class="rail-num">3</span>
    <div><strong>实时变声</strong><em>输出到虚拟麦克风</em></div>
  </label>
  <label class="rail-settings-btn" for="settings-toggle" id="rail-open-settings">系统设置</label>
  <p class="rail-note">点击步骤切换模块。换录音扬声器或输出麦克风请打开系统设置；音高、缓冲和训练超参也在里面。</p>
</nav>
"""

STEP1_HTML = """
<div class="step-card-title">
  <div class="step-head"><span class="step-num">1</span><p class="step-title">录制扬声器</p></div>
  <p class="step-help">录的是电脑正在播放的声音，不是麦克风。先放出 1–2 分钟纯净发言再开始。不要从 CABLE 录，以免把变声结果再录进去。要换录音设备，请打开左侧系统设置。</p>
</div>
"""

STEP3_HTML = """
<div class="step-card-title">
  <div class="step-head"><span class="step-num">3</span><p class="step-title">一键实时变声</p></div>
  <p class="step-help">默认输出到 CABLE Input，游戏或聊天软件的麦克风选 CABLE Output。请戴耳机，可随时切回真实麦克风。要换输出设备，请打开左侧系统设置。</p>
</div>
"""

SETTINGS_HEAD_HTML = """
<div id="settings-head">
  <input type="radio" name="settings-tab" id="tab-system" checked>
  <input type="radio" name="settings-tab" id="tab-advanced">
  <label class="modal-backdrop" for="settings-toggle" aria-label="关闭设置"></label>
  <div class="settings-panel-bar">
    <div class="settings-tabs">
      <label for="tab-system">系统设置</label>
      <label for="tab-advanced">高级设置</label>
    </div>
    <label class="modal-close" for="settings-toggle">关闭</label>
  </div>
</div>
"""

SYSTEM_HELP_HTML = """
<p class="settings-help">录音默认使用系统扬声器环回，不是麦克风。实时变声的输出请优先选 CABLE Input，聊天或游戏的麦克风选 CABLE Output。换线缆或声卡后点「刷新设备列表」。</p>
"""

ADVANCED_HELP_HTML = """
<p class="settings-help">一般不必改。音高、缓冲、路径和训练超参都在这里。麦克风可作为次要录音输入或实时输入。直通用于排查路由。</p>
"""

FOOT_HTML = """
<div id="game-foot">本工具只用于你本人的声音，或已获得明确授权的声音。录音、训练和变声都在本机进行，不会上传云端。许可说明见项目 LICENSE。</div>
"""


def _compute_banner_html() -> str:
    info = select_compute()
    cls = "ok" if info.is_directml else "cpu"
    adapter = f'<span class="compute-adapter">{info.adapter_name}</span>' if info.adapter_name else ""
    reason = f'<div class="compute-reason">{info.fallback_reason}</div>' if info.fallback_reason else ""
    return (
        f'<div id="compute-banner" class="{cls}">'
        f'<span class="compute-kicker">计算设备</span>'
        f"<strong>{info.label}</strong>"
        f"{adapter}{reason}</div>"
    )


def _step2_html() -> str:
    info = select_compute()
    if info.is_directml:
        help_txt = "参数已按 1–2 分钟干净发言设好。将使用显卡（DirectML）加速，可以中断。训练底模可在下方更换。"
    elif info.fallback_reason:
        help_txt = "参数已按短发言设好。当前为 CPU（DirectML 仅在 Windows 可用），可能需要数十分钟到约两小时，可以中断。训练底模可在下方更换。"
    else:
        help_txt = "参数已按 1–2 分钟干净发言设好。当前为 CPU，可能需要数十分钟到约两小时，可以中断。训练底模可在下方更换。"
    return f"""
<div class="step-card-title">
  <div class="step-head"><span class="step-num">2</span><p class="step-title">一键训练</p></div>
  <p class="step-help">{help_txt}</p>
</div>
"""


def _fmt_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _record_meter_html() -> str:
    running = recorder.running
    err = recorder.error if not running else None
    rms = max(0.0, min(1.0, float(recorder.last_rms) * 3.2))
    peak = max(0.0, min(1.0, float(recorder.last_peak)))
    dur = recorder.duration()
    target = recorder.max_seconds or 60
    if err:
        state, label = "err", f"录音失败：{err}"
    elif running:
        state, label = "rec", "正在录制扬声器"
    elif recorder.path:
        state, label = "ok", "已保存，可用于训练"
    else:
        state, label = "idle", "待命 · 默认录系统扬声器"
    bars = []
    for i in range(36):
        wave = 0.2 + 0.35 * abs(math.sin(i / 2.8))
        if running:
            height = min(1.0, wave * 0.35 + rms * (0.45 + 0.55 * abs(math.sin(i * 0.7 + dur))))
        elif recorder.path:
            height = wave * 0.7
        else:
            height = wave * 0.5
        bars.append(f'<i style="height:{int(8 + height * 36)}px"></i>')
    return (
        f'<div class="vu-panel" data-state="{state}">'
        f'<div class="vu-time"><strong>{_fmt_clock(dur)}</strong><span>/ {_fmt_clock(target)}</span></div>'
        f'<div class="vu-wave">{"".join(bars)}</div>'
        f'<div class="vu-level"><b style="width:{rms * 100:.1f}%"></b>'
        f'<em style="left:{peak * 100:.1f}%"></em></div>'
        f'<div class="vu-label">{label}</div></div>'
    )


def _live_meter_html() -> str:
    if session.error and not session.running:
        state, label = "err", "出错"
    elif session.running:
        state, label = "live", "实时变声进行中"
    else:
        state, label = "idle", "尚未启动"
    return f'<div class="live-pill" data-state="{state}"><i></i><span>{label}</span></div>'


def _record_outputs(text: str | None = None) -> tuple[str, str]:
    status = text if text is not None else recorder.status_text()
    return status, _record_meter_html()


def _rt_outputs(text: str) -> tuple[str, str]:
    return text, _live_meter_html()


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
    labels = all_capture_labels()
    default = default_loopback_label()
    if default and default in labels:
        return labels, default
    loopbacks = [x for x in labels if x.startswith("扬声器环回") or x.startswith("系统音频")]
    if loopbacks:
        return labels, loopbacks[0]
    return labels, (labels[0] if labels else None)


def _device_hint_html(rec: str | None = None, out: str | None = None) -> str:
    if not rec:
        _, rec = _record_choices()
    rec = rec or "默认扬声器环回"
    if not out:
        outs = _outputs()
        out = preferred_output_names(outs) or (outs[0] if outs else "未选择")
    return (
        f'<p class="device-hint">录音设备：{rec}　·　输出/虚拟麦克风：{out}'
        f'　·　<label for="settings-toggle" class="device-hint-link">在系统设置中更改</label></p>'
    )


def _empty_record_placeholder(label: str | None) -> bool:
    return not label or str(label).startswith("（")


def start_record(device, duration_label, authorized):
    if not authorized:
        yield _record_outputs("请先勾选：这是已授权的声音。")
        return
    if _empty_record_placeholder(device):
        yield _record_outputs(
            "没有可用的扬声器环回。请打开系统设置选择 WASAPI 环回或「立体声混音」；"
            "Linux 请选 speakers.monitor。也可以导入已有短 WAV。"
        )
        return
    seconds = GAME_DURATIONS.get(duration_label, 60)
    try:
        recorder.start(device, max_seconds=float(seconds))
    except AppError as exc:
        yield _record_outputs(exc.message)
        return
    while recorder.running:
        yield _record_outputs()
        time.sleep(0.4)
    path = recorder.stop()
    extra = f"\n已保存到：{path}" if path else ""
    if recorder.auto_stopped:
        extra = "\n已到目标时长，自动停止。" + extra
    yield _record_outputs(recorder.status_text() + extra)


def stop_record():
    path = recorder.stop()
    extra = f"\n已保存到：{path}" if path else ""
    return _record_outputs(recorder.status_text() + extra)


def import_wav(file_obj, authorized):
    if not authorized:
        return _record_outputs("请先勾选授权，再导入录音。")
    if file_obj is None:
        return _record_outputs("请先选择一个 WAV/FLAC 文件。")
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        dest = import_authorized_wav(path)
        return _record_outputs(f"已导入到：{dest}\n将使用目录：{dest.parent}")
    except AppError as exc:
        return _record_outputs(exc.message)


def start_train(name, audio_dir, epochs, batch, authorized, version, sr, slice_sec, base_label):
    if not authorized:
        yield "训练被拒绝：请勾选「这是已授权的声音」。"
        return
    audio_dir = (audio_dir or "").strip() or str(testdata_dir() / "authorized_reference")
    name = (name or "").strip() or DEFAULT_MODEL_NAME
    pair = resolve_pretrained(base_label)
    if pair:
        version = pair.version or version
        sr = pair.sample_rate or sr
        pretrained_g, pretrained_d = str(pair.generator), str(pair.discriminator)
    else:
        pretrained_g = pretrained_d = None
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
                pretrained_g=pretrained_g,
                pretrained_d=pretrained_d,
            )
            q.put(("DONE", result["message"]))
        except Exception as exc:  # noqa: BLE001
            q.put(("ERR", str(exc)))

    threading.Thread(target=worker, daemon=True).start()
    lines: list[str] = [game_train_estimate()]
    if pair:
        lines.append(f"选用底模：{pair.label}")
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
        return _rt_outputs("还没有可加载的音色模型。请先完成训练，或在本页选择已有模型。")
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
        info = select_compute()
        prefix = info.message + "\n"
        return _rt_outputs(prefix + (session.stats or "正在启动实时变声…"))
    except AppError as exc:
        return _rt_outputs(exc.message)


def stop_rt():
    return _rt_outputs(session.stop())


def rt_status():
    if session.error:
        return _rt_outputs(session.stats + "\n" + session.error)
    return _rt_outputs(session.stats or "尚未启动。")


def setup_audio():
    if not pulse_available():
        return _rt_outputs("当前不是 Linux PulseAudio 环境。Windows 请先安装 VB-CABLE，然后在系统设置中刷新设备列表。")
    try:
        info = setup_virtual_audio()
        return _rt_outputs("虚拟音频：" + "；".join(f"{k}={v}" for k, v in info.items()))
    except AppError as exc:
        return _rt_outputs(exc.message)


def simulate_rt(model, index, buffer_ms, gain_db, index_rate):
    from rvc_realtime.engine import format_run_result
    from rvc_realtime.session import run_simulated_realtime

    src = testdata_dir() / "generated" / "speech_jfk.wav"
    dest = testdata_dir() / "recorded" / "simulate_realtime.wav"
    if not src.is_file():
        return _rt_outputs("找不到测试音频 testdata/generated/speech_jfk.wav")
    if not model:
        return _rt_outputs("请先选择模型。")
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
        return _rt_outputs(format_run_result(result) + f"\n输出：{dest}")
    except AppError as exc:
        return _rt_outputs(exc.message)


def refresh_devices(cur_rec, cur_in, cur_out, cur_model):
    import gradio as gr

    rec_choices, rec_default = _record_choices()
    mics = mic_labels()
    outs = _outputs()
    models = _models()
    rec_val = cur_rec if cur_rec in rec_choices else rec_default
    mic_val = cur_in if cur_in in mics else preferred_mic_names(mics)
    out_val = cur_out if cur_out in outs else preferred_output_names(outs)
    model_val = cur_model if cur_model in models else _default_model()
    return (
        gr.update(choices=rec_choices, value=rec_val),
        gr.update(choices=mics, value=mic_val),
        gr.update(choices=outs, value=out_val),
        gr.update(choices=models, value=model_val),
        _index_for(model_val or ""),
        "设备已刷新。录音默认扬声器环回；实时输出请优先选 CABLE Input。",
        _live_meter_html(),
        _device_hint_html(rec_val, out_val),
    )


def refresh_bases(cur_label):
    import gradio as gr

    labels, default = pair_choices()
    value = cur_label if cur_label in labels else default
    pair = resolve_pretrained(value)
    ver = pair.version if pair else GAME_TRAIN["version"]
    sr = pair.sample_rate if pair else GAME_TRAIN["sample_rate"]
    msg = f"已扫描到 {len(labels)} 个底模组合。当前：{value or '（无）'}。"
    return gr.update(choices=labels, value=value), msg, ver, sr


def import_bases(g_file, d_file, name, cur_label):
    import gradio as gr

    if g_file is None or d_file is None:
        labels, default = pair_choices()
        value = cur_label if cur_label in labels else default
        pair = resolve_pretrained(value)
        ver = pair.version if pair else GAME_TRAIN["version"]
        sr = pair.sample_rate if pair else GAME_TRAIN["sample_rate"]
        return (
            gr.update(choices=labels, value=value),
            "请同时选择生成器（G）和判别器（D）两个 .pth 文件。",
            ver,
            sr,
        )
    g_path = g_file.name if hasattr(g_file, "name") else str(g_file)
    d_path = d_file.name if hasattr(d_file, "name") else str(d_file)
    try:
        pair = import_pretrained_pair(g_path, d_path, name=name)
        labels, _default = pair_choices()
        msg = f"已导入到底模目录：{pair.generator.parent}\n{placement_help()}"
        return gr.update(choices=labels, value=pair.label), msg, pair.version, pair.sample_rate
    except AppError as exc:
        labels, default = pair_choices()
        value = cur_label if cur_label in labels else default
        cur = resolve_pretrained(value)
        ver = cur.version if cur else GAME_TRAIN["version"]
        sr = cur.sample_rate if cur else GAME_TRAIN["sample_rate"]
        return gr.update(choices=labels, value=value), exc.message, ver, sr


def _on_base_change(label):
    import gradio as gr

    pair = resolve_pretrained(label)
    if not pair:
        return gr.update(), gr.update()
    return gr.update(value=pair.version), gr.update(value=pair.sample_rate)


def _on_device_change(rec, out):
    return _device_hint_html(rec, out)


def launch_app(host: str = "127.0.0.1", port: int = 7870, share: bool = False) -> None:
    import gradio as gr

    if share:
        raise RuntimeError("禁止开启 Gradio share：录音和音色必须留在本机，不能上传云端。")

    rec_choices, rec_default = _record_choices()
    mics = mic_labels()
    outs = _outputs()
    models = _models()
    mic_default = preferred_mic_names(mics)
    out_default = preferred_output_names(outs)
    model_default = _default_model()
    base_labels, base_default = pair_choices()
    base_pair = resolve_pretrained(base_default)
    css_path = str(Path(__file__).with_name("ui.css"))
    audio_dir_default = str(testdata_dir() / "authorized_reference")

    with gr.Blocks(
        title="游戏短音频变身",
        css=css_path,
        analytics_enabled=False,
    ) as demo:
        with gr.Row(elem_id="app-topbar"):
            gr.HTML(BRAND_HTML, show_label=False)
            gr.HTML(_compute_banner_html(), show_label=False)
            with gr.Group(elem_id="auth-row"):
                auth = gr.Checkbox(label=AUTH_LABEL, value=False)

        with gr.Row(elem_id="app-shell"):
            with gr.Column(elem_id="app-rail", scale=0, min_width=228):
                gr.HTML(RAIL_HTML, show_label=False)
            with gr.Column(elem_id="app-main", scale=1, min_width=640):
                device_hint = gr.HTML(_device_hint_html(rec_default, out_default), show_label=False, elem_id="device-hint")

                with gr.Group(elem_id="stage-record"):
                    gr.HTML(STEP1_HTML, show_label=False)
                    rec_meter = gr.HTML(_record_meter_html(), show_label=False, elem_id="record-meter")
                    rec_duration = gr.Radio(
                        choices=list(GAME_DURATIONS.keys()),
                        value="1 分钟",
                        label="目标时长",
                        elem_id="rec-duration",
                    )
                    with gr.Row(elem_id="rec-actions"):
                        rec_start = gr.Button("开始录制扬声器", variant="primary", elem_id="btn-record")
                        rec_stop = gr.Button("停止并保存", variant="stop", elem_id="btn-stop-record")
                    rec_status = gr.Textbox(label="录音状态", lines=3, elem_id="status-record")
                    gr.HTML(
                        '<p id="import-hint">也可以把已有的短 WAV / FLAC 拖到下面，或点击导入。</p>',
                        show_label=False,
                    )
                    with gr.Row(elem_id="import-row"):
                        wav_file = gr.File(label="短音频文件", elem_id="wav-import")
                        rec_import = gr.Button("导入并用作训练素材", variant="secondary", elem_id="btn-import")

                with gr.Group(elem_id="stage-train"):
                    gr.HTML(_step2_html(), show_label=False)
                    train_name = gr.Textbox(label="模型名", value=DEFAULT_MODEL_NAME)
                    train_base = gr.Dropdown(
                        choices=base_labels,
                        value=base_default,
                        label="训练底模（可替换）",
                        elem_id="train-base",
                    )
                    with gr.Row(elem_id="base-actions"):
                        base_refresh = gr.Button("刷新底模", variant="secondary", elem_id="btn-refresh-base")
                    gr.HTML(
                        f'<p id="base-help">{placement_help(short=True)}</p>',
                        show_label=False,
                    )
                    with gr.Row(elem_id="base-import-row"):
                        base_g = gr.File(label="导入生成器 G（.pth）", file_types=[".pth"], elem_id="base-g")
                        base_d = gr.File(label="导入判别器 D（.pth）", file_types=[".pth"], elem_id="base-d")
                    with gr.Row(elem_id="base-import-actions"):
                        base_name = gr.Textbox(label="自定义底模名称（可选）", placeholder="例如 MyBetterV2")
                        base_import = gr.Button("导入到底模目录", variant="secondary", elem_id="btn-import-base")
                    base_status = gr.Textbox(label="底模状态", lines=2, elem_id="status-base", value="默认使用 RVC v2 40k（f0G40k / f0D40k）。")
                    with gr.Row(elem_id="train-actions"):
                        train_start = gr.Button("一键训练", variant="primary", elem_id="btn-train")
                        train_stop = gr.Button("中断训练", variant="stop", elem_id="btn-stop-train")
                    train_log = gr.Textbox(label="训练进度", lines=8, elem_id="status-train")

                with gr.Group(elem_id="stage-live"):
                    gr.HTML(STEP3_HTML, show_label=False)
                    live_meter = gr.HTML(_live_meter_html(), show_label=False, elem_id="live-meter")
                    rt_model = gr.Dropdown(choices=models, value=model_default, label="音色模型")
                    with gr.Row(elem_id="live-actions"):
                        rt_start = gr.Button("一键开始实时变声", variant="primary", elem_id="btn-live")
                        rt_stop = gr.Button("停止变声", variant="stop", elem_id="btn-stop-live")
                    rt_status_box = gr.Textbox(label="实时状态", lines=3, elem_id="status-live")

                gr.HTML(FOOT_HTML, show_label=False)

        with gr.Group(elem_id="settings-modal"):
            with gr.Group(elem_id="settings-panel"):
                gr.HTML(SETTINGS_HEAD_HTML, show_label=False)
                with gr.Group(elem_id="settings-system"):
                    gr.HTML(SYSTEM_HELP_HTML, show_label=False)
                    rec_device = gr.Dropdown(
                        choices=rec_choices,
                        value=rec_default,
                        label="录音设备（默认系统扬声器环回）",
                        elem_id="rec-device",
                    )
                    rt_out = gr.Dropdown(
                        choices=outs,
                        value=out_default,
                        label="输出 / 虚拟麦克风（Windows 常用 CABLE Input）",
                        elem_id="rt-out",
                    )
                    rt_in = gr.Dropdown(
                        choices=mics,
                        value=mic_default,
                        label="实时输入（麦克风，次要）",
                        elem_id="rt-in",
                    )
                    with gr.Row(elem_id="sys-actions"):
                        rt_refresh = gr.Button("刷新设备列表", variant="primary", elem_id="btn-refresh-devices")
                        rt_setup = gr.Button("配置 Linux 虚拟音频", variant="secondary")

                with gr.Group(elem_id="settings-advanced"):
                    gr.HTML(ADVANCED_HELP_HTML, show_label=False)
                    train_dir = gr.Textbox(label="训练音频目录", value=audio_dir_default)
                    with gr.Row():
                        train_epochs = gr.Number(label="训练轮数", value=GAME_TRAIN["total_epoch"], precision=0)
                        train_batch = gr.Number(label="批大小", value=GAME_TRAIN["batch_size"], precision=0)
                        train_slice = gr.Number(label="切分秒数", value=GAME_TRAIN["slice_sec"])
                    with gr.Row():
                        train_ver = gr.Dropdown(
                            choices=["v2", "v1"],
                            value=(base_pair.version if base_pair else GAME_TRAIN["version"]),
                            label="版本",
                        )
                        train_sr = gr.Dropdown(
                            choices=["40k", "48k", "32k"],
                            value=(base_pair.sample_rate if base_pair else GAME_TRAIN["sample_rate"]),
                            label="采样率",
                        )
                    rt_index = gr.Textbox(label="索引路径", value=_index_for(model_default or ""))
                    with gr.Row():
                        rt_pitch = gr.Number(label="音高（半音）", value=GAME_REALTIME["f0_up_key"], precision=0)
                        rt_buffer = gr.Number(label="缓冲 ms", value=GAME_REALTIME["buffer_ms"], precision=0)
                        rt_gain = gr.Number(label="额外增益 dB", value=GAME_REALTIME["gain_db"])
                        rt_irate = gr.Number(label="索引比例", value=GAME_REALTIME["index_rate"])
                    rt_pass = gr.Checkbox(label="直通（只验证路由，不变声）", value=False)
                    with gr.Row():
                        rt_poll = gr.Button("刷新实时状态", variant="secondary")
                        rt_sim = gr.Button("模拟实时（测试 WAV）", variant="secondary")


        rec_start.click(start_record, [rec_device, rec_duration, auth], [rec_status, rec_meter])
        rec_stop.click(stop_record, [], [rec_status, rec_meter])
        rec_import.click(import_wav, [wav_file, auth], [rec_status, rec_meter])
        rec_device.change(_on_device_change, [rec_device, rt_out], [device_hint])
        rt_out.change(_on_device_change, [rec_device, rt_out], [device_hint])
        train_start.click(
            start_train,
            [train_name, train_dir, train_epochs, train_batch, auth, train_ver, train_sr, train_slice, train_base],
            [train_log],
        )
        train_stop.click(stop_train, [], [train_log])
        train_base.change(_on_base_change, [train_base], [train_ver, train_sr])
        base_refresh.click(refresh_bases, [train_base], [train_base, base_status, train_ver, train_sr])
        base_import.click(import_bases, [base_g, base_d, base_name, train_base], [train_base, base_status, train_ver, train_sr])
        rt_model.change(lambda m: _index_for(m or ""), [rt_model], [rt_index])
        rt_refresh.click(
            refresh_devices,
            [rec_device, rt_in, rt_out, rt_model],
            [rec_device, rt_in, rt_out, rt_model, rt_index, rt_status_box, live_meter, device_hint],
        )
        rt_setup.click(setup_audio, [], [rt_status_box, live_meter])
        rt_start.click(
            start_rt,
            [rt_model, rt_index, rt_in, rt_out, rt_pitch, rt_buffer, rt_gain, rt_irate, rt_pass],
            [rt_status_box, live_meter],
        )
        rt_stop.click(stop_rt, [], [rt_status_box, live_meter])
        rt_poll.click(rt_status, [], [rt_status_box, live_meter])
        rt_sim.click(simulate_rt, [rt_model, rt_index, rt_buffer, rt_gain, rt_irate], [rt_status_box, live_meter])

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
