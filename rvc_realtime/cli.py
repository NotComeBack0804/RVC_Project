from __future__ import annotations

import argparse
import sys

from rvc_realtime.errors import AppError, UsageError
from rvc_realtime.paths import project_root, testdata_dir


USAGE_BANNER = (
    "本工具只用于你本人的声音，或已获得明确授权的声音。"
    "请勿用它冒用他人音色。变声通话时建议明确告知对方。"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rvc_realtime",
        description="本地实时语音转换（CPU 起步）。先验证直通，再加载已授权的 RVC 模型。",
        epilog=USAGE_BANNER,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="查看环境、基础模型和用户模型状态")
    sub.add_parser("list-devices", help="列出输入/输出设备（PulseAudio / sounddevice）")

    p_setup = sub.add_parser("setup-virtual-audio", help="在 Linux 上创建虚拟线缆（等价 VB-CABLE）")
    p_setup.add_argument("--teardown", action="store_true", help="卸载本工具创建的虚拟设备")
    p_setup.add_argument("--sample-rate", type=int, default=48000)
    p_setup.add_argument("--channels", type=int, default=1)

    p_gen = sub.add_parser("generate-test-signal", help="生成用于直通验证的测试 WAV")
    p_gen.add_argument("--output", default=str(testdata_dir() / "generated" / "test_tone.wav"))
    p_gen.add_argument("--duration", type=float, default=4.0)
    p_gen.add_argument("--sample-rate", type=int, default=48000)

    p_run = sub.add_parser("run", help="运行直通或变声")
    p_run.add_argument("--mode", choices=("passthrough", "convert"), default="passthrough")
    p_run.add_argument("--input", required=True, help="输入文件或 Pulse/设备名，例如 testdata/generated/test_tone.wav 或 mic / RVC_TestSrc.monitor")
    p_run.add_argument("--output", required=True, help="输出文件或设备名，例如 testdata/recorded/out.wav 或 RVC_Cable_In / CABLE Input")
    p_run.add_argument("--sample-rate", type=int, default=48000)
    p_run.add_argument("--channels", type=int, default=1)
    p_run.add_argument("--buffer-ms", type=int, default=100, help="音频缓冲，建议先 100–200 ms")
    p_run.add_argument("--duration", type=float, default=None, help="实时输入时的最长秒数；文件输入默认跑完整段")
    p_run.add_argument("--model", default=None, help="已授权 RVC .pth 路径")
    p_run.add_argument("--index", default=None, help="可选的检索索引 .index 路径")
    p_run.add_argument("--f0-up-key", type=int, default=0, help="音高移动（半音）")
    p_run.add_argument("--formant", type=float, default=0.0)
    p_run.add_argument("--index-rate", type=float, default=0.5, help="检索索引融合比例，0 表示不使用索引")
    p_run.add_argument("--f0-method", choices=("rmvpe", "pm", "fcpe"), default="rmvpe")
    p_run.add_argument("--loudness-mode", choices=("match-input", "off"), default="match-input", help="match-input 会把输出响度对齐到输入并限制峰值，避免明显偏小")
    p_run.add_argument("--gain-db", type=float, default=0.0, help="在自动对齐之外再叠加的增益（dB）")
    p_run.add_argument("--peak-limit", type=float, default=0.95)
    p_run.add_argument("--simulate-realtime", action="store_true", help="按块时长等待，用于无麦克风时的模拟实时验收")

    p_test = sub.add_parser("self-test", help="在本机验证直通、虚拟设备和缺模型报错")
    p_test.add_argument("--skip-pulse", action="store_true")

    p_serve = sub.add_parser("serve", help="启动本地中文界面（录音、训练、实时变声）")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=7870)

    p_train = sub.add_parser("train", help="用已授权录音训练并导出 .pth / 索引")
    p_train.add_argument("--audio-dir", required=True)
    p_train.add_argument("--name", required=True, help="导出模型名，例如 myvoice")
    p_train.add_argument("--i-have-authorization", action="store_true")
    p_train.add_argument("--version", choices=("v1", "v2"), default="v2")
    p_train.add_argument("--sr", choices=("32k", "40k", "48k"), default="40k")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--f0-method", choices=("rmvpe", "pm"), default="rmvpe")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    print(USAGE_BANNER, flush=True)
    try:
        if args.command == "status":
            from rvc_realtime.status import gather_status

            print(gather_status())
            return 0

        if args.command == "list-devices":
            from rvc_realtime.devices import format_device_list, list_all_devices

            devices, notes = list_all_devices()
            print(format_device_list(devices, notes))
            return 0

        if args.command == "setup-virtual-audio":
            from rvc_realtime import pulse

            if args.teardown:
                removed = pulse.teardown_virtual_audio()
                print("已尝试卸载虚拟设备模块：" + (", ".join(removed) if removed else "没有找到可卸载模块"))
                return 0
            info = pulse.setup_virtual_audio(sample_rate=args.sample_rate, channels=args.channels)
            print("虚拟音频已就绪：")
            for key, value in info.items():
                print(f"  {key} = {value}")
            print()
            if info.get("mode") == "named":
                print(pulse.describe_routing())
                print("聊天软件请选择 RVC_Cable_Out 作为麦克风。")
            else:
                print("本机未能创建命名线缆，已复用已有空沉。请按上面的 cable_in / cable_out 来设置程序和聊天软件。")
                print(f"  程序输出设备：{info.get('cable_in')}")
                print(f"  聊天软件麦克风：{info.get('cable_out')}")
                print("在普通 Linux 桌面上（PulseAudio 允许 load-module 时），会创建 RVC_Cable_In / RVC_Cable_Out。")
                print("Windows 请使用 VB-CABLE：程序输出选 CABLE Input，聊天软件选 CABLE Output。")
            return 0

        if args.command == "generate-test-signal":
            from pathlib import Path
            from rvc_realtime.signal import analyze_wav, generate_test_tone

            path = generate_test_tone(Path(args.output), sample_rate=args.sample_rate, duration=args.duration)
            info = analyze_wav(path)
            print(f"已生成测试音：{path}")
            print(f"时长={info['duration_sec']:.3f}s  RMS={info['rms']:.4f}  主频={info['dominant_freq_hz']}")
            return 0

        if args.command == "run":
            from rvc_realtime.engine import format_run_result, run_engine

            result = run_engine(
                mode=args.mode,
                input_spec=args.input,
                output_spec=args.output,
                sample_rate=args.sample_rate,
                channels=args.channels,
                buffer_ms=args.buffer_ms,
                duration=args.duration,
                model=args.model,
                index=args.index,
                f0_up_key=args.f0_up_key,
                formant=args.formant,
                index_rate=args.index_rate,
                f0_method=args.f0_method,
                loudness_mode=args.loudness_mode,
                gain_db=args.gain_db,
                peak_limit=args.peak_limit,
                simulate_realtime=args.simulate_realtime,
            )
            print(format_run_result(result))
            return 0

        if args.command == "self-test":
            from rvc_realtime.selftest import format_report, run_selftest

            report = run_selftest(skip_pulse=args.skip_pulse)
            print(format_report(report))
            return 0 if report.get("passed") else 1

        if args.command == "serve":
            from rvc_realtime.app import launch_app

            launch_app(host=args.host, port=args.port, share=False)
            return 0

        if args.command == "train":
            from rvc_realtime.train_pipeline import run_training

            result = run_training(
                audio_dir=args.audio_dir,
                name=args.name,
                authorized=args.i_have_authorization,
                version=args.version,
                sample_rate=args.sr,
                total_epoch=args.epochs,
                batch_size=args.batch_size,
                f0_method=args.f0_method,
            )
            print(result.get("message", "训练结束"))
            return 0 if result.get("ok") else 1

        raise UsageError(f"未知命令：{args.command}")
    except AppError as exc:
        print(f"错误：{exc.message}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return 130
