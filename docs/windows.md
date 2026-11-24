# Windows 使用要点（VB-CABLE + CPU 起步）

目标机建议：Windows + Python 3.12 x64 + VB-CABLE。AMD RX 7800 XT 先不要作为主加速路径，CPU 跑通后再开独立环境试 DirectML。

推荐路径：**录制扬声器（1 或 2 分钟纯净发言）→ 一键训练 → 一键实时变声送到 VB-CABLE → 游戏/聊天软件选择 CABLE Output**。这是「游戏短音频变身」预设，面向鹅鸭杀、狼人杀这类发言局。请戴耳机。可随时把游戏麦克风切回真实麦克风。

## 安装

1. 安装 Python 3.12 x64，勾选加入 PATH。
2. 安装 [VB-CABLE](https://vb-audio.com/Cable/)，必要时重启。
3. 在本仓库执行：

```powershell
.\scripts\install_windows.ps1
.\.venv\Scripts\Activate.ps1
python -m rvc_realtime list-devices
.\scripts\download_base_models.ps1
```

完整推理和训练（独立环境 `.venv-rvc`，不要混入 CUDA）：

```powershell
.\scripts\install_windows.ps1 -WithRvcCpu
.\scripts\download_base_models.ps1 -WithPretrained
```

官方依赖文件名目前是 `requirments_cpu_py312.txt`。其中 `torch-directml` 与 `onnxruntime-directml` 带有 `platform_system == "Windows"` 标记，只在 Windows 安装，不会把 Linux CPU 环境变成 CUDA 环境。仍然不要额外执行任何 CUDA 安装命令。

## 启动本地中文界面

```powershell
.\scripts\start_ui.ps1
# 或
.\.venv-rvc\Scripts\python.exe -m rvc_realtime serve --host 127.0.0.1 --port 7870
```

用浏览器打开 http://127.0.0.1:7870 （只在本机监听，不要开 Gradio 公网分享）。首页就是「游戏短音频变身」，先勾选「这是已授权的声音」。

## 录制扬声器（主路径）

录的是电脑正在播放的声音，不是麦克风。界面会把「扬声器 / 系统音频 / 环回」类设备排在前面并设为默认：

- 优先：WASAPI 环回（默认播放设备），或已启用的「立体声混音」
- 不要默认去录 CABLE Input / CABLE Output，以免把变声结果再录进去
- 麦克风只放在折叠的「高级设置」里，作为次要备选

操作：选 1 分钟或 2 分钟 →「开始录制扬声器」→ 到时自动停止，也可提前「停止并保存」。保存到 `testdata\authorized_reference\`。没有环回时，用「导入已有短 WAV」。

录音时先不要开实时变声，并让游戏语音和 VB-CABLE 分开，避免环回把已经变声的声音再录进去。

## 一键训练与实时变声

模型名默认 `game_voice`。点「一键训练」即可，默认参数按 1–2 分钟干净说话预设：RVC v2、40k、60 轮、切分 3.0 秒、batch 1、rmvpe。轮数比官方常用的 200 轮低，减轻把那几句发言背下来，又尽量学到音色。CPU 上可能需要数十分钟到约两小时，界面显示进度，可以中断。

训练完成后点「一键开始实时变声」。默认音高 0、缓冲 160 ms（先稳再尽量低延迟）、响度与输入对齐，避免输出偏小。直通、麦克风选择、音高和缓冲都在高级设置里。

实时路由：

- 本程序实时输入：你说话用的真实麦克风（高级设置里，不是扬声器环回）
- 本程序输出：`CABLE Input`（有时显示为 `CABLE Input (VB-Audio Virtual Cable)`）
- 游戏或聊天软件麦克风：`CABLE Output`

请戴耳机。需要立刻恢复时，把游戏麦克风改回真实麦克风，并停止本程序的实时输出。

命令行直通（排障）示例：

```powershell
.\.venv-rvc\Scripts\python.exe -m rvc_realtime run --mode passthrough `
  --input "你的麦克风名称" `
  --output "CABLE Input" `
  --buffer-ms 160
```

变声示例：

```powershell
.\.venv-rvc\Scripts\python.exe -m rvc_realtime run --mode convert `
  --input "你的麦克风名称" `
  --output "CABLE Input" `
  --model assets\weights\game_voice.pth `
  --index assets\indices\added_game_voice.index `
  --buffer-ms 160 `
  --loudness-mode match-input
```

命令行训练（必须加授权开关；下面是与界面预设接近的参数）：

```powershell
.\.venv-rvc\Scripts\python.exe -m rvc_realtime train `
  --audio-dir testdata\authorized_reference `
  --name game_voice `
  --i-have-authorization `
  --version v2 --sr 40k `
  --epochs 60 --batch-size 1 --f0-method rmvpe
```

未勾选授权或未加 `--i-have-authorization` 时，训练会被拒绝。不要把录音或新训练的音色上传到云端，也不要下载、加载未授权的他人音色。

## 官方实时 GUI（可选）

也可以激活 `.venv-rvc` 后，进入 `third_party\Retrieval-based-Voice-Conversion-WebUI` 运行 `python realtime_gui.py`（或 `go-realtime_gui.bat`）。输入/输出设备同样按上面的 CABLE 路由设置。本仓库一般优先使用 `.\scripts\start_ui.ps1`。
