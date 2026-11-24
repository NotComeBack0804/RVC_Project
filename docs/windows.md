# Windows 使用要点（VB-CABLE + DirectML 显卡加速）

目标机建议：Windows + Python 3.12 x64 + VB-CABLE + AMD 显卡（例如 RX 7800 XT）。完整环境安装后，训练和实时变声默认走 DirectML 显卡；只有探测失败时才明确提示并回退 CPU。不要额外安装 CUDA。

推荐路径：**录制扬声器（1 或 2 分钟纯净发言）→ 一键训练 → 一键实时变声送到 VB-CABLE → 游戏/聊天软件选择 CABLE Output**。这是「游戏短音频变身」预设，面向鹅鸭杀、狼人杀这类发言局。请戴耳机。可随时把游戏麦克风切回真实麦克风。

## 开箱即用便携包（推荐）

不想安装 Python、也不想运行 `install_windows.ps1` 时，使用已打好的 Windows 便携包：解压后双击 `Start-UI.bat` 即可打开 http://127.0.0.1:7870 ，默认走 DirectML。完整说明见 [windows-portable.md](windows-portable.md)。

下面是从源码安装的步骤，开发或自行重建环境时使用。

## 安装

1. 安装 Python 3.12 x64，勾选加入 PATH。
2. 安装 [VB-CABLE](https://vb-audio.com/Cable/)，必要时重启。
3. 确认已安装较新的 AMD 显卡驱动（支持 DirectX 12，这是 DirectML 的前提）。
4. 在本仓库执行完整安装（会创建 `.venv-rvc`，并按官方 `requirments_cpu_py312.txt` 安装 `torch-directml` 与 `onnxruntime-directml`；同时下载含 `rmvpe.onnx` 的公开基础模型）：

```powershell
.\scripts\install_windows.ps1 -WithRvcCpu
```

`-WithRvc` 与 `-WithRvcCpu` 效果相同，是同一套官方 CPU/AMD 依赖，不是 CUDA。安装结束时会打印计算设备。要看到类似下面的内容，才说明走到了显卡：

```text
计算设备：DirectML（<适配器名>，privateuseone:0）。
  类型：DirectML
  设备名：privateuseone:0
  适配器：……（例如 AMD Radeon RX 7800 XT）
  RMVPE 需要 rmvpe.onnx：是
```

若打印的是 CPU，并附带回退原因（未安装 torch-directml、探测失败、onnxruntime 没有 DmlExecutionProvider 等），请先按提示检查驱动和安装，再重新运行上面的脚本。也可以随时自查：

```powershell
.\.venv-rvc\Scripts\python.exe -m rvc_realtime status
```

只想先验证音频直通、暂不装完整推理时：

```powershell
.\scripts\install_windows.ps1
.\.venv\Scripts\Activate.ps1
python -m rvc_realtime list-devices
```

需要训练自己的授权音色时，再下载官方底模（体积较大）：

```powershell
.\scripts\download_base_models.ps1 --with-pretrained
```

官方依赖文件名目前是 `requirments_cpu_py312.txt`。其中 `torch-directml==0.2.5.dev240914` 与 `onnxruntime-directml` 带有 `platform_system == "Windows"` 标记，只在 Windows 安装。Linux CPU 环境不会变成 CUDA 环境，也不要额外执行任何 CUDA 安装命令。

DirectML 上的 RMVPE 必须使用 `assets/rmvpe/rmvpe.onnx`（由 `onnxruntime-directml` 的 `DmlExecutionProvider` 执行）。Windows 上 `.\scripts\download_base_models.ps1` 会默认下载它。

## 启动本地中文界面

```powershell
.\scripts\start_ui.ps1
# 或
.\.venv-rvc\Scripts\python.exe -m rvc_realtime serve --host 127.0.0.1 --port 7870
```

`start_ui.ps1` 启动前会打印当前计算设备。用浏览器打开 http://127.0.0.1:7870 （只在本机监听，不要开 Gradio 公网分享）。首页就是「游戏短音频变身」，顶部会标明当前是 DirectML 还是 CPU。先勾选「这是已授权的声音」。

强制回退 CPU（排障用）：

```powershell
$env:RVC_COMPUTE_DEVICE = "cpu"
.\scripts\start_ui.ps1
# 或
.\.venv-rvc\Scripts\python.exe -m rvc_realtime serve --compute-device cpu
```

## 录制扬声器（主路径）

录的是电脑正在播放的声音，不是麦克风。界面会把「扬声器 / 系统音频 / 环回」类设备排在前面并设为默认：

- 优先：WASAPI 环回（默认播放设备），或已启用的「立体声混音」
- 不要默认去录 CABLE Input / CABLE Output，以免把变声结果再录进去
- 要换录音扬声器或输出/虚拟麦克风，打开左侧「系统设置」。麦克风作为次要备选，也在系统设置里

操作：选 1 分钟或 2 分钟 →「开始录制扬声器」→ 到时自动停止，也可提前「停止并保存」。保存到 `testdata\authorized_reference\`。没有环回时，用「导入已有短 WAV」。

录音时先不要开实时变声，并让游戏语音和 VB-CABLE 分开，避免环回把已经变声的声音再录进去。

## 一键训练与实时变声

模型名默认 `game_voice`。点「一键训练」即可，默认参数按 1–2 分钟干净说话预设：RVC v2、40k、60 轮、切分 3.0 秒、batch 1、rmvpe。轮数比官方常用的 200 轮低，减轻把那几句发言背下来，又尽量学到音色。DirectML 可用时，音高/特征提取走显卡，模型训练循环也会把张量放到 `privateuseone:0`。若训练日志里出现 DirectML 未实现的算子，先看报错，再用 `RVC_COMPUTE_DEVICE=cpu` 回退。界面显示进度，可以中断。

训练完成后点「一键开始实时变声」。默认音高 0、缓冲 160 ms（先稳再尽量低延迟）、响度与输入对齐，避免输出偏小。实时状态里会写明当前计算设备。直通、音高和缓冲在「系统设置」弹窗的「高级设置」页。训练页可以更换或导入训练底模（默认仍是 RVC v2 40k）。

实时路由：

- 本程序实时输入：你说话用的真实麦克风（系统设置里，不是扬声器环回）
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

变声示例（默认会在 DirectML 可用时走显卡）：

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

## 如何确认已经走到显卡

1. 安装结束或运行 `python -m rvc_realtime status`，计算设备应为 DirectML，设备名为 `privateuseone:0`（这是 `torch-directml` 的默认适配器）。
2. 启动界面后，首页顶部的「当前计算设备」应显示 DirectML 和适配器名，而不是 CPU。
3. 训练日志里，音高提取命令应类似 `extract_f0.py dml ...`，特征提取设备应包含 `privateuseone`，模型训练开始时应出现「训练设备：DirectML」。
4. 实时变声的状态或命令行输出会写明「计算设备：DirectML」。

如果以上任何一处写的是 CPU，以那条回退原因为准，不要假定已经在用 7800 XT。

## 官方实时 GUI（可选）

也可以激活 `.venv-rvc` 后，进入 `third_party\Retrieval-based-Voice-Conversion-WebUI` 运行 `python realtime_gui.py`（或 `go-realtime_gui.bat`）。官方配置在没有可用 NVIDIA GPU 时会自动选择 DirectML。输入/输出设备同样按上面的 CABLE 路由设置。本仓库一般优先使用 `.\scripts\start_ui.ps1`。
