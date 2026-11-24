# 本地实时语音转换

把**你本人的声音，或已获得明确授权的声音**，预先做成固定的 RVC 音色模型，再在聊天时把麦克风声音实时转换成目标音色，并作为虚拟麦克风送给聊天软件。

本项目在本地运行。参考音频和转换过程不要上传到云端。可以从 GitHub / Hugging Face 下载公开基础模型（HuBERT、RMVPE 等），但不要把用户语音或其音色模型传出去。

实时阶段做的是**语音转换（Voice Conversion）**：保留你的语音内容、节奏和大致语气，再换成目标音色。它不是把一段已经生成的声音文件循环播放。

使用变声通话时，建议明确告知对方声音经过了转换。请遵守聊天平台关于录音、声音数据和身份冒用的规定。不要使用未授权的他人声音。

## 总体链路

```text
授权的干净参考语音
        ↓
预先训练或准备 RVC 音色模型
        ↓
实时语音转换程序加载固定模型
        ↓
真实麦克风 → 实时语音转换 → 虚拟音频设备 → 聊天软件
```

在 Windows + AMD 显卡（例如 RX 7800 XT）上，完整环境默认用 DirectML 走显卡做训练和实时变声；DirectML 不可用时会明确提示并回退 CPU。Linux 开发环境继续使用 CPU。不要把这套环境改成 CUDA 安装。

## 推荐实施顺序

```text
安装虚拟音频（Linux 本工具脚本 / Windows VB-CABLE）
    ↓
验证：麦克风或测试音源 → 程序直通 → 虚拟设备 → 录音/聊天软件
    ↓
安装轻量 Python 环境（可选，仅直通）
    ↓
安装独立的完整 RVC 环境 .venv-rvc（Windows 含 DirectML，Linux 为 CPU）
    ↓
下载 HuBERT / RMVPE 公开基础模型（Windows 默认含 rmvpe.onnx）
    ↓
确认计算设备：status / 界面顶部应显示 DirectML 或 CPU 回退原因
    ↓
整理授权参考音频并训练自己的模型
    ↓
测量延迟并调节缓冲
```


## 使用本地界面（推荐）

完整 RVC 环境装好后，用本地中文界面完成预设 **「游戏短音频变声」**：录制扬声器（系统正在播放的声音）→ 一键训练 `.pth` → 一键实时变声。界面只监听本机 `127.0.0.1`，不会做 Gradio 公网分享，也不会上传录音或音色。

```bash
# Linux
./scripts/start_ui.sh
# 或
.venv-rvc/bin/python -m rvc_realtime serve --host 127.0.0.1 --port 7870

# Windows（PowerShell）
.\scripts\start_ui.ps1
# 或
.\.venv-rvc\Scripts\python.exe -m rvc_realtime serve --host 127.0.0.1 --port 7870
```

浏览器打开 http://127.0.0.1:7870 。首页就是「游戏短音频变声」一条龙，面向鹅鸭杀、狼人杀这类 1–2 分钟纯净发言。顶部会标明当前计算设备（Windows 上应为 DirectML，本开发盒子为 CPU）。左侧步骤栏可点击，分别进入录音、训练和实时变声：

1. 勾选「这是已授权的声音」。
2. 选择 1 分钟或 2 分钟，点 **「开始录制扬声器」**（到时自动停止，也可「停止并保存」）。录的是电脑正在播放的声音。Windows 默认优先 WASAPI 环回或立体声混音；Linux 默认 `speakers.monitor`。要换录音扬声器或输出/虚拟麦克风，点左侧 **「系统设置」**。麦克风作为次要输入，也在系统设置里。录音时先不要开实时变声，也不要从 CABLE 录，以免把变声结果再录进去。没有环回时，可导入已有短 WAV。
3. 模型名默认 `game_voice`，训练底模默认 RVC v2 40k（`f0G40k` / `f0D40k`），可在训练页更换或导入。点 **「一键训练」**。预设为 60 轮 / 切分 3.0 秒 / batch 1 / rmvpe，针对短干净说话，减轻背词。进度可看，可中断。产出 `assets/weights/game_voice.pth`。
4. 点 **「一键开始实时变声」**。默认音高 0、缓冲 160 ms、响度对齐输入。Windows：程序输出选 **CABLE Input**，游戏或聊天软件麦克风选 **CABLE Output**。请戴耳机。可随时切回真实麦克风。

轮数、音高、缓冲、路径和直通放在「系统设置」弹窗的 **「高级设置」** 页。那里的「模拟实时」只用测试 WAV 看处理耗时，**不能**当成已经用真实麦克风或游戏语音做了通话验收。

## 1. 安装

### 1.1 Linux

需要：`git`、`ffmpeg`、PulseAudio 或 PipeWire 的 Pulse 兼容层（`pactl` / `pacat` / `parec`）。官方 RVC 依赖面向 **Python 3.12 x64**。若系统没有 3.12，安装脚本会优先使用 `uv` 安装 3.12；找不到时再回退到系统 `python3`（3.10+ 可以跑直通，完整 RVC 仍建议 3.12）。

```bash
cd rvc-realtime-voice
./scripts/install_linux.sh
source .venv/bin/activate
python -m rvc_realtime status
python -m rvc_realtime setup-virtual-audio
python -m rvc_realtime self-test
```

下载公开基础模型（不是音色模型）：

```bash
./scripts/download_base_models.sh
```

完整 RVC CPU 推理（Linux；体积较大，请使用独立虚拟环境，不要混入 CUDA）：

```bash
./scripts/install_linux.sh --with-rvc-cpu
```

这会创建（或复用）独立的 `.venv-rvc`，并安装官方仓库中实际存在的 `requirments_cpu_py312.txt`（文件名以仓库为准，当前确认为这个拼写）。不要混入 CUDA。轻量 `.venv` 会被复用而不会被清空。

若使用 `uv` 安装该官方依赖文件，脚本会自动加上 `--index-strategy unsafe-best-match`。原因是该文件同时声明了多个 index，uv 默认只采用「某个包首次出现的源」，否则可能把 `tqdm` 等包卡在 PyTorch CPU 索引的旧版本上。这不会安装 CUDA。

### 1.2 Windows（目标使用环境：Windows + AMD RX 7800 XT）

若只想在 Windows 上使用，推荐解压开箱即用便携包，双击 `Start-UI.bat` 即可启动界面并默认走 AMD DirectML，不必自行安装 Python 或运行 `install_windows.ps1`。说明见 [`docs/windows-portable.md`](docs/windows-portable.md)。下面是从源码安装的步骤。

1. 安装 [Python 3.12 x64](https://www.python.org/downloads/)，安装时勾选 Add python.exe to PATH。
2. 安装 [VB-CABLE](https://vb-audio.com/Cable/)。本仓库不会在 Linux 上安装 Windows 驱动，也请不要在 Linux 盒子上强行安装它。
3. 建议同时安装 Git 与 FFmpeg。若使用官方 RVC GUI，也可按官方说明把 `ffmpeg.exe` / `ffprobe.exe` 放到官方仓库根目录。

```powershell
cd rvc-realtime-voice
.\scripts\install_windows.ps1
.\.venv\Scripts\Activate.ps1
python -m rvc_realtime status
python -m rvc_realtime list-devices
python -m rvc_realtime self-test --skip-pulse
.\scripts\download_base_models.ps1
```

需要完整推理和训练时，安装官方 CPU/AMD 依赖（Windows 会安装 DirectML，仅 Windows 生效；Linux 仍走 CPU）。这是 RX 7800 XT 的主路径，安装后默认走显卡：

```powershell
.\scripts\install_windows.ps1 -WithRvcCpu
```

安装结束时会打印计算设备。应看到 `DirectML` 和 `privateuseone:0`；若显示 CPU，请按回退原因检查 AMD 驱动或重新安装依赖。也可用 `.\.venv-rvc\Scripts\python.exe -m rvc_realtime status` 再确认。Windows 上 `download_base_models.ps1` 会默认下载 `rmvpe.onnx`（DirectML 的 RMVPE 必需）。训练自己的授权音色还需要：`.\scripts\download_base_models.ps1 --with-pretrained`。

更完整的安装、启动和确认步骤见 [docs/windows.md](docs/windows.md)。系统依赖也可以对照官方 README：不要额外执行 CUDA 安装命令。

## 2. 音频路由

### 2.1 先验证直通，再加载模型

在训练或变声之前，先确认这条路是通的：

```text
真实麦克风或测试音源
    ↓
本程序（--mode passthrough）
    ↓
虚拟线缆输入
    ↓
虚拟线缆输出
    ↓
录音软件 / 聊天软件麦克风
```

只有直通正常后，才在 `--mode convert` 下加载已授权的 `.pth`。

### 2.2 Linux 设备名称

运行 `python -m rvc_realtime setup-virtual-audio` 后：

| 角色 | 设备名 | 说明 |
| --- | --- | --- |
| 程序输出 | `RVC_Cable_In` | 虚拟沉，相当于 Windows 的 CABLE Input |
| 聊天软件麦克风 | `RVC_Cable_Out` | 由 `RVC_Cable_In.monitor` 映射的源，相当于 CABLE Output |
| 也可直接监听 | `RVC_Cable_In.monitor` | 同一路信号 |
| 测试音源沉 | `RVC_TestSrc` | 没有真实麦克风时，可把 WAV 播到这里，再从 `RVC_TestSrc.monitor` 读取 |

如果机器上没有真实麦克风（例如本开发用的 Linux 盒子），请用测试 WAV 或 `RVC_TestSrc` 做端到端验证，不要假定已经接好了话筒。

若 `pactl load-module` 失败（某些环境会以 `--disallow-module-loading` 启动 PulseAudio），脚本会自动改用已有空沉，例如把 `speakers` 当作程序输出、把 `speakers.monitor` 当作虚拟麦克风，并在命令输出中写明实际设备名。

示例：

```bash
python -m rvc_realtime generate-test-signal
python -m rvc_realtime run --mode passthrough \
  --input testdata/generated/test_tone.wav \
  --output testdata/recorded/passthrough.wav

# 实时：把转换结果送到虚拟线缆，供聊天软件使用
python -m rvc_realtime run --mode passthrough \
  --input <你的麦克风或 RVC_TestSrc.monitor> \
  --output RVC_Cable_In \
  --buffer-ms 150
```

聊天软件选择 `RVC_Cable_Out` 作为麦克风。请使用耳机监听，避免扬声器声音重新进入麦克风形成回声或啸叫。保留真实麦克风作为故障时的回退输入。

### 2.3 Windows 设备名称（VB-CABLE）

设备名称可能因系统语言或驱动版本略有不同，以 Windows 声音设置中的实际名称为准。常见名称是：

```text
真实麦克风
    ↓
本程序输入
    ↓
RVC 转换或直通
    ↓
CABLE Input
    ↓
CABLE Output
    ↓
聊天软件麦克风
```

设置步骤：

1. 安装 VB-CABLE。
2. 用 `python -m rvc_realtime list-devices` 确认能看到 CABLE 设备。
3. 程序输入选真实麦克风，输出选 `CABLE Input`。
4. 聊天软件麦克风选 `CABLE Output`。
5. 使用耳机；保留真实麦克风作为回退。

```powershell
python -m rvc_realtime run --mode passthrough `
  --input "你的麦克风名称" `
  --output "CABLE Input" `
  --buffer-ms 150
```

也可以先把测试 WAV 跑一遍文件直通，再接入麦克风。

### 2.4 官方实时 GUI（可选）

完整 RVC 环境安装完成后，也可以使用官方实时界面（Windows 上为 `go-realtime_gui.bat`，源码入口 `realtime_gui.py`）。把已授权 `.pth` 放到官方文档指定的 `assets/weights/`，或继续使用本仓库的 `assets/weights/` 并在 GUI 中浏览该路径。Windows 上官方配置在没有 NVIDIA GPU 时会自动选择 DirectML。

## 3. 模型如何放置

```text
assets/
├── hubert_base/          # 公开 HuBERT（download 脚本下载）
├── rmvpe/rmvpe.pt        # 公开 RMVPE
├── rmvpe/rmvpe.onnx      # Windows DirectML 的 RMVPE 必需；Windows 下载脚本默认获取
├── weights/              # 你自己的已授权 RVC .pth
└── indices/              # 可选的检索索引 .index
```

- 用户自己的音色模型：`assets/weights/*.pth`
- 索引：`assets/indices/*.index`
- 本仓库不附带任何他人音色模型。没有 `.pth` 时，`--mode convert` 会明确报告缺少模型并退出，不会崩溃。直通模式不依赖这些文件。

检查状态：

```bash
python -m rvc_realtime status
```

## 4. 启动变声

```bash
# 先直通
python -m rvc_realtime run --mode passthrough --input <麦克风> --output RVC_Cable_In

# 再变声（需要已授权 .pth、基础模型，以及 .venv-rvc 中的完整依赖）
source .venv-rvc/bin/activate   # Windows: .\.venv-rvc\Scripts\Activate.ps1
python -m rvc_realtime run --mode convert \
  --input <麦克风> \
  --output RVC_Cable_In \
  --model assets/weights/你的模型.pth \
  --index assets/indices/你的索引.index \
  --f0-up-key 0 \
  --buffer-ms 150
```

建议先朗读一段固定文本，检查音色、吐字、断音、爆音和延迟，再进入短时间测试通话。

在本仓库开发盒子上已用授权测试模型 Shadowww T02（CC BY-NC 4.0，仅本机测试）完成文件到文件的变声验收。请在项目根目录使用 `.venv-rvc` 中的解释器（不要用轻量 `.venv`）：

```bash
cd /workspace/rvc-realtime-voice
.venv-rvc/bin/python -m rvc_realtime run --mode convert \
  --input testdata/generated/speech_jfk.wav \
  --output testdata/recorded/shadowww-t02_convert_v2.wav \
  --model assets/weights/shadowww-t02.pth \
  --index assets/indices/added_shadowww-t02.index \
  --index-rate 0.5 \
  --f0-method rmvpe \
  --f0-up-key 0 \
  --buffer-ms 1000 \
  --sample-rate 48000 \
  --channels 1
```

请不要覆盖已有的 `shadowww-t02_convert.wav`。优化后的对比输出是 `testdata/recorded/shadowww-t02_convert_v2.wav`（响度对齐输入，按帧融合索引）。这里的 `--buffer-ms 1000` 是为文件转换减少推理次数，不是实时通话推荐值。实时使用请从 100–200 ms 调起，并加上 `--loudness-mode match-input`（默认已开启）。输入语音是公有领域的 JFK 演说测试片段（来源见 `testdata/generated/SPEECH_SOURCE.txt`），不是目标音色。转换结果只留在本机，不要再发布到公网。实测记录见 `VERIFICATION.md`。

## 5. 延迟与缓冲

实时通话应以实际测量为准。建议顺序：

1. 先保证没有爆音和断续。
2. 在稳定的基础上逐步减小 `--buffer-ms`。
3. 关闭不必要的后台高负载程序。
4. 检查聊天软件的降噪、自动增益和回声消除是否破坏音质。
5. 使用耳机，避免声学回声。

单向端到端延迟的控制目标约 **100–200 ms**。若明显超过约 **250 ms**，聊天时可能出现明显迟滞。实际数值取决于 CPU、模型、音高提取器、音频驱动、缓冲区和聊天软件处理。程序会打印估计的缓冲延迟（输入+输出缓冲），这还不包含驱动和聊天软件额外开销。

## 6. 训练自己的授权音色

训练必须在确认「这是已授权录音」之后才能开始：界面勾选授权，或命令行加 `--i-have-authorization`。未确认会被拒绝（退出码 2）。不要用未授权的他人声音，也不要把录音或新训练的音色上传到云端。

推荐流程：本地界面录制扬声器（或导入已有短 WAV）→「一键训练」→「一键开始实时变声」。

命令行等价示例：

```bash
# 1. 把本人或已授权录音放到 testdata/authorized_reference/（只要音频文件）
# 2. 确认授权后再训练。先用 1–5 轮冒烟，再按质量需要增加轮数。
.venv-rvc/bin/python -m rvc_realtime train \
  --audio-dir testdata/authorized_reference \
  --name myvoice \
  --i-have-authorization \
  --version v2 --sr 40k \
  --epochs 10 --batch-size 1 --f0-method rmvpe
```

产出：

- 模型：`assets/weights/myvoice.pth`
- 索引：`assets/indices/added_myvoice.index`（以及官方命名的 `added_IVF…` 文件）

本开发盒子（Linux CPU）耗时很粗的经验：约 11 秒公有领域测试音频、1 轮，大约 1 分钟内可以跑完切分、特征、训练和索引（其中 1 轮训练约 23 秒）。在 Windows + RX 7800 XT 上，DirectML 通常会更快，但仍以本机日志为准。若回退到 CPU，1–3 分钟录音、数轮训练往往要数十分钟；10–30 分钟语料、上百轮可能需要数小时。请在界面查看进度和计算设备，必要时点「中断训练」。

训练还需要官方底模和 mute 样本。若缺失，运行：

```bash
./scripts/download_base_models.sh --with-pretrained
```

也可以把更好的底模对（`f0G40k.pth` 与 `f0D40k.pth` 这类成对文件）放到 `assets/pretrained_v2/` 或 `assets/pretrained_custom/`，在界面训练区选择并刷新即可，不必改代码。约定见 [docs/pretrained-bases.md](docs/pretrained-bases.md)。


`testdata/train_smoke_jfk/` 里的 JFK 片段只用于验证训练链路能写出权重，说明见 `testdata/TRAIN_SMOKE_JFK.md`。不要把它当成对外使用的音色，也不要用 Shadowww 的声音再训一个「新音色」去冒充。

## 7. 常见问题

### 对方听不到声音

检查聊天软件是否选择了 `CABLE Output`（Windows）或 `RVC_Cable_Out`（Linux），以及本程序是否确实输出到 `CABLE Input` / `RVC_Cable_In`。先用直通 + 本地录音确认链路。

### 声音断续或爆音

增大 `--buffer-ms`，关闭其他高负载应用，必要时降低模型或音高提取设置，并确认所有设备采样率一致（本程序默认 48 kHz）。

### 出现回声或啸叫

使用耳机，关闭不必要的监听回路，确认转换输出没有被重新选作转换输入。

### 变声模式立刻退出

这是预期行为：缺少已授权 `.pth`、缺少 HuBERT/RMVPE，或尚未安装 `.venv-rvc` 时，程序会打印原因并退出，而不会崩溃。请先跑通直通，再按提示补齐模型或依赖。

### 音色不稳定

检查参考音频是否含噪声、混响、多人声音或过度处理。优先换用更干净、录音条件更一致的素材。

### 聊天软件中的声音很闷或失真

逐项测试其降噪、自动增益和回声消除。某些聊天软件会对虚拟麦克风进行二次处理。

### 转换后声音偏小

默认已开启 `--loudness-mode match-input`：按输入 RMS 做响度对齐，并把峰值限制在约 0.95，避免削波。界面实时页也可调节输出音量。若仍偏小，可适当增加增益，但不要把峰值顶满。

### 延迟太高

先确认界面或 `status` 显示的计算设备。Windows 上若仍是 CPU，按回退原因检查 DirectML 是否可用。已走到 DirectML 时，先减小其他后台负载，再逐步减小 `--buffer-ms`。

## 8. Windows + AMD RX 7800 XT（DirectML 主路径）

目标使用环境是 Windows + RX 7800 XT。完整安装后，训练和实时变声默认走 DirectML，不再把显卡加速留到“以后再说”。

1. 运行 `.\scripts\install_windows.ps1 -WithRvcCpu`（不要安装 CUDA）。官方依赖会在 Windows 上安装 `torch-directml` 与 `onnxruntime-directml`。
2. 安装脚本会探测计算设备，并默认下载 `rmvpe.onnx`。DirectML 上的 RMVPE 必须用该文件，设备名是 `privateuseone:0`。
3. 用 `python -m rvc_realtime status` 或界面顶部的「当前计算设备」确认显示 DirectML 和适配器名。
4. 按「游戏短音频变声」录制、训练、实时变声。可用 `RVC_COMPUTE_DEVICE=cpu` 强制回退。
5. Linux 开发盒子没有这块显卡，也没有 DirectML，会保持 CPU 回退；这是预期行为。
6. 不要把 CUDA 配置照搬到 AMD 环境。若使用 Linux 本机 AMD 显卡，官方路径仍是 CPU，而不是 ROCm 或 CUDA。

显卡能否把延迟压到 100–200 ms，取决于驱动、模型、缓冲和聊天软件，需要在目标机上实测，不能只根据 RX 7800 XT 这个型号推断。详细步骤见 [docs/windows.md](docs/windows.md)。

## 9. 开发盒子上的限制

本仓库可在无真实麦克风的 Linux 环境中用测试 WAV + 虚拟线缆做端到端直通验证。该环境不能代替你在 Windows 目标机上用真实麦克风和 VB-CABLE、再对接聊天软件的验收。盒上自检记录见 `testdata/recorded/selftest_report.json` 与 `VERIFICATION.md`。

开发盒子上的额外限制：

- 没有真实麦克风。预置的 `mic` 是空的静音源，不能当作话筒。
- 系统 PulseAudio 使用 `--disallow-module-loading`，因此无法新建 `RVC_Cable_In`。自检已改用预置空沉 `speakers` 和监听源 `speakers.monitor`，并成功录到测试音。
- 轻量直通环境在 `.venv`。完整 RVC 推理环境已在 `.venv-rvc` 装好（Python 3.12.16，`torch 2.4.1+cpu`，CUDA 不可用）。本开发盒子是 Linux，计算设备为 CPU 回退。Windows 目标机上同一套依赖会安装 DirectML 并默认走显卡。变声请用 `.venv-rvc` 里的解释器。
- 已用授权测试模型 `assets/weights/shadowww-t02.pth` 完成文件到文件变声：保留 `testdata/recorded/shadowww-t02_convert.wav`（优化前），对比文件为 `shadowww-t02_convert_v2.wav`（优化后，响度与输入对齐，索引按帧融合不再整块跳过）。
- 实时路径在盒子上做了「模拟实时」：用测试 WAV 按 200 ms 块喂入，写出 `testdata/recorded/simulate_realtime.wav`。这不是真实麦克风或聊天软件验收。
- 本地界面可在 `127.0.0.1:7870` 启动，首页是「游戏短音频变声」。盒子上默认录音设备是 `speakers.monitor`（扬声器环回）。若环回是静音或不可用，会明确提示，并可导入已有短 WAV。
- 已用公有领域 JFK 片段做 1 轮冒烟训练，导出 `assets/weights/smoke_jfk_pd.pth`（仅验证训练链路，不是可用音色产品）。
- 盒子上没有真实麦克风，因此未做实时通话验收，也不能代替 Windows + VB-CABLE 对接聊天软件的测试。随时可以把聊天软件的麦克风切回真实麦克风。

## 10. 常用命令速查

```bash
# 本地中文界面（须使用 .venv-rvc）
./scripts/start_ui.sh
# 或：.venv-rvc/bin/python -m rvc_realtime serve --host 127.0.0.1 --port 7870

.venv-rvc/bin/python -m rvc_realtime status
.venv-rvc/bin/python -m rvc_realtime list-devices
.venv-rvc/bin/python -m rvc_realtime setup-virtual-audio
.venv-rvc/bin/python -m rvc_realtime generate-test-signal
.venv-rvc/bin/python -m rvc_realtime run --mode passthrough --input <文件或设备> --output <文件或设备>
.venv-rvc/bin/python -m rvc_realtime run --mode convert --input <麦克风> --output RVC_Cable_In --model assets/weights/xxx.pth --buffer-ms 160
# 文件到文件的变声（不要覆盖旧的 shadowww-t02_convert.wav）：
# .venv-rvc/bin/python -m rvc_realtime run --mode convert --input testdata/generated/speech_jfk.wav \
#   --output testdata/recorded/shadowww-t02_convert_v2.wav --model assets/weights/shadowww-t02.pth \
#   --index assets/indices/added_shadowww-t02.index --index-rate 0.5 --buffer-ms 1000 --loudness-mode match-input
# 无麦克风时的模拟实时验收：
# .venv-rvc/bin/python -m rvc_realtime run --mode convert --input testdata/generated/speech_jfk.wav \
#   --output testdata/recorded/simulate_realtime.wav --model assets/weights/shadowww-t02.pth \
#   --index assets/indices/added_shadowww-t02.index --buffer-ms 200 --simulate-realtime --loudness-mode match-input
# 授权录音训练（必须加 --i-have-authorization）：
# .venv-rvc/bin/python -m rvc_realtime train --audio-dir testdata/authorized_reference --name myvoice --i-have-authorization --epochs 10
.venv-rvc/bin/python -m rvc_realtime self-test
```
