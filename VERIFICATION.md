# 开发盒子实测记录

本记录只写实际跑过的结果，未跑通的步骤不会写成已经成功。时区均为 Asia/Shanghai（CST，UTC+8）。

## 2026-11-24 09:37–09:38：直通与缺模型路径

测试目录：`/workspace/rvc-realtime-voice`
解释器：Python 3.12.16（由 `uv` 安装；系统自带仅为 Python 3.13.5）
虚拟环境：`.venv`（轻量依赖：numpy 2.5.5、soundfile 0.14.0、huggingface_hub）

已通过：

1. **环境创建**：`uv venv --python 3.12 .venv` 成功。轻量依赖安装成功。
2. **官方仓库**：已浅克隆 `third_party/Retrieval-based-Voice-Conversion-WebUI`。当前 CPU 依赖文件名确认为 `requirments_cpu_py312.txt`。
3. **无真实麦克风**：系统预置 Pulse 源 `mic` 是 `module-null-source`，输出静音，不能当作话筒。
4. **虚拟音频回退可用**：系统 PulseAudio 以 `--disallow-module-loading` 启动，`pactl load-module` 失败。脚本自动改用已有空沉 `speakers` / 监听源 `speakers.monitor`。
5. **文件直通**：测试音 4.000 s 原样写出，RMS=0.1762，与源一致。
6. **虚拟设备直通**：从 `speakers.monitor` 录回约 3.680 s，有声，主频约 439.95 Hz。
7. **缺模型变声路径**：当时 `assets/weights/` 为空时，`--mode convert` 以退出码 2 结束，打印中文说明，没有崩溃，也没有写出输出文件。
8. **公开基础模型**：HuBERT 与 `assets/rmvpe/rmvpe.pt` 已下载。未下载他人音色模型。`rmvpe.onnx` 未下载（Linux CPU 路径不需要；Windows DirectML 的 RMVPE 必需，由 Windows 下载脚本默认获取）。
9. **训练骨架**：无授权确认或无录音时拒绝开始训练。

`python -m rvc_realtime self-test` 当时总体结果：通过。原始 JSON 见 `testdata/recorded/selftest_report.json`。

## 2026-11-24 10:02–10:06：完整 RVC CPU 环境与变声验收

### 环境安装

- 命令：`./scripts/install_linux.sh --with-rvc-cpu`
- 安装脚本已改为：若 `.venv` / `.venv-rvc` 已存在则复用，不会 `--clear` 掉已能跑直通的轻量环境。
- 首次用 `uv` 安装官方 `requirments_cpu_py312.txt` 时，因 uv 默认「首次出现的源优先」，在 PyTorch CPU 索引上解析 `tqdm>=4.67.0,<5` 失败。脚本随后加上 `--index-strategy unsafe-best-match`（接近 pip 的多源检索，不安装 CUDA）后安装成功。
- 未混入 CUDA 安装命令。
- 未停止 JiSuan 预览进程（`jisuan-api` 当时 RSS 约 41 MB，监听 3000 端口）。也未杀掉 Chrome。安装前后可用内存从约 2.4 GiB 变为约 6.1 GiB（页面缓存回收），足够完成本次推理。

`.venv-rvc` 关键包（Python 3.12.16）：

- `torch==2.4.1+cpu`
- `torchaudio==2.4.1+cpu`
- `torchvision==0.19.1+cpu`
- `numpy==1.26.4`
- `onnxruntime==1.32.0`
- `faiss-cpu==1.16.0`
- `librosa==0.10.2.post1`
- `soundfile==0.14.0`
- `transformers==4.49.0`

验证：`torch.cuda.is_available() == False`。轻量 `.venv` 仍可导入（numpy 2.5.5），直通环境未被破坏。

公开基础模型在本次安装前已经就绪，未再重复下载。为匹配官方代码中的相对路径，将 `third_party/Retrieval-based-Voice-Conversion-WebUI/assets/{hubert_base,rmvpe,weights,indices}` 符号链接到项目根下对应目录。

### 变声测试（文件到文件，不是直通）

授权测试模型（说话人本人发布，许可 CC BY-NC 4.0；仅本机测试，未上传云端，不要再把结果发到公网）：

- `assets/weights/shadowww-t02.pth`（约 54.9 MiB）
- `assets/indices/added_shadowww-t02.index`（约 58.8 MiB）

输入：`testdata/generated/speech_jfk.wav`（11.000 s、48 kHz、单声道、有声，RMS=0.1421，峰值=0.7836）。来源是 OpenAI Whisper 测试音频中的 JFK 就职演说片段（美国政府作品，公有领域），只作输入内容，不是目标音色。说明见 `testdata/generated/SPEECH_SOURCE.txt`。

命令（项目根目录，10:06:25–10:06:48 Asia/Shanghai）：

```bash
cd /workspace/rvc-realtime-voice
.venv-rvc/bin/python -m rvc_realtime run --mode convert \
  --input testdata/generated/speech_jfk.wav \
  --output testdata/recorded/shadowww-t02_convert.wav \
  --model assets/weights/shadowww-t02.pth \
  --index assets/indices/added_shadowww-t02.index \
  --index-rate 0.5 \
  --f0-method rmvpe \
  --f0-up-key 0 \
  --buffer-ms 1000 \
  --sample-rate 48000 \
  --channels 1
```

结果：**成功**（退出码 0）。输出路径：`testdata/recorded/shadowww-t02_convert.wav`。

| 项目 | 输入 | 输出 |
| --- | --- | --- |
| 时长 | 11.000 s | 11.000 s |
| 大小 | 1,056,078 bytes | 1,056,044 bytes |
| 采样 | 48 kHz / 单声道 | 48 kHz / 单声道 |
| 有声 | 是 | 是 |
| RMS | 0.1421 | 0.0411 |
| 峰值 | 0.7836 | 0.2530 |
| 主频（粗测） | 521.1 Hz | 247.8 Hz |
| 谱质心（粗测） | 1369 Hz | 1697 Hz |
| SHA-256 | 559d8572…18c2e56c | d58319ca…93a8865（不同） |

对齐后互相关系数约 0.12（3 ms 位移），明显不是原样拷贝或直通。处理耗时 16.575 s，含首次加载 RMVPE 的约 11.6 s；整体 RTF=1.507。`--buffer-ms 1000` 只用于文件转换减推理次数，不代表实时延迟已达标（程序据此估算的 2000 ms 缓冲延迟不应解读为通话延迟）。

日志：`Index search enabled`，并成功打印 `Loading RMVPE model`。首块之后的若干块出现 `Invalid index: use added_xxxx.index, not trained_xxxx.index`。实际使用的就是 `added_shadowww-t02.index`；官方代码在 faiss 检索出现 `-1` 时会打印这句，这些帧会跳过索引融合，但推理仍继续并写出完整 11 s 音频。这是质量提示，不是失败。

未使用目录中可能存在的其他 `.pth`。未做 DirectML / RX 7800 XT。未对接真实麦克风或聊天软件。

## 2026-11-24 10:21–10:31（Asia/Shanghai，UTC+8）：音量/索引优化、模拟实时、界面与冒烟训练

### 文件转换对比（同一输入与同一授权测试模型）

输入仍是公有领域 `testdata/generated/speech_jfk.wav`（11.000 s，48 kHz 单声道，RMS 0.1421，峰值 0.7836），模型仍是 `assets/weights/shadowww-t02.pth` + `assets/indices/added_shadowww-t02.index`。未覆盖旧输出。

| 项目 | 源音频 | 优化前 `shadowww-t02_convert.wav`（10:06） | 优化后 `shadowww-t02_convert_v2.wav`（10:21） |
| --- | --- | --- | --- |
| RMS | 0.1421 | 0.0411 | 0.1421 |
| 峰值 | 0.7836 | 0.2530 | 0.8853 |
| 索引融合 | — | 部分帧 faiss 返回 `-1` 时整块跳过（日志出现 Invalid index） | 按帧融合有效邻居，本段 550/550（100%） |
| 相对源的相关 | — | 约 0.12 | 约 0.005（音色已变，不是响度回归成原声拷贝） |
| 相对 v1 的相关 | — | — | 0.964（内容与 v1 接近，主要是响度和索引稳定） |

v2 使用默认 `--loudness-mode match-input`（文件级对齐后再限制峰值 0.95）以及 `rvc_infer_safe` 的按帧索引融合。当时记录：文件级增益约 0.93×，对齐前块 RMS 约 0.1520，对齐后 0.1421；`--buffer-ms 1000` 下整体 RTF 约 0.571。不要把该缓冲值当成实时通话延迟。

### 模拟实时（不是真实麦克风验收）

命令：`.venv-rvc` 下对同一 JFK + Shadowww T02 运行 `--buffer-ms 200 --simulate-realtime`，输出 `testdata/recorded/simulate_realtime.wav`（约 10:25:49–10:26:06）。

- 写出 11.000 s，RMS 0.1383，峰值 0.9500（触达限制器上限附近）
- 索引同样 550/550
- 平均块处理约 163 ms，处理 RTF 约 0.816（首块因加载 RMVPE 明显更慢，约 1.4–1.6 s）
- 程序估计的输入+输出缓冲约 400 ms（200+200），尚未达到 100–200 ms 单向端到端目标
- 没有真实麦克风，也没有接入聊天软件，不能当成通话验收

### 本地界面

```bash
.venv-rvc/bin/python -m rvc_realtime serve --host 127.0.0.1 --port 7870
# 或 ./scripts/start_ui.sh
```

进程仍在 `127.0.0.1:7870` 监听（PID 310080），HTTP 200。页面含「麦克风录音」「训练模型」「实时变声」「说明」，以及授权勾选、导入授权录音、直通和模拟实时。`share=False`，未做公网分享。盒子无麦时录音应明确报错，并改用导入 WAV。

### 冒烟训练（公有领域，仅验证链路）

素材：`testdata/train_smoke_jfk/speech_jfk.wav`（公有领域 JFK，说明见 `testdata/TRAIN_SMOKE_JFK.md`）。**没有**用 Shadowww 的声音再训新音色。

未加 `--i-have-authorization` 时于 10:28 被拒绝，退出码 2。

成功命令（10:30:59–10:31:42 Asia/Shanghai，总用时约 0.7 分钟；其中 1 轮训练约 22.7 秒）：

```bash
.venv-rvc/bin/python -m rvc_realtime train   --audio-dir testdata/train_smoke_jfk   --name smoke_jfk_pd   --i-have-authorization   --version v2 --sr 40k   --epochs 1 --batch-size 1 --f0-method rmvpe
```

产出：

- `assets/weights/smoke_jfk_pd.pth`（约 53 MiB，10:31）
- `assets/indices/added_smoke_jfk_pd.index`（指向官方 `added_IVF30_Flat_nprobe_1_smoke_jfk_pd_v2.index`）

这只证明切分 → 音高/特征提取 → 训练 → 导出权重和索引能跑通。1 轮、约 11 秒语料不能当成品音色使用。官方脚本通过 `python -m rvc_realtime.run_official` 以模块方式启动，避免 `python train/preprocess.py` 的循环导入，以及 `exec` 导致的 `multiprocessing` 无法 pickle。

本次重跑时实验目录里还留有上一轮失败尝试的 `1_*` 切片，filelist 实际喂了 8 段（外加 mute）。当前训练代码会在开始前清掉旧的特征/切片目录，并只把音频暂存进 `_authorized_input`，避免 README 等非音频文件进入预处理。未再为这一点重跑冒烟。

完整质量训练（10–30 分钟授权录音、更多轮次）未在本盒子上执行；CPU 上预计数小时量级。


## 2026-11-24 11:03–11:04（Asia/Shanghai，UTC+8）：游戏短音频变身界面与扬声器录音主路径

按用户纠正，录音主路径改为系统扬声器 / 环回，不再把麦克风当成默认。主界面改为「游戏短音频变身」一条龙，并做了深色自定义样式改版。

### 界面

先停旧进程 PID 310080，再于 11:03:45 之后重新启动：

```bash
.venv-rvc/bin/python -m rvc_realtime serve --host 127.0.0.1 --port 7870
```

新进程 PID 331660，HTTP 200。页面标题与文案含「游戏短音频变身」「开始录制扬声器」「停止并保存」「一键训练」「一键开始实时变声」，默认录音设备为「扬声器环回 · speakers.monitor」，模型名默认 `game_voice`，有 1 分钟 / 2 分钟目标时长。页面中已不再出现「麦克风录音 / 开始麦克风录音」。`share=False`。自定义 CSS（深色背景、主按钮、卡片步骤）已注入页面。

### 预设训练入口冒烟

未使用 Shadowww 再训音色，也未覆盖 `testdata/recorded/shadowww-t02_convert*.wav` 或 `assets/weights/smoke_jfk_pd.pth`。

用公有领域 JFK 走「导入 + 游戏预设训练（缩短为 1 轮）」：11:04:08–11:04:50，总用时 41.5 秒，其中 1 轮训练约 17 秒。切分 3.0 秒，得到 4 个片段。

- 导入：`testdata/train_game_smoke/imported_speech_jfk_20261124-110408.wav`
- 模型：`assets/weights/game_voice_smoke.pth`（约 53 MiB）
- 索引：`assets/indices/added_game_voice_smoke.index`

这只验证预设训练入口可用。产品默认仍是 60 轮，不是这 1 轮冒烟权重。

## 仍未在本盒子上完成

1. 没有真实麦克风，因此没有对真人说话做实时变声或直通验收，也没有对接聊天软件。也没有用真实游戏语音做扬声器环回录音验收（盒子上仅确认默认设备为 speakers.monitor，并用导入 WAV 验证训练入口）。
2. 未能创建命名虚拟线缆 `RVC_Cable_In` / `RVC_Cable_Out`（本盒子 Pulse 禁止 `load-module`）。文件转换和模拟实时已作为本次验收。
3. 未在 Windows 上安装或测试 VB-CABLE，未对接聊天软件。
4. 未尝试 AMD RX 7800 XT / DirectML / ROCm。
5. 未使用用户本人的授权参考录音做完整质量训练（仅用公有领域 JFK 做了 1 轮链路冒烟）。
6. `rmvpe.onnx` 未下载（Linux CPU 路径不需要；Windows DirectML 的 RMVPE 必需，由 Windows 下载脚本默认获取）。
7. 模拟实时在 200 ms 缓冲下，估计输入+输出缓冲约 400 ms，尚未达到 100–200 ms 单向端到端目标。

## 在目标 Windows 机器上建议复测

- 安装 Python 3.12 与 VB-CABLE，运行 `.\scripts\install_windows.ps1`。
- 用真实麦克风做直通：程序输出 `CABLE Input`，聊天软件选 `CABLE Output`。
- 变声：`.\scripts\install_windows.ps1 -WithRvcCpu`，激活 `.venv-rvc`，运行 `.\scripts\start_ui.ps1`。先直通验证 CABLE 路由，再加载已授权 `.pth` 变声；或命令行使用 `--mode convert`。实时缓冲从 100–200 ms 调起，使用耳机，可随时把聊天软件麦克风切回真实麦克风。
- 训练：在界面勾选授权后录音或导入 WAV，再训练导出 `.pth`。不要上传录音或音色，不要使用未授权音色。
