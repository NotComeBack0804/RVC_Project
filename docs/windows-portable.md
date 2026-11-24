# Windows 开箱即用便携包（AMD DirectML）

给 Windows + AMD 显卡（例如 RX 7800 XT）准备的便携目录：已内置 Python 3.12 x64、官方 CPU/DirectML 依赖、ffmpeg，以及公开底模。解压后双击即可启动本地界面，不必再运行 `install_windows.ps1`，也不必自行安装 Python。

打包脚本：`scripts/build_windows_portable.py`。成品不要提交进 git，默认输出在仓库外的 `/workspace/rvc-windows-directml.zip`（若体积过大则分成 runtime / models 两个 zip）。GitHub Actions 工作流：`.github/workflows/build-windows-portable.yml`。

## 解压与启动

1. 安装 [VB-CABLE](https://vb-audio.com/Cable/)，必要时重启。
2. 确认已安装较新的 AMD 显卡驱动（需要 DirectX 12）。
3. 若尚未安装 64 位 Microsoft Visual C++ 2015–2022 运行库，先运行包内 `runtime\vc_redist.x64.exe`。
4. 将压缩包完整解压到较短路径（例如 `D:\RVC-Realtime`）。若分成两个 zip，先解压 runtime，再把 models 解压到**同一个父目录**，使 `assets\` 合并进同一文件夹。
5. 双击 `Start-UI.bat`（或 `启动界面.bat` / `Start-UI.exe`）。
6. 等待数秒，浏览器会打开 http://127.0.0.1:7870 。不要关闭黑色命令行窗口。

界面只监听本机，不会做 Gradio 公网分享。

## 界面（v4）

左侧步骤栏可以点击，分别进入「录制扬声器」「一键训练」「实时变声」，不再只是摆设。

扬声器和输出/虚拟麦克风放在 **「系统设置」** 弹窗里选择（点左侧「系统设置」，或页面上的「在系统设置中更改」）。音高、缓冲、训练轮数等在同一弹窗的「高级设置」页。

训练页可以更换训练底模。默认仍是 RVC v2 40k（`f0G40k.pth` / `f0D40k.pth`）。把更好的成对底模放到 `assets\pretrained_custom\`（可建子目录），或在训练页直接导入，再点「刷新底模」即可选用，不必改代码。约定见 [pretrained-bases.md](pretrained-bases.md)。


## 如何确认已经走到显卡

- 启动时命令行会打印计算设备，应类似：`计算设备：DirectML（AMD Radeon RX 7800 XT，privateuseone:0）`。
- 也可双击 `Check-GPU.bat`（或 `查看计算设备.bat`），等价于 `runtime\python.exe -m rvc_realtime status`。
- 界面顶部「当前计算设备」应显示 DirectML 和适配器名，而不是 CPU。

若显示 CPU，以打印出的「回退原因」为准。常见原因：驱动过旧、未装 VC++ 运行库，或设置了 `RVC_COMPUTE_DEVICE=cpu`。

## VB-CABLE 路由

请戴耳机。实时变声时：

- 本程序输入：你说话用的真实麦克风
- 本程序输出：`CABLE Input`
- 游戏或聊天软件麦克风：`CABLE Output`

录音主路径默认录系统扬声器（WASAPI 环回），不要从 CABLE 再录一遍，以免把变声结果录进去。可随时把游戏麦克风切回真实麦克风。

## 强制 CPU（排障）

双击 `Start-UI-CPU.bat`（或 `使用CPU启动.bat`），或在启动前设置：

```bat
set RVC_COMPUTE_DEVICE=cpu
Start-UI.bat
```

## 包内有什么、没有什么

包含：便携 Python 3.12、PyTorch 2.4.1 CPU、`torch-directml`、`onnxruntime-directml`、ffmpeg、HuBERT、`rmvpe.pt`、`rmvpe.onnx`，以及默认训练预设所需的 `assets/pretrained_v2/f0G40k.pth` 与 `f0D40k.pth`。不含 CUDA。

不含：`.venv`、用户录音、他人或自训音色（如 Shadowww）、冒烟测试 `.pth`、`.git`。把你自己训练的模型放到 `assets\weights\`，索引放到 `assets\indices\`。

若要训练 32k / 48k 或 v1，可在本目录运行：

```bat
runtime\python.exe scripts\download_base_models.py --with-pretrained
```

## 重新打包

在 Linux 或 Windows 上，于仓库根目录：

```bash
python scripts/build_windows_portable.py --cache /workspace/rvc-windows-cache --output /workspace/rvc-windows-directml.zip
```

脚本会交叉下载 Windows wheels、组装可携目录并做结构检查。Linux 上无法真正运行 `python.exe` 或加载 DirectML 适配器。
