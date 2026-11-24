RVC 实时变声 — Windows 开箱即用包（AMD DirectML）

这是一份便携包，已内置 Python 3.12、官方 CPU/DirectML 依赖、ffmpeg，
以及 HuBERT、RMVPE（含 rmvpe.onnx）和默认训练所需的 v2 40k 底模。
不需要再运行 install_windows.ps1，也不需要自行安装 Python。

========================================
使用步骤
========================================

1. 安装 VB-CABLE（https://vb-audio.com/Cable/），必要时重启电脑。
2. 确认已安装较新的 AMD 显卡驱动（需要支持 DirectX 12）。
3. 若尚未安装 64 位 Microsoft Visual C++ 2015-2022 运行库，
   请先双击运行 runtime\vc_redist.x64.exe，装完后再继续。
4. 把本文件夹完整解压到较短路径，例如 D:\RVC-Realtime。
   如果下载的是两个压缩包：先解压 *runtime*.zip，再把 *models*.zip
   解压到同一个父目录，让 assets 合并进来。
5. 双击 Start-UI.bat（或「启动界面.bat」/ Start-UI.exe）。
6. 等待数秒，浏览器会打开 http://127.0.0.1:7870 。
   请不要关闭弹出的黑色命令行窗口，关闭后界面会停止。

界面只在本机监听，不会做公网分享，也不会上传录音或音色。

========================================
界面（v4）
========================================

左侧步骤栏可以点击，分别进入「录制扬声器」「一键训练」「实时变声」。

扬声器和输出/虚拟麦克风在「系统设置」弹窗里选择（点左侧「系统设置」，
或页面上的「在系统设置中更改」）。音高、缓冲、训练轮数等在同一弹窗的
「高级设置」页。

训练页可以更换训练底模。默认是 RVC v2 40k（f0G40k / f0D40k）。
把更好的成对底模放到 assets\pretrained_custom\（可建子目录），
或在训练页直接导入，再点「刷新底模」即可选用，不必改代码。
详见 docs\pretrained-bases.md。


========================================
如何确认已经走到 AMD 显卡
========================================

启动时命令行会打印计算设备，正常时应类似：

  计算设备：DirectML（AMD Radeon RX 7800 XT，privateuseone:0）。
    类型：DirectML
    设备名：privateuseone:0
    适配器：AMD Radeon RX 7800 XT
    RMVPE 需要 rmvpe.onnx：是

也可以双击 Check-GPU.bat（或「查看计算设备.bat」）。
打开界面后，首页顶部「当前计算设备」应显示 DirectML 和适配器名，而不是 CPU。

如果显示的是 CPU，请阅读「回退原因」。常见情况：
- AMD 驱动过旧或未正确安装（DirectML 需要 DirectX 12）
- 尚未安装 VC++ 运行库
- 手动设置了 RVC_COMPUTE_DEVICE=cpu

========================================
VB-CABLE 与游戏短音频变身
========================================

请戴上耳机。界面默认录制系统扬声器（环回），不是麦克风。

勾选「这是已授权的声音」之后：
  录 1 或 2 分钟干净发言 → 一键训练 → 一键开始实时变声。

实时路由：
  - 本程序输入：你说话用的真实麦克风（在「系统设置」里）
  - 本程序输出：CABLE Input
  - 游戏或聊天软件的麦克风：CABLE Output

录音时先不要开实时变声，也不要从 CABLE 再录一遍。
需要立刻恢复时，把游戏麦克风改回真实麦克风，并停止本程序的实时输出。

========================================
排障：强制使用 CPU
========================================

双击 Start-UI-CPU.bat（或「使用CPU启动.bat」），或在命令行：

  set RVC_COMPUTE_DEVICE=cpu
  Start-UI.bat

========================================
本包包含什么
========================================

- runtime\  便携 Python 3.12 x64，以及已安装的 Windows 依赖
  （PyTorch 2.4.1 CPU、torch-directml、onnxruntime-directml 等，不含 CUDA）
- runtime\ffmpeg\  ffmpeg / ffprobe
- runtime\vc_redist.x64.exe  Visual C++ 运行库安装程序
- assets\hubert_base\、assets\rmvpe\rmvpe.pt、assets\rmvpe\rmvpe.onnx
- 训练底模 assets\pretrained_v2\f0G40k.pth 与 f0D40k.pth（默认 v2 / 40k 预设）
- 本项目界面与官方 RVC 源码（third_party\...）

不含：他人或自训音色、测试用冒烟模型、录音、.git、Python 虚拟环境。
把你自己训练得到的 .pth 放到 assets\weights\，索引放到 assets\indices\。

若要训练 32k / 48k 或 v1 模型，需要额外底模。在本目录打开命令行运行：

  runtime\python.exe scripts\download_base_models.py --with-pretrained

========================================
授权与第三方许可
========================================

请只使用你本人或已获明确授权的声音。参考音频和音色模型请留在本机。
使用变声通话时，建议明确告知对方声音经过了转换。

第三方许可见 docs\、NOTICE.txt，以及 runtime\ffmpeg 目录中的许可证文件。
