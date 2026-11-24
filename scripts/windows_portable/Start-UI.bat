@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PATH=%ROOT%runtime;%ROOT%runtime\Scripts;%ROOT%runtime\ffmpeg;%PATH%"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not defined RVC_COMPUTE_DEVICE set "RVC_COMPUTE_DEVICE=auto"
if not defined PORT set "PORT=7870"

if not exist "%ROOT%runtime\python.exe" (
    echo 未找到 runtime\python.exe。请确认已把压缩包完整解压到同一个文件夹。
    pause
    exit /b 1
)

"%ROOT%runtime\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo 便携 Python 无法启动，通常是缺少 64 位 Microsoft Visual C++ 运行库。
    echo 请先安装本目录中的 runtime\vc_redist.x64.exe，必要时重启后再双击本脚本。
    if exist "%ROOT%runtime\vc_redist.x64.exe" start "" "%ROOT%runtime\vc_redist.x64.exe"
    pause
    exit /b 1
)

echo 正在准备运行环境（资产目录、底模检查）...
"%ROOT%runtime\python.exe" -m rvc_realtime.windows_bootstrap
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo 正在检查计算设备（Windows + AMD 显卡应显示 DirectML / privateuseone:0）...
"%ROOT%runtime\python.exe" -m rvc_realtime status
echo.
echo 启动本地界面：http://127.0.0.1:%PORT%
echo 请不要关闭本窗口；关闭后界面会停止。稍后会自动打开浏览器。
echo.
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:%PORT%/"
"%ROOT%runtime\python.exe" -m rvc_realtime serve --host 127.0.0.1 --port %PORT%
echo.
echo 界面已退出。
pause
