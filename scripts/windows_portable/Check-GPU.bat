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
if not exist "%ROOT%runtime\python.exe" (
    echo 未找到 runtime\python.exe。请确认已完整解压。
    pause
    exit /b 1
)
"%ROOT%runtime\python.exe" -m rvc_realtime.windows_bootstrap
echo.
"%ROOT%runtime\python.exe" -m rvc_realtime status
echo.
echo 若上面不是 DirectML 和 privateuseone:0，请查看「回退原因」。
echo 常见原因：AMD 驱动过旧（需要 DirectX 12）、未安装 VC++ 运行库，或强制设置了 RVC_COMPUTE_DEVICE=cpu。
pause
