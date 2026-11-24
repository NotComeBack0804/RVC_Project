@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "RVC_COMPUTE_DEVICE=cpu"
echo 已强制使用 CPU（排障模式）。
call "%~dp0Start-UI.bat"
