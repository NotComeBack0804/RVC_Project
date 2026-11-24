$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv-rvc\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "请先运行 .\scripts\install_windows.ps1 -WithRvcCpu" }
Write-Host "正在检查计算设备（Windows 上应优先为 DirectML）..."
& $Py -c "from rvc_realtime.compute import format_compute_status; print(format_compute_status())"
if ($LASTEXITCODE -ne 0) { throw "计算设备检查失败。" }
$port = 7870
if ($env:PORT) { $port = [int]$env:PORT }
Write-Host "启动本地界面：http://127.0.0.1:$port"
& $Py -m rvc_realtime serve --host 127.0.0.1 --port $port
