$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv-rvc\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "请先运行 .\scripts\install_windows.ps1 -WithRvcCpu" }
$port = 7870
if ($env:PORT) { $port = [int]$env:PORT }
& $Py -m rvc_realtime serve --host 127.0.0.1 --port $port
