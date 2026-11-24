$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv-rvc\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Py)) {
    $Py = (Get-Command python -ErrorAction Stop).Source
}
# Windows 上脚本会默认下载 rmvpe.onnx（DirectML 的 RMVPE 必需）。
& $Py (Join-Path $Root "scripts\download_base_models.py") @args
