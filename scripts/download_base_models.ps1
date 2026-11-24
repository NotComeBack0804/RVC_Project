$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = (Get-Command python -ErrorAction Stop).Source
}
& $Py (Join-Path $Root "scripts\download_base_models.py") @args
