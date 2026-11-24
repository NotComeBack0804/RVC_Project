# 在 Windows 上创建 Python 3.12 虚拟环境并安装依赖。默认只装轻量直通环境。
# 完整环境（-WithRvcCpu / -WithRvc）按官方 requirments_cpu_py312.txt 安装：
# Windows 会安装 torch-directml 与 onnxruntime-directml，用于 RX 7800 XT 等 AMD 显卡。
# 不要在该环境中混入 CUDA 安装命令。
[CmdletBinding()]
param(
    [switch]$WithRvcCpu,
    [switch]$WithRvc,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$WithFull = $WithRvcCpu -or $WithRvc

function Find-Python312 {
    $candidates = @()
    if ($env:PYTHON_BIN) { $candidates += $env:PYTHON_BIN }
    try {
        $py = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $py) { $candidates += $py.Trim() }
    } catch {}
    foreach ($name in @("python3.12", "python")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $candidates += $cmd.Source }
    }
    foreach ($exe in $candidates) {
        if (-not (Test-Path $exe)) { continue }
        $ver = & $exe -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($ver -eq "3.12") { return $exe }
        if ([version]$ver -ge [version]"3.10") {
            Write-Warning "未找到 Python 3.12，将使用 $exe ($ver)。官方 RVC 依赖面向 3.12。"
            return $exe
        }
    }
    throw "未找到 Python 3.12 x64。请先从 python.org 安装，并勾选 Add python.exe to PATH。"
}

function Ensure-RvcRepo {
    $dest = Join-Path $Root "third_party\Retrieval-based-Voice-Conversion-WebUI"
    $req = Join-Path $dest "requirments_cpu_py312.txt"
    if (Test-Path $req) { return }
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "third_party") | Out-Null
    Write-Host "正在浅克隆官方 RVC 仓库到 third_party\ ..."
    git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git $dest
}

$Python = Find-Python312
Write-Host "使用解释器: $Python"
& $Python -V

if (-not (Test-Path (Join-Path $Root ".venv\Scripts\python.exe"))) {
    & $Python -m venv (Join-Path $Root ".venv")
}
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip setuptools wheel
& $VenvPy -m pip install -r (Join-Path $Root "requirements-lite.txt")
Write-Host "轻量环境已就绪: $Root\.venv"
Write-Host "激活:  .\.venv\Scripts\Activate.ps1"

Ensure-RvcRepo

if ($WithFull) {
    $Req = Join-Path $Root "third_party\Retrieval-based-Voice-Conversion-WebUI\requirments_cpu_py312.txt"
    if (-not (Test-Path $Req)) {
        throw "未找到官方依赖文件 requirments_cpu_py312.txt（文件名以仓库当前版本为准）。"
    }
    Write-Host "正在创建独立的 RVC 环境 .venv-rvc（官方 CPU/DirectML 依赖，不安装 CUDA）..."
    if (-not (Test-Path (Join-Path $Root ".venv-rvc\Scripts\python.exe"))) {
        & $Python -m venv (Join-Path $Root ".venv-rvc")
    }
    $RvcPy = Join-Path $Root ".venv-rvc\Scripts\python.exe"
    & $RvcPy -m pip install --upgrade pip setuptools wheel
    & $RvcPy -m pip install -r $Req
    Write-Host "正在探测计算设备（Windows 上应优先出现 DirectML / privateuseone:0）..."
    & $RvcPy -c @"
import sys
sys.path.insert(0, r'$Root')
from rvc_realtime.compute import format_compute_status, probe_directml, probe_onnxruntime_directml, select_compute
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available(), '(本项目不使用 CUDA)')
ok, device, name, err = probe_directml()
print('torch-directml:', 'OK', device, name if ok else 'FAIL', err)
ort_ok, ort_err = probe_onnxruntime_directml()
print('onnxruntime-directml:', 'OK' if ort_ok else 'FAIL', ort_err)
print(format_compute_status())
info = select_compute()
if not info.is_directml:
    print('警告: 当前没有走到 DirectML，训练和实时变声将回退 CPU。请检查 AMD 驱动与官方依赖是否安装完整。')
    sys.exit(0)
"@
    if (-not $SkipModels) {
        Write-Host "正在下载公开基础模型（含 rmvpe.onnx，DirectML 音高提取必需）..."
        & $RvcPy (Join-Path $Root "scripts\download_base_models.py")
        Write-Host "若要训练自己的授权音色，还需要底模：.\scripts\download_base_models.ps1 --with-pretrained"
    }
    Write-Host "RVC 环境已就绪: $Root\.venv-rvc"
} else {
    Write-Host "未安装完整 RVC 推理/训练依赖。要在 RX 7800 XT 上用 DirectML 变声或训练，请运行:"
    Write-Host "  .\scripts\install_windows.ps1 -WithRvcCpu"
}

Write-Host ""
Write-Host "Windows 音频路由还需要安装 VB-CABLE: https://vb-audio.com/Cable/"
Write-Host "安装后: 程序输出选 CABLE Input，聊天软件麦克风选 CABLE Output。"
Write-Host "接下来建议:"
if ($WithFull) {
    Write-Host "  .\.venv-rvc\Scripts\python.exe -m rvc_realtime status"
    Write-Host "  .\.venv-rvc\Scripts\python.exe -m rvc_realtime list-devices"
    Write-Host "  .\scripts\start_ui.ps1"
} else {
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  python -m rvc_realtime list-devices"
    Write-Host "  python -m rvc_realtime self-test --skip-pulse"
    Write-Host "  .\scripts\download_base_models.ps1"
}
