$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = "1"

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
  & (Join-Path $PSScriptRoot "setup_venv.bat")
  if ($LASTEXITCODE -ne 0) {
    throw "虚拟环境创建失败"
  }
}

& $pythonPath -X utf8 .\generate_mdm.py ".\test\no title.mp3" `
  --checkpoint ".\checkpoints\MuseChart_v1.0.pt" `
  --profile balanced `
  --difficulty 2 `
  --play-level 7 `
  --scene scene_01 `
  --speed 2
