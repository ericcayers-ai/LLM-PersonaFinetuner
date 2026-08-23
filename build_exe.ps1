$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

python -m pip install --quiet pyinstaller

pyinstaller --noconfirm --onefile --console `
  --name PersonaFinetuner `
  --distpath . `
  --workpath build `
  --specpath build `
  launcher.py

Write-Host "Built PersonaFinetuner.exe in $Root"
Write-Host "Run setup.bat once (to create .venv with all dependencies), then double-click PersonaFinetuner.exe."
