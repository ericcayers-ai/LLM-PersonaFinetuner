$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python 3.10+ required." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = Join-Path $Root ".venv\Scripts\python.exe"
$pip = Join-Path $Root ".venv\Scripts\pip.exe"
& $py -m pip install --upgrade pip --quiet
Write-Host "Installing PyTorch (CUDA 12.4)..."
& $pip install torch --index-url https://download.pytorch.org/whl/cu124
Write-Host "Installing requirements..."
& $pip install -r requirements.txt
Write-Host "Setup complete. Run run.bat to start."
