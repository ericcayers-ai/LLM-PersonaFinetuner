$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$VenvUvicorn = Join-Path $Root ".venv\Scripts\uvicorn.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not found. Run setup.bat or setup.ps1 first." -ForegroundColor Yellow
    exit 1
}
$Port = if ($env:PF_PORT) { $env:PF_PORT } else { "8000" }
$Url = "http://localhost:$Port"
Write-Host "PersonaFinetuner starting at $Url (Ctrl+C to stop)"
Start-Process $Url | Out-Null
& $VenvUvicorn app.main:app --host 127.0.0.1 --port $Port --reload
