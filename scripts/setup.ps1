# One-time setup: creates .venv with Python 3.11, installs dependencies, checks the machine.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [--pull]
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv with Python 3.11..."
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 not found. Install it: winget install Python.Python.3.11" }
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed; see the error above." }

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

.\.venv\Scripts\python.exe scripts\check_setup.py @args
exit $LASTEXITCODE
