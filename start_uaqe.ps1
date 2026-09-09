<#
.SYNOPSIS
    Universal AI Quantization Engine (UAQE) PowerShell Launcher.
.DESCRIPTION
    Resolves repository root dynamically, verifies prerequisites, recreates runtime
    output directories, and launches the FastAPI backend and Vite frontend.
#>

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "UAQE — UNIVERSAL AI QUANTIZATION ENGINE LAUNCHER (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Resolve repository root dynamically
$RepoRoot = $PSScriptRoot
Set-Location -Path $RepoRoot

# 2. Detect Python interpreter: .venv -> venv -> PATH python
$PythonBin = $null
if (Test-Path "$RepoRoot\.venv\Scripts\python.exe") {
    $PythonBin = "$RepoRoot\.venv\Scripts\python.exe"
    Write-Host "[INFO] Detected virtual environment: .venv" -ForegroundColor Green
} elseif (Test-Path "$RepoRoot\venv\Scripts\python.exe") {
    $PythonBin = "$RepoRoot\venv\Scripts\python.exe"
    Write-Host "[INFO] Detected virtual environment: venv" -ForegroundColor Green
} else {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython) {
        $PythonBin = "python"
        Write-Host "[INFO] Using system Python from PATH." -ForegroundColor Green
    }
}

# 3. Verify Python availability
if (-not $PythonBin) {
    Write-Host "`n[ERROR] Python interpreter not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.11+ or configure a virtual environment (.venv).`n" -ForegroundColor Red
    exit 1
}

# 4. Verify essential backend dependencies (FastAPI, Uvicorn)
Write-Host "[1/4] Checking Python backend dependencies..." -ForegroundColor Gray
$depCheck = & $PythonBin -c "import fastapi, uvicorn" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Required backend dependencies (fastapi, uvicorn) are missing!" -ForegroundColor Red
    Write-Host "Please install dependencies with: pip install -r requirements.txt`n" -ForegroundColor Red
    exit 1
}

# 5. Verify Node.js / npm availability
Write-Host "[2/4] Checking Node.js and npm..." -ForegroundColor Gray
$npmCheck = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCheck) {
    Write-Host "`n[ERROR] 'npm' was not found in PATH!" -ForegroundColor Red
    Write-Host "Please install Node.js 18+ (https://nodejs.org/) to run the web frontend.`n" -ForegroundColor Red
    exit 1
}

# 6. Verify frontend dependencies installed
if (-not (Test-Path "$RepoRoot\frontend\node_modules")) {
    Write-Host "[INFO] Frontend node_modules not found. Running npm install..." -ForegroundColor Yellow
    Push-Location "$RepoRoot\frontend"
    npm install
    Pop-Location
    if (-not (Test-Path "$RepoRoot\frontend\node_modules")) {
        Write-Host "`n[ERROR] Failed to install frontend dependencies.`n" -ForegroundColor Red
        exit 1
    }
}

# 7. Recreate required runtime directories automatically
Write-Host "[3/4] Ensuring runtime directories exist..." -ForegroundColor Gray
$runtimeDirs = @(
    "$RepoRoot\output\jobs",
    "$RepoRoot\output\uploads\models",
    "$RepoRoot\output\uploads\datasets",
    "$RepoRoot\scratch"
)
foreach ($dir in $runtimeDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# 8. Start backend on port 8000
Write-Host "[4/4] Starting UAQE Backend and Frontend..." -ForegroundColor Gray
Write-Host "  -> Starting Python Backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$backendCmd = "cd /d `"$RepoRoot`" && `"$PythonBin`" -c `"import sys; sys.path.insert(0, 'src'); import uvicorn; uvicorn.run('uaqe.server:app', host='127.0.0.1', port=8000, log_level='info')`""
Start-Process cmd.exe -ArgumentList "/k `"$backendCmd`""

Start-Sleep -Seconds 2

# 9. Start frontend on port 3000
Write-Host "  -> Starting Vite Frontend on http://localhost:3000 ..." -ForegroundColor Cyan
$frontendCmd = "cd /d `"$RepoRoot\frontend`" && npm run dev"
Start-Process cmd.exe -ArgumentList "/k `"$frontendCmd`""

Start-Sleep -Seconds 3

# Open browser
Write-Host "  -> Opening default web browser..." -ForegroundColor Cyan
Start-Process "http://localhost:3000"

# 10-12. Status messages, URLs, shutdown guidance
Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "UAQE ENGINE IS RUNNING!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  - Frontend Dashboard : http://localhost:3000" -ForegroundColor White
Write-Host "  - Backend REST API   : http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - Interactive Docs   : http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "SHUTDOWN GUIDANCE:" -ForegroundColor Yellow
Write-Host "To stop UAQE: Close the separate 'UAQE Backend' and 'UAQE Frontend'" -ForegroundColor Yellow
Write-Host "console windows, or press Ctrl+C inside those windows." -ForegroundColor Yellow
Write-Host "======================================================================`n" -ForegroundColor Green
