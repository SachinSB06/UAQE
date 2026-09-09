@echo off
setlocal enabledelayedexpansion
title UAQE — Universal AI Quantization Engine

echo ======================================================================
echo UAQE — UNIVERSAL AI QUANTIZATION ENGINE LAUNCHER
echo ======================================================================

:: 1. Resolve repository root dynamically from script location
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

cd /d "%REPO_ROOT%"

:: 2. Detect Python interpreter: .venv -> venv -> system python
set "PYTHON_BIN="
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%REPO_ROOT%\.venv\Scripts\python.exe"
    echo [INFO] Detected virtual environment: .venv
) else if exist "%REPO_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_BIN=%REPO_ROOT%\venv\Scripts\python.exe"
    echo [INFO] Detected virtual environment: venv
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_BIN=python"
        echo [INFO] Using system Python from PATH.
    )
)

:: 3. Verify Python availability
if "!PYTHON_BIN!"=="" (
    echo.
    echo [ERROR] Python interpreter not found!
    echo Please install Python 3.11+ (https://www.python.org/) or create a virtual environment (.venv).
    echo.
    pause
    exit /b 1
)

:: 4. Verify essential backend dependencies (FastAPI, Uvicorn)
echo [1/4] Checking Python backend dependencies...
"!PYTHON_BIN!" -c "import fastapi, uvicorn" >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Essential backend dependencies (fastapi, uvicorn) are missing.
    echo Please install dependencies with:
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 5. Verify Node.js / npm availability
echo [2/4] Checking Node.js and npm...
where npm >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] 'npm' was not found in PATH!
    echo Please install Node.js 18+ (https://nodejs.org/) to run the web frontend.
    echo.
    pause
    exit /b 1
)

:: 6. Verify frontend dependencies installed
if not exist "%REPO_ROOT%\frontend\node_modules" (
    echo [INFO] Frontend node_modules not found. Running npm install...
    pushd "%REPO_ROOT%\frontend"
    call npm install
    popd
    if not exist "%REPO_ROOT%\frontend\node_modules" (
        echo.
        echo [ERROR] Failed to install frontend dependencies. Please run 'npm install' in frontend directory.
        echo.
        pause
        exit /b 1
    )
)

:: 7. Recreate required runtime directories automatically
echo [3/4] Ensuring runtime directories exist...
if not exist "%REPO_ROOT%\output\jobs" mkdir "%REPO_ROOT%\output\jobs"
if not exist "%REPO_ROOT%\output\uploads\models" mkdir "%REPO_ROOT%\output\uploads\models"
if not exist "%REPO_ROOT%\output\uploads\datasets" mkdir "%REPO_ROOT%\output\uploads\datasets"
if not exist "%REPO_ROOT%\scratch" mkdir "%REPO_ROOT%\scratch"

:: 8. Start backend using repository-relative paths
echo [4/4] Starting UAQE Backend and Frontend...
echo   -> Starting Python Backend on http://127.0.0.1:8000 ...
start "UAQE Backend (Port 8000)" cmd /k "cd /d ""%REPO_ROOT%"" && ""!PYTHON_BIN!"" -c ""import sys; sys.path.insert(0, 'src'); import uvicorn; uvicorn.run('uaqe.server:app', host='127.0.0.1', port=8000, log_level='info')"""

timeout /t 2 /nobreak >nul

:: 9. Start frontend using repository-relative paths
echo   -> Starting Vite Frontend on http://localhost:3000 ...
start "UAQE Frontend (Port 3000)" cmd /k "cd /d ""%REPO_ROOT%\frontend"" && npm run dev"

timeout /t 3 /nobreak >nul

:: Open browser
echo   -> Opening default web browser...
start http://localhost:3000

:: 10-12. Useful status messages, URLs, shutdown guidance
echo.
echo ======================================================================
echo UAQE ENGINE IS RUNNING!
echo ======================================================================
echo   - Frontend Dashboard : http://localhost:3000
echo   - Backend REST API   : http://127.0.0.1:8000
echo   - Interactive Docs   : http://127.0.0.1:8000/docs
echo ======================================================================
echo SHUTDOWN GUIDANCE:
echo To stop UAQE: Close the separate 'UAQE Backend' and 'UAQE Frontend'
echo console windows, or press Ctrl+C inside those windows.
echo ======================================================================
echo Press any key in this window to dismiss launcher.
pause >nul
