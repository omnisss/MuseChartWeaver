@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating inference\.venv ...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto :failed
) else (
  echo [1/3] Reusing inference\.venv
)

echo [2/3] Updating pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo [3/3] Installing dependencies ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo Virtual environment is ready: %CD%\.venv
exit /b 0

:failed
echo Failed to prepare the virtual environment.
exit /b 1
