@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  call setup_venv.bat
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

if "%~1"=="" (
  ".venv\Scripts\python.exe" -X utf8 generate_mdm.py
) else (
  ".venv\Scripts\python.exe" -X utf8 generate_mdm.py %*
)

pause
