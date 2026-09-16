@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
  call setup_venv.bat
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

start "MuseChart" ".venv\Scripts\pythonw.exe" -X utf8 gui.py
