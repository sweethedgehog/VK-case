@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install_local.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements-telegram.txt
pause
