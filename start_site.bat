@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install_local.bat first.
  pause
  exit /b 1
)
if not exist ".env" (
  copy ".env.local.example" ".env" >nul
  echo Created .env. Add GEMINI_API_KEY and run start_site.bat again.
  notepad .env
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
