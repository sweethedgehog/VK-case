@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install_local.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scrape_hse.py --limit 30 --max-pages 5 --year 2026 --course-filter 1.1.1.4
if errorlevel 1 (
  echo Course collection failed. Check the error above and your internet connection.
) else (
  echo Finished. Check data\courses.json.
)
pause
