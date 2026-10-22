@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install_local.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m py_compile assistant.py semantic_search.py scrape_hse.py server.py bot.py conversation_store.py query_utils.py tests.py api_smoke_tests.py run_benchmark.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe tests.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe api_smoke_tests.py
if errorlevel 1 goto failed
echo All local checks passed.
pause
exit /b 0
:failed
echo A check failed. Copy the error output for troubleshooting.
pause
exit /b 1
