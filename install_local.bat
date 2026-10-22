@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo === HSEHelper local setup ===
where py >nul 2>nul
if errorlevel 1 goto no_python
py -3.12 --version >nul 2>nul
if not errorlevel 1 goto use312
py -3 --version >nul 2>nul
if errorlevel 1 goto no_python
set "PY_CMD=py -3"
goto setup
:use312
set "PY_CMD=py -3.12"
:setup
if not exist ".venv\Scripts\python.exe" (
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto venv_error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto install_error
python -m pip install -r requirements-local.txt
if errorlevel 1 goto install_error
if not exist ".env" copy ".env.local.example" ".env" >nul
echo.
echo Setup complete.
echo Next: edit .env and add GEMINI_API_KEY.
echo Then run collect_courses.bat if data\courses.json is missing.
echo Finally run start_site.bat.
pause
exit /b 0
:no_python
echo Python Launcher not found. Install Python 3.12 and enable the Python Launcher.
pause
exit /b 1
:venv_error
echo Could not create the virtual environment. Check your Python installation.
pause
exit /b 1
:install_error
echo Package installation failed. Check the last error above and your internet connection.
pause
exit /b 1
