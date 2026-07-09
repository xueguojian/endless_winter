@echo off
cd /d "%~dp0"

echo ========================================
echo   Endless Winter - Setup
echo ========================================
echo.

where python >nul 2>&1
if errorlevel 1 goto :no_python

echo Python:
python --version
echo.

if exist ".venv\Scripts\python.exe" goto :has_venv
echo [1/3] Creating virtual env .venv ...
python -m venv .venv
if errorlevel 1 goto :venv_fail
goto :install_deps

:has_venv
echo .venv exists, updating dependencies...

:install_deps
echo [2/3] Installing packages (may take a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 goto :pip_fail

echo [3/3] Preparing instance config config_5555.yaml ...
".venv\Scripts\python.exe" -c "from core.config_path import ensure_config_file, PRIMARY_CONFIG_PATH; ensure_config_file(PRIMARY_CONFIG_PATH)"
".venv\Scripts\python.exe" tools\sync_config_from_example.py -c config_5555.yaml

echo.
echo ========================================
echo   Setup complete
echo ========================================
echo.
echo Next steps:
echo   1. Edit config_5555.yaml - set device.adb_path to your LDPlayer adb.exe
echo   2. Multi-instance: copy or run run_gui_5557.bat to create config_5557.yaml
echo   3. After git pull: .venv\Scripts\python.exe tools\sync_config_from_example.py --all
echo   4. Start emulator and game
echo   5. Double-click run_gui_5555.bat or run_gui_5557.bat
echo.
pause
exit /b 0

:no_python
echo [ERROR] Python not found. Install Python 3.10+ and check "Add to PATH"
echo         https://www.python.org/downloads/
pause
exit /b 1

:venv_fail
echo [ERROR] Failed to create virtual env
pause
exit /b 1

:pip_fail
echo [ERROR] Failed to install dependencies. Check network or Python version.
pause
exit /b 1
