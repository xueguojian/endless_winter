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
echo [1/2] Creating virtual env .venv ...
python -m venv .venv
if errorlevel 1 goto :venv_fail
goto :install_deps

:has_venv
echo .venv exists, updating dependencies...

:install_deps
echo [2/2] Installing packages (may take a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 goto :pip_fail

echo [3/3] Sync config from template (if config.yaml exists)...
if exist "config.yaml" (
  ".venv\Scripts\python.exe" tools\sync_config_from_example.py
) else if exist "config.example.yaml" (
  echo Creating config.yaml from config.example.yaml ...
  copy /Y config.example.yaml config.yaml >nul
  echo   Please edit config.yaml - set device.adb_path to your LDPlayer adb.exe
) else (
  echo [WARN] config.example.yaml not found, skip config setup
)

echo.
echo ========================================
echo   Setup complete
echo ========================================
echo.
echo Next steps:
echo   1. Edit config.yaml - set device.adb_path to your LDPlayer adb.exe
echo   2. Set device.adb_port (default 5555)
echo   3. After git pull, run: .venv\Scripts\python.exe tools\sync_config_from_example.py
echo   4. Start emulator and game
echo   5. Double-click run_gui.vbs
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
