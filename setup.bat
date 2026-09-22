@echo off
REM Builds (or repairs) the Python environment. Nothing else.
REM Run this once after moving the project folder.
setlocal
cd /d "%~dp0"

set "VPY=.venv\Scripts\python.exe"

if exist ".venv" if not exist "%VPY%" (
  echo Existing .venv is broken - removing it.
  rmdir /s /q ".venv"
)

if not exist "%VPY%" (
  echo Creating the environment in:
  echo   %CD%\.venv
  echo.
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo Could not create the environment. Is Python installed and on PATH?
    echo Check with:  python --version
    pause
    exit /b 1
  )
)

echo Installing packages. This takes a few minutes the first time...
"%VPY%" -m pip install --upgrade pip >nul 2>&1
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package install failed. See the messages above.
  pause
  exit /b 1
)

echo.
"%VPY%" -c "import importlib.util,sys;m=[x for x in ('numpy','scipy','mne','matplotlib','pygame','pylsl','sklearn','mujoco') if importlib.util.find_spec(x) is None];print('Missing:',', '.join(m)) if m else print('All packages present, including MuJoCo.');sys.exit(1 if m else 0)"
if errorlevel 1 (
  pause
  exit /b 1
)

echo.
echo Environment ready. You can now run robot.bat
pause
