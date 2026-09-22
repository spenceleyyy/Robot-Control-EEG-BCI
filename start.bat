@echo off
REM One-click launcher. Builds the environment on first run, then records.
setlocal
cd /d "%~dp0"

set "VPY=.venv\Scripts\python.exe"

REM --- rebuild the venv if it is missing or incomplete ---------------------
if exist ".venv" if not exist "%VPY%" (
  echo Existing .venv has no interpreter in it - rebuilding from scratch.
  rmdir /s /q ".venv"
)

if not exist "%VPY%" (
  echo Creating the environment. This takes a minute.
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo Could not create the environment.
    echo Is Python 3.9+ installed and ticked "Add to PATH"?
    echo Check with:  python --version
    pause
    exit /b 1
  )
  if not exist "%VPY%" (
    echo.
    echo The venv was created but contains no python.exe.
    pause
    exit /b 1
  )
  "%VPY%" -m pip install --upgrade pip >nul 2>&1
  echo Installing dependencies...
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Dependency install failed. Check the output above.
    pause
    exit /b 1
  )
)

REM --- verify the libraries actually import --------------------------------
"%VPY%" -c "import importlib.util,sys;m=[x for x in ('numpy','scipy','mne','matplotlib','pygame','pylsl','sklearn') if importlib.util.find_spec(x) is None];print('Missing:',', '.join(m)) if m else print('All dependencies present.');sys.exit(1 if m else 0)"
if errorlevel 1 (
  echo.
  echo Dependencies are not importable. Delete the .venv folder and rerun.
  pause
  exit /b 1
)

echo.
"%VPY%" check_stream.py
if errorlevel 1 (
  echo.
  echo Stream check failed. Fix the above, then run this again.
  pause
  exit /b 1
)

echo.
echo Next: the live signal quality monitor. Put the cap on and gel
echo while watching it. Press ENTER in that screen when you are happy.
pause
"%VPY%" signal_check.py

echo.
echo Ready to record? This runs about 18 minutes.
pause
"%VPY%" session.py

echo.
pause
