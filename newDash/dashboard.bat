@echo off
REM Launch the session dashboard. Put this in the mi-testbed folder.
setlocal
cd /d "%~dp0"

set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if exist ".venv\bin\python.exe"     set PY=.venv\bin\python.exe
if exist ".venv\bin\python"         set PY=.venv\bin\python
if "%PY%"=="" (
  echo No virtualenv found. Creating one...
  python -m venv .venv || goto :fail
  set PY=.venv\Scripts\python.exe
)

"%PY%" -c "import pygame, cv2" 2>nul
if errorlevel 1 (
  echo Installing dashboard dependencies...
  "%PY%" -m pip install --quiet pygame opencv-python || goto :fail
)

"%PY%" bci_dashboard.py %*
goto :eof

:fail
echo.
echo Setup failed. Run the commands above by hand to see the error.
pause
