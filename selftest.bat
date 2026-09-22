@echo off
REM Arm self-test: no window, no EEG. Finds out why the arm misses its target.
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo No environment found. Run setup.bat first.
  pause
  exit /b 1
)
"%VPY%" selftest.py
echo.
pause
