@echo off
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo No environment found. Run start.bat first.
  pause
  exit /b 1
)
"%VPY%" bci_feedback.py %*
echo.
pause
