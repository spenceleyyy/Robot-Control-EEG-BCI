@echo off
REM Re-run the ERD analysis on a recording, using the right interpreter.
REM   analyze.bat                      -> newest recording in data\
REM   analyze.bat data\mi_xxx_raw.fif  -> a specific one
setlocal
cd /d "%~dp0"

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo No environment found. Run start.bat first.
  pause
  exit /b 1
)

"%VPY%" analyze_erd.py %*
echo.
pause
