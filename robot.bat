@echo off
REM Brain-controlled robot arm. Installs MuJoCo on first run.
setlocal
cd /d "%~dp0"

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo No environment found. Run start.bat first.
  pause
  exit /b 1
)

"%VPY%" -c "import mujoco" >nul 2>&1
if errorlevel 1 (
  echo Installing MuJoCo - one time only...
  "%VPY%" -m pip install "mujoco>=3.1"
  if errorlevel 1 (
    echo MuJoCo install failed. See the output above.
    pause
    exit /b 1
  )
)

echo.
echo   1  Replay your recorded session  ^(no cap needed^)
echo   2  Live EEG                      ^(cap on, connector linked^)
echo   3  Keyboard only                 ^(check the arm works^)
echo.
choice /c 123 /n /m "Choose 1, 2 or 3: "
set "MODE=replay"
if errorlevel 2 set "MODE=live"
if errorlevel 3 set "MODE=keys"

if "%MODE%"=="replay" if not exist decoder_holdout.pkl (
  echo.
  echo Training a decoder that holds back the last 20 trials, so the replay
  echo shows it data it has never seen...
  "%VPY%" train_decoder.py --holdout 20
  if errorlevel 1 ( pause & exit /b 1 )
)
if "%MODE%"=="live" if not exist decoder.pkl (
  echo.
  echo Training the decoder on your full session...
  "%VPY%" train_decoder.py
  if errorlevel 1 ( pause & exit /b 1 )
)

echo.
"%VPY%" sim_arm.py --source %MODE%
echo.
pause
