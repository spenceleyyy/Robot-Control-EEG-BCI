#!/usr/bin/env bash
# Brain-controlled robot arm. Installs MuJoCo on first run.
cd "$(dirname "$0")"

VPY=""
for p in ".venv/bin/python" ".venv/Scripts/python.exe" ".venv/bin/python3"; do
  [ -f "$p" ] && { VPY="$p"; break; }
done
[ -n "$VPY" ] || { echo "No environment found. Run ./start.sh first."; read -p "ENTER"; exit 1; }

if ! "$VPY" -c "import mujoco" >/dev/null 2>&1; then
  echo "Installing MuJoCo - one time only..."
  "$VPY" -m pip install "mujoco>=3.1" || { echo "Install failed."; read -p "ENTER"; exit 1; }
fi

# macOS: the MuJoCo viewer must run under mjpython, which ships with mujoco
RUN="$VPY"
if [ "$(uname -s)" = "Darwin" ] && [ -f ".venv/bin/mjpython" ]; then
  RUN=".venv/bin/mjpython"
fi

echo
echo "  1  Replay your recorded session  (no cap needed)"
echo "  2  Live EEG                      (cap on, connector linked)"
echo "  3  Keyboard only                 (check the arm works)"
echo
read -p "Choose 1, 2 or 3: " c
case "$c" in 2) MODE=live;; 3) MODE=keys;; *) MODE=replay;; esac

if [ "$MODE" = replay ] && [ ! -f decoder_holdout.pkl ]; then
  echo; echo "Training a decoder that holds back the last 20 trials..."
  "$VPY" train_decoder.py --holdout 20 || { read -p "ENTER"; exit 1; }
fi
if [ "$MODE" = live ] && [ ! -f decoder.pkl ]; then
  echo; echo "Training the decoder on your full session..."
  "$VPY" train_decoder.py || { read -p "ENTER"; exit 1; }
fi

echo
"$RUN" sim_arm.py --source "$MODE"
echo
read -p "Press ENTER to close."
