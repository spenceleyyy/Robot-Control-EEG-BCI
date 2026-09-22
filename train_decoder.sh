#!/usr/bin/env bash
cd "$(dirname "$0")"
VPY=""
for p in ".venv/bin/python" ".venv/Scripts/python.exe" ".venv/bin/python3"; do
  [ -f "$p" ] && { VPY="$p"; break; }
done
[ -n "$VPY" ] || { echo "No environment found. Run ./start.sh first."; read -p "ENTER"; exit 1; }
"$VPY" train_decoder.py "$@"
echo
read -p "Press ENTER to close."
