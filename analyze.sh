#!/usr/bin/env bash
# Re-run the ERD analysis on a recording, using the right interpreter.
#   ./analyze.sh                      -> newest recording in data/
#   ./analyze.sh data/mi_xxx_raw.fif  -> a specific one
cd "$(dirname "$0")"

VPY=""
for p in ".venv/bin/python" ".venv/Scripts/python.exe" ".venv/bin/python3"; do
  [ -f "$p" ] && { VPY="$p"; break; }
done

if [ -z "$VPY" ]; then
  echo "No environment found. Run ./start.sh first."
  read -p "Press ENTER to close."
  exit 1
fi

"$VPY" analyze_erd.py "$@"
echo
read -p "Press ENTER to close."
