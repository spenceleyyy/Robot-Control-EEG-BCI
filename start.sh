#!/usr/bin/env bash
# One-click launcher. Builds the environment on first run, then records.
cd "$(dirname "$0")"

fail() { echo; echo "ERROR: $*"; echo; read -p "Press ENTER to close."; exit 1; }

case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "NOTE: this looks like Git Bash on Windows."
    echo "start.bat is the better launcher there, but this will work too."
    echo
    ;;
esac

# --- find a usable python -------------------------------------------------
PY=""
for c in python3 python py; do
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' \
      >/dev/null 2>&1; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || fail "No Python 3.9+ found on PATH. Install Python, then retry."
echo "Using $("$PY" --version 2>&1) from $(command -v "$PY")"

# --- locate the venv interpreter, whichever layout it used ----------------
venv_py() {
  for p in ".venv/bin/python" ".venv/Scripts/python.exe" ".venv/bin/python3"; do
    [ -f "$p" ] && { echo "$p"; return; }
  done
}

VPY="$(venv_py)"

if [ -z "$VPY" ]; then
  if [ -d ".venv" ]; then
    echo "Existing .venv has no interpreter in it - rebuilding from scratch."
    rm -rf .venv || fail "Could not remove the broken .venv directory."
  fi

  echo "Creating the environment. This takes a minute."
  if ! "$PY" -m venv .venv; then
    echo
    echo "Creating the virtual environment failed."
    echo "On Debian/Ubuntu this usually means the venv module is missing:"
    echo "    sudo apt install python3-venv python3-pip"
    fail "Install that, then run this script again."
  fi

  VPY="$(venv_py)"
  [ -n "$VPY" ] || {
    echo "The venv was created but contains no interpreter. It holds:"
    ls -la .venv 2>/dev/null
    fail "Something is wrong with this Python installation."
  }

  echo "Installing dependencies..."
  "$VPY" -m pip install --upgrade pip >/dev/null 2>&1 || true
  "$VPY" -m pip install -r requirements.txt \
    || fail "Dependency install failed. Check the output above."
fi

echo "Environment: $VPY"

# --- verify the libraries actually import ---------------------------------
"$VPY" - <<'PYEOF' || fail "Dependencies are not importable. Delete .venv and rerun."
import importlib.util
import sys
missing = [m for m in ("numpy", "scipy", "mne", "matplotlib", "pygame",
                       "pylsl", "sklearn")
           if importlib.util.find_spec(m) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    sys.exit(1)
print("All dependencies present.")
PYEOF

# --- run ------------------------------------------------------------------
echo
"$VPY" check_stream.py || {
  echo
  echo "Stream check failed. Fix the above, then run this again."
  read -p "Press ENTER to close."
  exit 1
}

echo
echo "Next: the live signal quality monitor. Put the cap on and gel"
echo "while watching it. Press ENTER in that screen when you are happy."
read -p "Press ENTER to open the monitor."
"$VPY" signal_check.py

echo
read -p "Ready to record? This runs about 18 minutes. Press ENTER."
"$VPY" session.py

echo
read -p "Press ENTER to close."
