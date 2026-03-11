#!/usr/bin/env bash
# SACPID venv setup — run steps in order. If a step fails, fix and re-run from that step.
# Usage:  cd ~/project/SACPID
#         bash scripts/setup_venv.sh          # run all steps
#         bash scripts/setup_venv.sh 2        # run from step 2 only
# Then:   source .venv/bin/activate

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SACPID_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SACPID_ROOT"

VENV_DIR="$SACPID_ROOT/.venv"
PYTHON="${PYTHON:-}"
REQUIREMENTS="$SACPID_ROOT/requirements.txt"
START_FROM="${1:-1}"

# -----------------------------------------------------------------------------
# STEP 1: Ubuntu packages (run once)
# -----------------------------------------------------------------------------
if [ "$START_FROM" -le 1 ]; then
  echo "=== STEP 1/4: install a supported Python for SACPID ==="
  sudo apt-get update -qq
  sudo apt-get install -y python3.10 python3.10-venv python3.10-dev || \
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
  echo "Step 1 done."
  echo ""
fi

# -----------------------------------------------------------------------------
# STEP 2: Create .venv (run once, or to recreate)
# -----------------------------------------------------------------------------
if [ "$START_FROM" -le 2 ]; then
  echo "=== STEP 2/4: Create .venv ==="
  if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
  fi
  if [ -n "$PYTHON" ]; then
    PYTHON_BIN="$PYTHON"
  elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    echo "ERROR: python3.10 or python3.11 not found."
    echo "This project currently should not use Python 3.12 because omnisafe pulls"
    echo "pandas==2.0.3, which causes install/build issues there."
    echo "Install one of these first, then rerun:"
    echo "  sudo apt-get install -y python3.10 python3.10-venv python3.10-dev"
    echo "or"
    echo "  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev"
    exit 1
  fi
  echo "Using Python interpreter: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  echo "Step 2 done."
  echo ""
fi

# -----------------------------------------------------------------------------
# STEP 3: pip install (can re-run if it failed partway)
# -----------------------------------------------------------------------------
if [ "$START_FROM" -le 3 ]; then
  if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: requirements.txt not found at $REQUIREMENTS"
    exit 1
  fi
  echo "Using venv Python: $("$VENV_DIR/bin/python" --version 2>&1)"
  echo "=== STEP 3/4: pip install (setuptools, wheel, torch, requirements) ==="
  echo "  3a. Upgrade pip, setuptools, wheel..."
  "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
  echo "  3b. Install torch..."
  "$VENV_DIR/bin/pip" install torch torchvision torchaudio
  echo "  3c. Install pandas from wheel first (avoids building from source and pkg_resources errors)..."
  "$VENV_DIR/bin/pip" install "pandas>=2.0,<3"
  echo "  3d. Install rest of requirements.txt..."
  "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS" --no-build-isolation
  echo "Step 3 done."
  echo ""
fi

# -----------------------------------------------------------------------------
# STEP 4: Verify
# -----------------------------------------------------------------------------
if [ "$START_FROM" -le 4 ]; then
  echo "=== STEP 4/4: Verify ==="
  "$VENV_DIR/bin/python" -c "import torch; import omnisafe; import gymnasium; print('OK')"
  echo "Step 4 done."
  echo ""
fi

echo "=== Setup complete ==="
echo "Activate:  source $VENV_DIR/bin/activate"
echo "Then:      python test_drive.py   or   ./run_train_spot.sh 1"
echo ""
