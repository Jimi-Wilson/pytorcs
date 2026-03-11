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
PYTHON="${PYTHON:-python3}"
REQUIREMENTS="$SACPID_ROOT/requirements.txt"
START_FROM="${1:-1}"

# -----------------------------------------------------------------------------
# STEP 1: Ubuntu packages (run once)
# -----------------------------------------------------------------------------
if [ "$START_FROM" -le 1 ]; then
  echo "=== STEP 1/4: apt install python3-full python3-venv ==="
  sudo apt-get update -qq
  sudo apt-get install -y python3-full python3-venv
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
  $PYTHON -m venv "$VENV_DIR"
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
  echo "=== STEP 3/4: pip install (setuptools, wheel, torch, requirements) ==="
  echo "  3a. Upgrade pip, setuptools, wheel..."
  "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
  echo "  3b. Install torch..."
  "$VENV_DIR/bin/pip" install torch torchvision torchaudio
  echo "  3c. Install requirements.txt (--no-build-isolation so build sees setuptools)..."
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
