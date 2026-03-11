#!/usr/bin/env bash
# One-shot setup: install Ubuntu venv packages, create .venv, install SACPID deps.
# Uses the venv's pip by path so you never hit "externally-managed-environment".
#
# Usage:  cd ~/project/SACPID   &&   bash scripts/setup_venv.sh
# Then:   source .venv/bin/activate   before running python or ./run_train_spot.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SACPID_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SACPID_ROOT"

VENV_DIR="$SACPID_ROOT/.venv"
PYTHON="${PYTHON:-python3}"

echo "=== SACPID environment setup ==="
echo ""

# 1. Ubuntu packages for a proper venv (fixes PEP 668 / externally-managed issues)
echo "[1/4] Installing python3-full and python3-venv (sudo)..."
sudo apt-get update -qq
sudo apt-get install -y python3-full python3-venv

# 2. Create venv
echo "[2/4] Creating .venv..."
if [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi
$PYTHON -m venv "$VENV_DIR"

# 3. Install with venv's pip by path (never touch system pip)
REQUIREMENTS="$SACPID_ROOT/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
  echo "ERROR: requirements.txt not found at $REQUIREMENTS"
  echo "Run this script from the SACPID repo (e.g. cd ~/project/SACPID)."
  echo "If you used sparse checkout, ensure requirements.txt is checked out:  git sparse-checkout list"
  exit 1
fi
echo "[3/4] Installing pip, setuptools, wheel, torch, and requirements..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel --quiet
"$VENV_DIR/bin/pip" install torch torchvision torchaudio --quiet
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS" --quiet

# 4. Verify
echo "[4/4] Verifying..."
"$VENV_DIR/bin/python" -c "import torch; import omnisafe; import gymnasium; print('OK')"

echo ""
echo "=== Done ==="
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Then run:       python test_drive.py   or   ./run_train_spot.sh 1"
echo ""
