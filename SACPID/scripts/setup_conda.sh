#!/usr/bin/env bash
# SACPID conda setup — installs Miniconda, creates env, installs deps.
# Usage:  cd ~/project/SACPID
#         bash scripts/setup_conda.sh
# Then:   conda activate torcs_env
#         ./run_train_spot.sh 1

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SACPID_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SACPID_ROOT"

CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3}"
ENV_NAME="torcs_env"

# -----------------------------------------------------------------------------
# STEP 0: System packages for pygame (omnisafe dep)
# -----------------------------------------------------------------------------
echo "=== STEP 0/5: System packages for pygame ==="
sudo apt-get update -qq
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev libsdl2-mixer-dev libportmidi-dev
echo ""

# -----------------------------------------------------------------------------
# STEP 1: Install Miniconda (skip if already present)
# -----------------------------------------------------------------------------
echo "=== STEP 1/5: Miniconda ==="
if [ -f "$CONDA_PREFIX/bin/conda" ]; then
  echo "Miniconda already at $CONDA_PREFIX, skipping install."
else
  echo "Installing Miniconda to $CONDA_PREFIX ..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_PREFIX"
  rm -f /tmp/miniconda.sh
  echo "Miniconda installed."
fi

# Add conda to ~/.bashrc so it works in new terminals
"$CONDA_PREFIX/bin/conda" init bash 2>/dev/null || true
echo ""

# -----------------------------------------------------------------------------
# STEP 2: Init conda for this shell
# -----------------------------------------------------------------------------
echo "=== STEP 2/5: Init conda ==="
eval "$("$CONDA_PREFIX/bin/conda" shell.bash hook)"

# Accept Anaconda Terms of Service (required for pkgs/main and pkgs/r)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
echo ""

# -----------------------------------------------------------------------------
# STEP 3: Create env and install packages
# -----------------------------------------------------------------------------
echo "=== STEP 3/5: Create env '$ENV_NAME' and install packages ==="
ENV_DIR="$CONDA_PREFIX/envs/$ENV_NAME"
if [ -d "$ENV_DIR" ]; then
  echo "Removing old env at $ENV_DIR ..."
  rm -rf "$ENV_DIR"
fi
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

echo "  3a. Install PyTorch (CPU; for GPU run: conda install pytorch-cuda=11.8 -c pytorch -c nvidia)..."
conda install pytorch torchvision torchaudio -c pytorch -y

echo "  3b. Install gymnasium, numpy, scipy, wandb..."
pip install gymnasium numpy scipy wandb

echo "  3c. Install mujoco and pygame via pip (Python 3.10 has pygame 2.1.0 wheels; 3.11 does not)..."
pip install --no-cache-dir mujoco==2.3.3 pygame==2.1.0

echo "  3d. Install omnisafe (will use already-installed mujoco/pygame)..."
pip install --no-cache-dir omnisafe

echo ""

# -----------------------------------------------------------------------------
# STEP 4: Verify
# -----------------------------------------------------------------------------
echo "=== STEP 4/5: Verify ==="
python -c "import torch; import omnisafe; import gymnasium; print('OK')"
echo ""

echo "=== Setup complete ==="
echo ""
echo "IMPORTANT: In a NEW terminal (or run 'source ~/.bashrc'), conda will be in PATH."
echo "If 'conda' not found, run:  source ~/.bashrc"
echo ""
echo "Activate:  conda activate $ENV_NAME"
echo "Then:      ./run_train_spot.sh 1   or   python test_drive.py"
echo ""
