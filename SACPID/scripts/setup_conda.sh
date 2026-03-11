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
# STEP 1: Install Miniconda (skip if already present)
# -----------------------------------------------------------------------------
echo "=== STEP 1/4: Miniconda ==="
if [ -f "$CONDA_PREFIX/bin/conda" ]; then
  echo "Miniconda already at $CONDA_PREFIX, skipping install."
else
  echo "Installing Miniconda to $CONDA_PREFIX ..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_PREFIX"
  rm -f /tmp/miniconda.sh
  echo "Miniconda installed."
fi
echo ""

# -----------------------------------------------------------------------------
# STEP 2: Init conda for this shell
# -----------------------------------------------------------------------------
echo "=== STEP 2/4: Init conda ==="
eval "$("$CONDA_PREFIX/bin/conda" shell.bash hook)"
echo ""

# -----------------------------------------------------------------------------
# STEP 3: Create env and install packages
# -----------------------------------------------------------------------------
echo "=== STEP 3/4: Create env '$ENV_NAME' and install packages ==="
if conda env list | grep -qw "$ENV_NAME"; then
  echo "Env '$ENV_NAME' exists. Recreate? Run: conda env remove -n $ENV_NAME"
  echo "Continuing with existing env..."
else
  conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

echo "  3a. Install PyTorch (CPU; for GPU run: conda install pytorch-cuda=11.8 -c pytorch -c nvidia)..."
conda install pytorch torchvision torchaudio -c pytorch -y

echo "  3b. Install gymnasium, numpy, scipy, wandb..."
pip install gymnasium numpy scipy wandb

echo "  3c. Install omnisafe..."
pip install omnisafe

echo ""

# -----------------------------------------------------------------------------
# STEP 4: Verify
# -----------------------------------------------------------------------------
echo "=== STEP 4/4: Verify ==="
python -c "import torch; import omnisafe; import gymnasium; print('OK')"
echo ""

echo "=== Setup complete ==="
echo "Activate:  conda activate $ENV_NAME"
echo "Then:      ./run_train_spot.sh 1   or   python test_drive.py"
echo ""
