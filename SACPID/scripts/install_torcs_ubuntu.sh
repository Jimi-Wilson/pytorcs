#!/usr/bin/env bash
# Install TORCS 1.3.7 with SCR server (port 3101) on Ubuntu 20.04/22.04.
# Run on the AWS instance (or any Ubuntu): bash install_torcs_ubuntu.sh
# Requires: sudo, git, build-essential
#
# After install: run TORCS with   torcs -nofuel -nodamage -nolaptime
# In GUI: Quick Race → add "scr_server" → select track (e.g. Corkscrew) → New Race.
# The Python client (snakeoil3, gym_torcs) connects on port 3101.

set -e

echo "=== Installing TORCS 1.3.7 (SCR server, port 3101) ==="

# Build dependencies (Ubuntu 20.04/22.04)
echo "[1/4] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
  build-essential git \
  libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
  libplib-dev libopenal-dev libalut-dev \
  libxi-dev libxmu-dev libxrender-dev libxrandr-dev libxt-dev libxxf86vm-dev \
  libpng-dev libvorbis-dev zlib1g-dev

# Clone pre-patched TORCS (includes scr_server)
TORCS_SRC="$HOME/torcs-1.3.7"
if [ -d "$TORCS_SRC" ]; then
  echo "[2/4] Source dir $TORCS_SRC exists; skipping clone. Remove it to re-clone."
else
  echo "[2/4] Cloning fmirus/torcs-1.3.7 (SCR-patched)..."
  git clone https://github.com/fmirus/torcs-1.3.7.git "$TORCS_SRC"
fi
cd "$TORCS_SRC"

# Build and install to /usr/local so 'torcs' is in PATH
echo "[3/4] Configuring and building (this may take a few minutes)..."
export CFLAGS="-fPIC"
export CPPFLAGS="$CFLAGS"
export CXXFLAGS="$CFLAGS"
./configure --prefix=/usr/local
make -j"$(nproc)"
sudo make install
sudo make datainstall

echo "[4/4] Verifying..."
if command -v torcs &>/dev/null; then
  echo "  torcs binary: $(which torcs)"
  torcs --help 2>/dev/null | head -1 || true
  echo ""
  echo "=== TORCS installed successfully ==="
  echo "Run:  torcs -nofuel -nodamage -nolaptime"
  echo "In GUI: Quick Race → add 'scr_server' → choose track → New Race."
  echo "Then your SACPID trainer can connect on port 3101."
else
  echo "  WARNING: 'torcs' not in PATH. You may need to run:  export PATH=/usr/local/bin:\$PATH"
  echo "  Or log out and back in. Binary is at: /usr/local/bin/torcs"
fi
