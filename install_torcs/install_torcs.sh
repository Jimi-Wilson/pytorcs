#!/usr/bin/env bash

# Please run this script on your machine to confirm it works.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="/tmp/torcs-1.3.7-scr-build"
SRC_REPO="https://github.com/fmirus/torcs-1.3.7.git"

APT_DEPS=(
  libglib2.0-dev
  libgl1-mesa-dev
  libglu1-mesa-dev
  freeglut3-dev
  libplib-dev
  libopenal-dev
  libalut-dev
  libxi-dev
  libxmu-dev
  libxxf86vm-dev
  libxrender-dev
  libxrandr-dev
  libpng-dev
  libvorbis-dev
)

say() {
  echo "[install_torcs] $*"
}

die() {
  echo "[install_torcs] ERROR: $*" >&2
  exit 1
}

require_sudo() {
  if ! command -v sudo >/dev/null 2>&1; then
    die "sudo is required for install steps."
  fi
}

check_os() {
  if [[ ! -f /etc/os-release ]]; then
    die "Cannot detect OS. This installer supports Ubuntu/Debian only."
  fi
  # shellcheck disable=SC1091
  source /etc/os-release
  local distro="${ID:-}"
  local like="${ID_LIKE:-}"
  if [[ "$distro" != "ubuntu" && "$distro" != "debian" && "$like" != *"debian"* ]]; then
    cat >&2 <<'EOF'
[install_torcs] This installer supports Ubuntu/Debian only.
[install_torcs] If you are on Windows, use the Docker workflow for this project.
EOF
    exit 1
  fi
}

remove_existing_torcs() {
  say "Removing existing TORCS package installs (if present)..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y torcs torcs-data >/dev/null 2>&1 || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y torcs torcs-data >/dev/null 2>&1 || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get autoremove -y >/dev/null 2>&1 || true

  say "Removing common source-install TORCS paths under /usr/local..."
  sudo rm -f /usr/local/bin/torcs || true
  sudo rm -rf /usr/local/share/games/torcs || true
  sudo rm -rf /usr/local/lib/games/torcs || true
  sudo rm -rf /usr/local/lib/torcs || true

  hash -r
  if command -v torcs >/dev/null 2>&1; then
    die "torcs is still found at $(command -v torcs). Remove it manually, then rerun."
  fi
}

install_dependencies() {
  say "Installing build dependencies with apt..."
  if ! sudo DEBIAN_FRONTEND=noninteractive apt-get update; then
    cat >&2 <<'EOF'
[install_torcs] WARNING: `apt-get update` failed.
[install_torcs] This is usually caused by unrelated broken third-party apt sources.
[install_torcs] Continuing with current package indexes and attempting dependency install.
[install_torcs] If install fails, fix/remove invalid apt sources, then rerun this script.
EOF
  fi
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git \
    "${APT_DEPS[@]}"
}

build_from_source() {
  say "Cloning pre-patched TORCS 1.3.7 source..."
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  git clone --depth 1 "$SRC_REPO" "$BUILD_DIR/src"

  cd "$BUILD_DIR/src"
  export CFLAGS="-fPIC"
  export CPPFLAGS="-fPIC"
  export CXXFLAGS="-fPIC"

  say "Configuring and building TORCS from source..."
  ./configure
  make -j1
  sudo make install
  sudo make datainstall
}

cleanup() {
  rm -rf "$BUILD_DIR"
}

main() {
  trap cleanup EXIT
  check_os
  require_sudo
  remove_existing_torcs
  install_dependencies
  build_from_source

  cat <<'EOF'
[install_torcs] Success: TORCS 1.3.7 (SCR-patched) is installed.
EOF
}

main "$@"
