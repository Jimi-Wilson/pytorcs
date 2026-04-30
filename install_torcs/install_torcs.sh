#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_TRACK_DIR="$REPO_ROOT/install_torcs/assets/tracks/corkscrew"
BUILD_DIR="/tmp/torcs-1.3.7-scr-build"
SRC_REPO="https://github.com/fmirus/torcs-1.3.7.git"
CORKSCREW_REPO_URL="https://github.com/jeremybennett/torcs.git"
CORKSCREW_REPO_REF="r1-3-1"
CORKSCREW_REPO_PATH="data/tracks/road/corkscrew"

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

disable_img_server_driver() {
  local drivers_dir="/usr/local/lib/torcs/drivers"
  local img_server_dir="$drivers_dir/img_server"
  local img_server_disabled_dir="$drivers_dir/img_server.disabled"

  if [[ -d "$img_server_dir" ]]; then
    say "Disabling img_server driver to prevent player-selection crash..."
    sudo rm -rf "$img_server_disabled_dir" || true
    sudo mv "$img_server_dir" "$img_server_disabled_dir"
  fi
}

copy_corkscrew_assets() {
  local source_track_dir="$ASSET_TRACK_DIR"
  local tmp_track_root=""

  if [[ ! -f "$source_track_dir/corkscrew.acc" || ! -f "$source_track_dir/corkscrew.xml" ]]; then
    say "Local Corkscrew assets are incomplete. Fetching full track from upstream..."
    tmp_track_root="$(mktemp -d)"
    if ! git clone --depth 1 --filter=blob:none --no-checkout \
      "$CORKSCREW_REPO_URL" -b "$CORKSCREW_REPO_REF" "$tmp_track_root/repo" >/dev/null 2>&1; then
      die "Failed to clone Corkscrew source repository: $CORKSCREW_REPO_URL"
    fi
    (
      cd "$tmp_track_root/repo"
      git checkout "$CORKSCREW_REPO_REF" -- "$CORKSCREW_REPO_PATH" >/dev/null 2>&1
    ) || die "Failed to checkout Corkscrew track files from upstream repository."
    source_track_dir="$tmp_track_root/repo/$CORKSCREW_REPO_PATH"
  fi

  [[ -d "$source_track_dir" ]] || die "Corkscrew source directory missing: $source_track_dir"
  [[ -f "$source_track_dir/corkscrew.xml" ]] || die "Missing required file: corkscrew.xml"
  [[ -f "$source_track_dir/corkscrew.acc" ]] || die "Missing required file: corkscrew.acc"

  local target_data_dir
  if [[ -d /usr/local/share/games/torcs ]]; then
    target_data_dir="/usr/local/share/games/torcs"
  elif [[ -d /usr/share/games/torcs ]]; then
    target_data_dir="/usr/share/games/torcs"
  else
    die "Could not find TORCS data directory after install."
  fi

  local target_track_dir="$target_data_dir/tracks/road/corkscrew"
  say "Copying Corkscrew assets to $target_track_dir..."
  sudo mkdir -p "$target_track_dir"
  sudo cp -a "$source_track_dir"/. "$target_track_dir"/
  [[ -n "$tmp_track_root" ]] && rm -rf "$tmp_track_root"
}

set_scr_server_default_car() {
  local target_data_dir
  if [[ -d /usr/local/share/games/torcs ]]; then
    target_data_dir="/usr/local/share/games/torcs"
  elif [[ -d /usr/share/games/torcs ]]; then
    target_data_dir="/usr/share/games/torcs"
  else
    die "Could not find TORCS data directory after install."
  fi

  local scr_server_xml="$target_data_dir/drivers/scr_server/scr_server.xml"
  [[ -f "$scr_server_xml" ]] || die "Missing SCR config file: $scr_server_xml"

  say "Setting SCR default cars to car1-ow1 in $scr_server_xml..."
  sudo sed -E -i \
    's#(<attstr name="car name" val=")car1-trb1("></attstr>)#\1car1-ow1\2#g' \
    "$scr_server_xml"

  if ! sudo grep -q '<attstr name="car name" val="car1-ow1"></attstr>' "$scr_server_xml"; then
    die "Failed to set SCR default car mapping to car1-ow1 in $scr_server_xml"
  fi
}

run_sanity_check() {
  say "Running sanity check: torcs -t 100000"
  set +e
  torcs -t 100000 >/tmp/torcs_install_sanity.log 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[install_torcs] torcs -t 100000 failed with exit code $rc" >&2
    echo "[install_torcs] Last output:" >&2
    tail -n 40 /tmp/torcs_install_sanity.log >&2
    exit $rc
  fi
  say "Sanity check passed."
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
  disable_img_server_driver
  copy_corkscrew_assets
  set_scr_server_default_car
  run_sanity_check
  cat <<'EOF'
[install_torcs] Success: TORCS 1.3.7 (SCR-patched) is installed.
[install_torcs] Next step: run `python3 install_torcs/check_install.py`
EOF
}

main "$@"
