#!/usr/bin/env bash
set -euo pipefail

start_xvfb() {
  if ! pgrep -f "Xvfb :1" >/dev/null 2>&1; then
    Xvfb :1 -screen 0 1280x720x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    sleep 1
  fi
}

if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=:1
fi

if [[ "${DISPLAY}" == ":1" ]]; then
  start_xvfb
fi

# Mesa: prefer swap-on-vblank when a real GPU / compositor honors it (harmless under Xvfb).
if [[ -z "${vblank_mode:-}" ]]; then
  export vblank_mode=2
fi

# Cap TORCS OpenGL frame rate (default 60). Set TORCS_FPS_LIMIT=0 to run torcs without strangle.
_fps_limit="${TORCS_FPS_LIMIT:-60}"
if [[ "$#" -ge 1 && "$_fps_limit" != "0" ]] && command -v strangle >/dev/null 2>&1; then
  if [[ "$(basename "$1")" == "torcs" ]]; then
    exec strangle "${_fps_limit}" "$@"
  fi
fi

exec "$@"
