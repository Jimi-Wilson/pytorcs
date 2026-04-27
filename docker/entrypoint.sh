#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=:1
fi

if [[ "${DISPLAY}" == ":1" ]]; then
  if ! pgrep -f "Xvfb :1" >/dev/null 2>&1; then
    Xvfb :1 -screen 0 1280x720x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    sleep 1
  fi
fi

exec "$@"
