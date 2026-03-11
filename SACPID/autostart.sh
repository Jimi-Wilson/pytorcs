#!/usr/bin/env bash
# Start a virtual display and TORCS so the SACPID trainer can connect on port 3101.
# Run this before: python train_sacpid.py --stage N
#
# Usage:
#   chmod +x autostart.sh
#   ./autostart.sh
#
# Then in TORCS: Quick Race → add scr_server → choose track (e.g. Corkscrew for Stage 2/3) → New Race.
# When the race is running and listening on 3101, run the trainer in another terminal.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start virtual display for headless (e.g. AWS). No-op if DISPLAY already set.
if [ -z "${DISPLAY:-}" ]; then
  echo "Starting Xvfb on :99 ..."
  Xvfb :99 -screen 0 1024x768x24 &
  XVFB_PID=$!
  export DISPLAY=:99
  sleep 2
  echo "DISPLAY=:99 (Xvfb PID $XVFB_PID)"
fi

# Start TORCS. Use -nofuel -nodamage -nolaptime for training.
# Track is chosen in the TORCS GUI (e.g. Corkscrew for Laguna Seca).
if ! command -v torcs &>/dev/null; then
  echo "ERROR: 'torcs' not in PATH. Install TORCS/vtorcs first."
  exit 1
fi

echo "Launching TORCS (port 3101). Select track and start race in the GUI."
exec torcs -nofuel -nodamage -nolaptime
