#!/usr/bin/env bash
# Run this ON THE AWS INSTANCE after installing TORCS to verify the setup.
# Usage: bash check_torcs_setup.sh

set -e
echo "=== TORCS & environment check ==="

# 1. TORCS binary
if command -v torcs &>/dev/null; then
  echo "[OK] torcs is in PATH: $(which torcs)"
  torcs --help 2>/dev/null | head -1 || true
else
  echo "[FAIL] 'torcs' not found in PATH. Install vtorcs/SCR TORCS (e.g. from source)."
  exit 1
fi

# 2. Xvfb (for headless)
if command -v Xvfb &>/dev/null; then
  echo "[OK] Xvfb found: $(which Xvfb)"
else
  echo "[WARN] Xvfb not found. Install with: sudo apt-get install -y xvfb"
fi

# 3. Display (optional; needed to actually run TORCS headless)
if [ -n "${DISPLAY:-}" ]; then
  echo "[OK] DISPLAY is set: $DISPLAY"
else
  echo "[INFO] DISPLAY not set. For headless run: export DISPLAY=:99 and start Xvfb :99"
fi

# 4. Python / conda (for later training)
if command -v python3 &>/dev/null; then
  echo "[OK] python3: $(python3 --version)"
else
  echo "[WARN] python3 not found. Install Python 3.10+ and create conda env."
fi

# 5. Check for common TORCS data dirs (track files)
for d in /usr/share/games/torcs /usr/local/share/games/torcs "$HOME/.torcs"; do
  if [ -d "$d" ]; then
    echo "[OK] TORCS data dir exists: $d"
    if [ -d "$d/tracks" ]; then
      echo "     tracks: $(ls "$d/tracks" 2>/dev/null | head -5 | tr '\n' ' ')"
    fi
    break
  fi
done

echo ""
echo "=== Quick test: start TORCS (will need to be killed if headless) ==="
echo "To fully test: run 'Xvfb :99 -screen 0 1024x768x24 &' then 'export DISPLAY=:99' then 'torcs -nofuel -nodamage -nolaptime'"
echo "In TORCS GUI: Quick Race → add 'scr_server' → select Corkscrew (or your track) → New Race. Then the trainer can connect on port 3101."
echo ""
echo "Done. Fix any [FAIL] or [WARN] before cloning and running the SACPID trainer."
