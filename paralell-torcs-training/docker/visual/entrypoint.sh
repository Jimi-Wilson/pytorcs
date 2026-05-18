#!/bin/sh

# 1. Install xautomation on the fly if missing
if ! command -v xte >/dev/null 2>&1; then
    echo "[Entrypoint] xte not found. Installing xautomation..."
    apt-get update && apt-get install -y xautomation > /dev/null
fi

echo "[Entrypoint] Launching TORCS in the background..."
/torcs/BUILD/bin/torcs &

echo "[Entrypoint] Waiting 5 seconds for GUI to initialize..."
sleep 5

echo "[Entrypoint] Sending key sequence..."
xte 'key Return' 'usleep 500000' \
    'key Return' 'usleep 500000' \
    'key Up'     'usleep 500000' \
    'key Up'     'usleep 500000' \
    'key Return' 'usleep 500000' \
    'key Return'

echo "[Entrypoint] Automation sequence complete. Holding container open..."
wait