#!/usr/bin/env bash
# Run SACPID training with options that reduce risk from Spot instance interruption.
# Use a persistent EBS volume for LOG_DIR so a new instance can resume from the same path.
#
# Usage:
#   conda activate torcs_env                 # activate env first (see scripts/setup_conda.sh)
#   export SACPID_LOG_DIR=/data/sacpid-runs  # optional; default below
#   ./run_train_spot.sh 1                    # stage 1
#   ./run_train_spot.sh 2                    # stage 2
#   ./run_train_spot.sh 1 --resume-from /data/sacpid-runs/run-xxx   # resume after interrupt

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STAGE="${1:-1}"
shift || true

LOG_DIR="${SACPID_LOG_DIR:-./runs}"
mkdir -p "$LOG_DIR"

echo "Stage: $STAGE | Log/checkpoint dir: $LOG_DIR | Save freq: 10"
exec python train_sacpid.py --stage "$STAGE" --log-dir "$LOG_DIR" --save-freq 10 "$@"
