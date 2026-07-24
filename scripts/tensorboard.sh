#!/usr/bin/env bash
# Launch a TensorBoard web server on the Pi login node, reading the shared TB
# logs the training jobs write to /clusterfs. Then open it from your Mac browser:
#     http://<pi-tailscale-ip>:6006     (e.g. http://100.68.119.28:6006)
#
# Run as hpc:  sudo -iu hpc bash /clusterfs/btd3-rl/scripts/tensorboard.sh
#
# The TB server runs on the Pi (ARM) in its own small venv — separate from the
# x86_64 training env — since only the Pi serves the dashboard.
set -euo pipefail

TBVENV=/clusterfs/tb-venv
LOGDIR=/clusterfs/btd3-rl/tb
PORT=6006
mkdir -p "$LOGDIR"

if [ ! -x "$TBVENV/bin/tensorboard" ]; then
  echo ">> first run: creating tensorboard venv on the Pi"
  python3 -m venv "$TBVENV"
  "$TBVENV/bin/pip" install -q -U pip tensorboard
fi

IP=$(hostname -I | tr ' ' '\n' | grep -E '^100\.' | head -1)   # tailscale IP if present
echo ">> TensorBoard: http://${IP:-<pi-ip>}:${PORT}   (open on your Mac; Ctrl-C to stop)"
exec "$TBVENV/bin/tensorboard" --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT"
