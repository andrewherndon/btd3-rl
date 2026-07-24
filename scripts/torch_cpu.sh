#!/usr/bin/env bash
# Replace the default CUDA torch build with CPU-only torch (these nodes have no
# GPU), reclaiming ~2 GB of unused libs and shrinking each job's memory. Runs the
# pip ops on a compute node (x86_64):
#     sudo -iu hpc srun -N1 -n1 bash /clusterfs/btd3-rl/scripts/torch_cpu.sh
set -euo pipefail
PY=/clusterfs/miniforge3/envs/btd/bin/python

echo ">> removing CUDA torch + nvidia/cuda/triton libs"
"$PY" -m pip list 2>/dev/null | grep -iE '^(torch|triton|cuda-|nvidia-)' | awk '{print $1}' \
  | xargs -r "$PY" -m pip uninstall -y

echo ">> installing CPU-only torch"
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu

"$PY" -c 'import torch; print("torch", torch.__version__)'
