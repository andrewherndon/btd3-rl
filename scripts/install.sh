#!/usr/bin/env bash
# HPC install. Run as the hpc user on the Pi login node:
#     sudo -iu hpc bash /clusterfs/btd3-rl/scripts/install.sh
#
# Puts an x86_64 miniforge env + deps in /clusterfs (shared NFS, visible on every
# node). The miniforge installer and pip run ON a compute node (via srun) because
# the nodes are x86_64 while the Pi login node is ARM and only orchestrates.
set -euo pipefail

CFS=/clusterfs
REPO="$CFS/btd3-rl"
MF="$CFS/miniforge3"

if [ ! -x "$MF/bin/conda" ]; then
  echo ">> downloading miniforge (x86_64) to $CFS"
  curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o "$CFS/miniforge.sh"
  echo ">> installing miniforge on a compute node (x86_64 binaries)"
  srun -N1 -n1 bash "$CFS/miniforge.sh" -b -p "$MF"
fi

echo ">> creating env 'btd' (python 3.12) + installing deps on a compute node"
srun -N1 -n1 bash -lc "
  '$MF/bin/conda' env list | grep -qE '^btd[[:space:]]' || '$MF/bin/conda' create -y -n btd python=3.12
  '$MF/envs/btd/bin/python' -m pip install -U pip
  '$MF/envs/btd/bin/python' -m pip install -e '$REPO[rl]'
"

echo ">> done. verifying import on a node..."
srun -N1 -n1 "$MF/envs/btd/bin/python" -c "import torch, sb3_contrib; print('torch', torch.__version__)"
echo ">> env: $MF/envs/btd   repo: $REPO"
