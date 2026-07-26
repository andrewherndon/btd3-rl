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
# Keep the rust toolchain on shared NFS so every node sees the same cargo/rustc.
export RUSTUP_HOME="$CFS/rustup"
export CARGO_HOME="$CFS/cargo"

if [ ! -x "$MF/bin/conda" ]; then
  echo ">> downloading miniforge (x86_64) to $CFS"
  curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o "$CFS/miniforge.sh"
  echo ">> installing miniforge on a compute node (x86_64 binaries)"
  srun -N1 -n1 bash "$CFS/miniforge.sh" -b -p "$MF"
fi

echo ">> creating env 'btd' (python 3.12) + installing deps on a compute node"
# Install CPU-only torch FIRST (these nodes have no GPU) so the [rl] extra finds
# torch already satisfied and pip skips the ~2 GB CUDA build.
srun -N1 -n1 bash -lc "
  '$MF/bin/conda' env list | grep -qE '^btd[[:space:]]' || '$MF/bin/conda' create -y -n btd python=3.12
  '$MF/envs/btd/bin/python' -m pip install -U pip
  '$MF/envs/btd/bin/python' -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  '$MF/envs/btd/bin/python' -m pip install -e '$REPO[rl]'
"

# --- Rust sim backend (btd_rs) -------------------------------------------------
# The sim-rs crate is a compiled x86_64 extension (not in git), so it must be
# built on a compute node, not the ARM Pi. maturin installs the .so into the
# shared conda env, so one build is visible on every node.
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  echo ">> installing rust toolchain to $CFS (shared) on a compute node"
  srun -N1 -n1 bash -lc "
    export RUSTUP_HOME='$RUSTUP_HOME' CARGO_HOME='$CARGO_HOME'
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal
  "
fi

echo ">> building btd_rs into the env (release) on a compute node"
srun -N1 -n1 bash -lc "
  export RUSTUP_HOME='$RUSTUP_HOME' CARGO_HOME='$CARGO_HOME'
  export PATH=\"\$CARGO_HOME/bin:\$PATH\"
  '$MF/envs/btd/bin/python' -m pip install -U maturin
  '$MF/envs/btd/bin/maturin' develop --release -m '$REPO/sim-rs/Cargo.toml'
"

echo ">> done. verifying imports on a node..."
srun -N1 -n1 "$MF/envs/btd/bin/python" -c \
  "import torch, sb3_contrib, btd_rs; print('torch', torch.__version__, '| btd_rs OK')"
echo ">> env: $MF/envs/btd   repo: $REPO"
