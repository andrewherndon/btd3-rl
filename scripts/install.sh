#!/usr/bin/env bash
# One-time setup on the HPC (run from the repo root, on the login node with a
# SHARED home so the env is visible on every compute node). Installs a modern
# Python via miniforge — the RHEL system python (3.9) is too old for torch and
# must NOT be replaced — then installs the project into a conda env named "btd".
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

MF="$HOME/miniforge3"
CONDA="$MF/bin/conda"

if [ ! -x "$CONDA" ]; then
  echo ">> installing miniforge to $MF"
  URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  curl -L "$URL" -o /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$MF"
fi

if ! "$CONDA" env list | grep -qE '^btd\s'; then
  echo ">> creating conda env 'btd' (python 3.12)"
  "$CONDA" create -y -n btd python=3.12
fi

PY="$MF/envs/btd/bin/python"
echo ">> installing project deps into 'btd'"
"$PY" -m pip install -U pip
"$PY" -m pip install -e ".[rl]"

echo ">> done. Verify:  conda activate btd && python -c 'import torch, sb3_contrib; print(torch.__version__)'"
echo ">> then submit the sweep:  sbatch agent/sweep.sbatch   (watch: python agent/sweep_dashboard.py)"
