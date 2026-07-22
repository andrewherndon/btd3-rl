#!/bin/bash
# Per-node performance + sanity check. Runs agent/bench.py once on every node so
# you can confirm the env/deps work and see each node's steps/sec BEFORE
# committing to real training. Submit from the repo root:
#     sbatch agent/bench.sbatch
#     cat logs/bench_*.out          # per-node results
#
#SBATCH --job-name=btd-bench
#SBATCH --nodes=8                 # set to your node count (check: sinfo)
#SBATCH --ntasks-per-node=1       # one benchmark per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --output=logs/bench_%j.out
# #SBATCH --partition=compute     # uncomment + set to your partition

set -euo pipefail
export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate btd

# --label prefixes each node's output with its task id.
srun --label python agent/bench.py --timesteps 30000
