#!/bin/bash
# Per-node performance + sanity check. Runs agent/bench.py on every compute node.
# Submit as the hpc user:  sudo -iu hpc sbatch /clusterfs/btd3-rl/agent/bench.sbatch
# (or use scripts/bench.sh to submit + wait + print results).
#
#SBATCH --job-name=btd-bench
#SBATCH --nodes=3                 # this cluster has 3 compute nodes
#SBATCH --ntasks-per-node=1       # one benchmark per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --chdir=/clusterfs/btd3-rl
#SBATCH --output=/clusterfs/btd3-rl/logs/bench_%j.out

set -euo pipefail
export OMP_NUM_THREADS=1
mkdir -p /clusterfs/btd3-rl/logs
source /clusterfs/miniforge3/etc/profile.d/conda.sh
conda activate btd

# --label prefixes each node's output with its task id. BACKEND is exported by
# scripts/bench.sh (default python); "rust" uses the btd_rs sim.
srun --label python agent/bench.py --timesteps 10000 --backend "${BACKEND:-python}"
