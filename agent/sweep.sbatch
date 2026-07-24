#!/bin/bash
# Hyperparameter sweep as a SLURM job array. One array task = one training run.
# Submit as the hpc user from the repo:
#     sudo -iu hpc sbatch /clusterfs/btd3-rl/agent/sweep.sbatch
# If you change the grid size, match the range:
#     ... sbatch --array=0-$(python agent/sweep_configs.py --count | awk '{print $1-1}') ...
#
#SBATCH --job-name=btd-sweep
#SBATCH --array=0-23%6           # 24 configs; ~6 at once (3 nodes x ~2 by 3.5G RAM)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1        # training is single-core (env-bound)
#SBATCH --mem=1500M
#SBATCH --time=24:00:00
#SBATCH --chdir=/clusterfs/btd3-rl
#SBATCH --output=/clusterfs/btd3-rl/logs/sweep_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1
mkdir -p /clusterfs/btd3-rl/logs
source /clusterfs/miniforge3/etc/profile.d/conda.sh
conda activate btd

ARGS=$(python agent/sweep_configs.py "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID -> $ARGS"
python agent/train.py \
  --timesteps 5000000 --n-envs 4 \
  --difficulties easy,medium,hard --eval-difficulty hard \
  --eval-freq 100000 \
  $ARGS
