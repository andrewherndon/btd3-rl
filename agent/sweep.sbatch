#!/bin/bash
# Hyperparameter sweep as a SLURM job array. One array task = one training run.
# Submit as the hpc user from the repo:
#     sudo -iu hpc sbatch /clusterfs/btd3-rl/agent/sweep.sbatch
# If you change the grid size, match the range:
#     ... sbatch --array=0-$(python agent/sweep_configs.py --count | awk '{print $1-1}') ...
#
#SBATCH --job-name=btd-sweep
#SBATCH --array=0-5%6            # 6 configs; all at once (3 nodes x 2 jobs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2        # 2 cores/job => 2 jobs/node (even 2/2/2 spread
                                 # across all 3 nodes; RAM isn't scheduled here, so
                                 # cores are what forces the balance). Also gives
                                 # torch 2 threads for a faster PPO update.
#SBATCH --mem=1500M
#SBATCH --time=24:00:00
#SBATCH --chdir=/clusterfs/btd3-rl
#SBATCH --output=/clusterfs/btd3-rl/logs/sweep_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=2
mkdir -p /clusterfs/btd3-rl/logs
source /clusterfs/miniforge3/etc/profile.d/conda.sh
conda activate btd

ARGS=$(python agent/sweep_configs.py "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID -> $ARGS"
python agent/train.py \
  --timesteps 5000000 --n-envs 4 --backend rust --freeplay \
  --difficulties easy --eval-difficulty easy \
  --ent-coef 0.005 \
  --curriculum-p 0.3 --curriculum-min 8 --curriculum-max 45 \
  --eval-freq 100000 \
  --tb-log /clusterfs/btd3-rl/tb \
  $ARGS
