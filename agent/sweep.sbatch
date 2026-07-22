#!/bin/bash
# Hyperparameter sweep as a SLURM job array. One array task = one training run.
# Submit from the repo root:
#     sbatch --array=0-$(($(python agent/sweep_configs.py --count)-1)) agent/sweep.sbatch
# (or hardcode the range below to match `sweep_configs.py --count`).
#
#SBATCH --job-name=btd-sweep
#SBATCH --array=0-23%16          # 24 configs; run at most 16 at once (tune to node RAM)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1        # training is single-core (env-bound)
#SBATCH --mem=1500M              # ~2 jobs/node on 3.5G nodes; raise if jobs OOM
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%A_%a.out
# #SBATCH --partition=compute    # uncomment + set to your partition (check: sinfo)

set -euo pipefail
export OMP_NUM_THREADS=1         # keep each job truly single-threaded (net is tiny)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Activate the conda env created by scripts/install.sh.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate btd

ARGS=$(python agent/sweep_configs.py "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID -> $ARGS"
python agent/train.py \
  --timesteps 5000000 --n-envs 4 \
  --difficulties easy,medium,hard --eval-difficulty hard \
  --eval-freq 100000 \
  $ARGS
