"""Map a sweep index -> train.py flags. Used by agent/sweep.sbatch under SLURM.

    python agent/sweep_configs.py --count     # number of configs (for --array=0-(N-1))
    python agent/sweep_configs.py <index>     # prints the flags for that config

Edit the grid below to sweep whatever you want; each combination is one run.
"""

from __future__ import annotations

import itertools
import sys

# --- the grid to sweep (edit freely) --------------------------------------
GAMMA = [0.997, 0.999]          # discount — the lever that broke the hoarding plateau
ENT_COEF = [0.005, 0.01, 0.02]  # exploration
LR = [1e-4, 3e-4]               # learning rate — stability/speed
SEED = [0, 1]                   # robustness: does it win reliably, or just once?

CONFIGS = list(itertools.product(GAMMA, ENT_COEF, LR, SEED))


def config(i: int) -> dict:
    gamma, ent, lr, seed = CONFIGS[i]
    return {"gamma": gamma, "ent_coef": ent, "lr": lr, "seed": seed}


def args_for(i: int) -> str:
    c = config(i)
    return (f"--gamma {c['gamma']} --ent-coef {c['ent_coef']} "
            f"--learning-rate {c['lr']} --seed {c['seed']} "
            f"--save-path agent/models/sweep_{i:03d}/model")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(CONFIGS))
    else:
        print(args_for(int(sys.argv[1])))
