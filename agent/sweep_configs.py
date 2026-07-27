"""Map a sweep index -> train.py flags. Used by agent/sweep.sbatch under SLURM.

    python agent/sweep_configs.py --count     # number of configs (for --array=0-(N-1))
    python agent/sweep_configs.py <index>     # prints the flags for that config

Current wave: **easy freeplay, push past the ~round-56 warm-start wall.** Six
explicit strategies (not a product grid) — each isolates one lever against the
control (#0). Flags common to ALL configs (backend/freeplay/difficulty/ent_coef/
curriculum/timesteps) live in sweep.sbatch; only the per-config levers are here.

Bases (must exist on the cluster under agent/models/, copied from the Mac):
  run13         = easy round-50 winner (reaches ~56 in freeplay untuned)
  freeplay_warm = the Mac warm-start run that already reached ~56
"""

from __future__ import annotations

import sys

RUN13 = "agent/models/run13/best_model"
FREEPLAY_WARM = "agent/models/freeplay_warm/best_model"

# Each dict: name (for save-path/tb), init_from (None = from scratch), gamma, lr,
# and optional milestone_bonus / milestone_every.
CONFIGS = [
    # 0: control — warm-start run13, baseline extend
    dict(name="control",     init_from=RUN13,         gamma=0.999,  lr=1e-4),
    # 1: higher gamma — longer horizon for the distal freeplay win
    dict(name="gamma9995",   init_from=RUN13,         gamma=0.9995, lr=1e-4),
    # 2: from scratch — control for "what does warm-start buy?" (lr 3e-4: fresh
    #    init needs faster learning, per the sweep)
    dict(name="scratch",     init_from=None,          gamma=0.999,  lr=3e-4),
    # 3: single milestone at round 50 — restore the reachable pull freeplay deleted
    dict(name="milestone50", init_from=RUN13,         gamma=0.999,  lr=1e-4,
         milestone_bonus=10.0, milestone_every=0),
    # 4: stepping-stone milestones every 10 rounds — a reward trail through freeplay
    dict(name="milestone10", init_from=RUN13,         gamma=0.999,  lr=1e-4,
         milestone_bonus=5.0,  milestone_every=10),
    # 5: chained base — continue the run that already reached ~56
    dict(name="chain",       init_from=FREEPLAY_WARM, gamma=0.999,  lr=1e-4),
]


def config(i: int) -> dict:
    return CONFIGS[i]


def args_for(i: int) -> str:
    c = CONFIGS[i]
    parts = [
        f"--gamma {c['gamma']}",
        f"--learning-rate {c['lr']}",
        f"--save-path agent/models/fp_{c['name']}/model",
    ]
    if c.get("init_from"):
        parts.append(f"--init-from {c['init_from']}")
    if c.get("milestone_bonus"):
        parts.append(f"--milestone-bonus {c['milestone_bonus']}")
        parts.append(f"--milestone-every {c.get('milestone_every', 0)}")
    return " ".join(parts)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(CONFIGS))
    else:
        print(args_for(int(sys.argv[1])))
