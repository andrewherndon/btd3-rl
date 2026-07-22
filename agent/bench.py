"""Quick per-node performance + sanity check before committing to real runs.
Prints node info, verifies the env/model build end-to-end, and measures training
throughput (steps/sec).

    python agent/bench.py --timesteps 30000
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import time
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=30000)
    p.add_argument("--n-envs", type=int, default=4)
    args = p.parse_args()

    host = socket.gethostname()
    print(f"[{host}] python {platform.python_version()}  cpus={os.cpu_count()}", flush=True)

    import numpy as np
    import torch
    torch.set_num_threads(1)                       # single-threaded (fair fps + packing)
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from btd.game import SimConfig
    from envs import BloonsEnv

    print(f"[{host}] torch {torch.__version__}  numpy {np.__version__}  "
          f"threads={torch.get_num_threads()}", flush=True)

    # --- sanity: env builds, obs in space, mask non-empty, one step runs ---
    env = BloonsEnv(SimConfig(difficulty="easy"))
    obs, _ = env.reset(seed=0)
    mask = env.action_masks()
    assert env.observation_space.contains(obs), "obs not in space"
    assert mask.any(), "empty action mask"
    env.step(int(np.flatnonzero(mask)[0]))
    print(f"[{host}] env OK (obs + mask + step)", flush=True)

    # --- benchmark: time a short training (domain-randomized, like the real run) ---
    venv = DummyVecEnv([
        lambda: Monitor(BloonsEnv(SimConfig(difficulty="easy"),
                                  difficulty_choices=("easy", "medium", "hard"),
                                  curriculum_p=0.5))
        for _ in range(args.n_envs)
    ])
    model = MaskablePPO("MultiInputPolicy", venv, n_steps=2048, batch_size=256,
                        verbose=0, seed=0)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    dt = time.time() - t0
    fps = args.timesteps / dt
    print(f"[{host}] RESULT: {fps:.0f} steps/s  "
          f"({args.timesteps} steps in {dt:.1f}s)  PASS", flush=True)


if __name__ == "__main__":
    main()
