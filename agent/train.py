"""Train a MaskablePPO agent on BloonsEnv.

    python agent/train.py --timesteps 200000 --difficulty easy --n-envs 8

User-owned RL code (see OBJECTIVE.md). The sim/env contract it trains against
is described in RL_DESIGN.md. Algorithm choice (MaskablePPO) and the reasoning
behind the obs/action/reward design live there too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from btd.game import SimConfig
from envs import BloonsEnv


def make_env(difficulty: str, curriculum_p: float = 0.0):
    """Factory for one Monitor-wrapped env. Each reset draws a fresh sim seed
    (domain randomization); Monitor records episode reward/length for logging.
    `curriculum_p` > 0 starts some episodes mid-game at hard rounds (training
    only; eval passes 0.0 to keep metrics on full round-1 games)."""
    def _init():
        return Monitor(BloonsEnv(SimConfig(difficulty=difficulty), curriculum_p=curriculum_p))
    return _init


def build_model(vec_env, seed: int) -> MaskablePPO:
    return MaskablePPO(
        # Dict observation -> MultiInputPolicy (per-key encoders, then concat).
        "MultiInputPolicy",
        vec_env,
        seed=seed,
        verbose=1,
        # --- rollout / optimization ---
        n_steps=2048,        # env steps per env before an update; batch = n_steps*n_envs
        batch_size=256,      # minibatch size for the SGD epochs
        n_epochs=10,         # passes over each rollout (PPO reuses data via clipping)
        learning_rate=3e-4,  # Adam step size
        # --- return / advantage estimation ---
        gamma=0.995,         # discount over DECISION steps. High because the horizon
                             # is long (many economic steps/round); prime tuning knob.
        gae_lambda=0.95,     # GAE bias/variance knob for the advantage (critic baseline)
        # --- PPO stability / exploration ---
        clip_range=0.2,      # the "proximal" trust region on the policy update
        ent_coef=0.01,       # entropy bonus -> keeps the masked policy exploring
        vf_coef=0.5,         # weight of the value (critic) loss
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", default="agent/models/maskable_ppo")
    p.add_argument("--eval-freq", type=int, default=10_000)
    # SubprocVecEnv (multi-process) barely beats Dummy here: the main process
    # serializes N Dict-obs + N action-masks every step, so IPC (Amdahl) caps
    # the gain at ~1.15x on this env. Default to the simpler single-process env.
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    # Fraction of training episodes that start mid-game at a hard round (dense
    # MOAB-era experience). Eval always uses full round-1 games (0.0).
    p.add_argument("--curriculum-p", type=float, default=0.5)
    args = p.parse_args()

    # Multiple envs still help PPO (decorrelated batch) even at equal throughput.
    VecEnv = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    train_env = VecEnv([make_env(args.difficulty, args.curriculum_p) for _ in range(args.n_envs)])
    # Separate eval env (single game, full round-1 start) for honest metrics.
    eval_env = DummyVecEnv([make_env(args.difficulty, 0.0)])

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(save_path.parent),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = build_model(train_env, args.seed)
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
    model.save(str(save_path))
    print(f"saved model -> {save_path}.zip")


if __name__ == "__main__":
    main()
