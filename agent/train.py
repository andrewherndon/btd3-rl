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


def make_env(difficulty: str, curriculum_p: float = 0.0, diversity_bonus: float = 0.0):
    """Factory for one Monitor-wrapped env. Each reset draws a fresh sim seed
    (domain randomization); Monitor records episode reward/length for logging.
    `curriculum_p` > 0 starts some episodes mid-game at hard rounds and
    `diversity_bonus` > 0 rewards trying new tower types (both training only;
    eval passes 0.0 so metrics stay on honest full round-1 games)."""
    def _init():
        return Monitor(BloonsEnv(SimConfig(difficulty=difficulty),
                                 curriculum_p=curriculum_p, diversity_bonus=diversity_bonus))
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
        gamma=0.999,         # discount over DECISION steps. Raised from 0.995: the
                             # agent plateaus at ~25 towers then hoards because the
                             # payoff of late-game towers (surviving rounds 42-50) is
                             # too distal to propagate. Now safe to raise since the
                             # anti-churn fix shortened episodes (~130 vs ~660 steps).
        gae_lambda=0.95,     # GAE bias/variance knob for the advantage (critic baseline)
        # --- PPO stability / exploration ---
        clip_range=0.2,      # the "proximal" trust region on the policy update
        ent_coef=0.01,       # entropy bonus (back to default: dart+bomb IS the
                             # winning strategy, so exploration was never the
                             # problem — under-building/hoarding is)
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
    # One-time reward for first placing each tower type. Default OFF: dart+bomb
    # already wins the game, so diversity was a non-problem; kept as a knob only.
    p.add_argument("--diversity-bonus", type=float, default=0.0)
    args = p.parse_args()

    # Multiple envs still help PPO (decorrelated batch) even at equal throughput.
    VecEnv = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    train_env = VecEnv([make_env(args.difficulty, args.curriculum_p, args.diversity_bonus)
                        for _ in range(args.n_envs)])
    # Separate eval env (single game, full round-1 start, no scaffolds) for
    # honest metrics.
    eval_env = DummyVecEnv([make_env(args.difficulty, 0.0, 0.0)])

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
