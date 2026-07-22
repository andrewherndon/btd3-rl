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


def make_env(difficulty: str, curriculum_p: float = 0.0, diversity_bonus: float = 0.0,
             difficulty_choices: tuple[str, ...] = (), freeplay: bool = False):
    """Factory for one Monitor-wrapped env. Each reset draws a fresh sim seed;
    Monitor records episode reward/length for logging. Training aids (all off for
    eval): `curriculum_p` starts some episodes mid-game, `diversity_bonus` rewards
    new tower types, `difficulty_choices` randomizes the difficulty per episode
    (domain randomization). `freeplay` lets episodes run past round 50 (procedural
    51-149) instead of winning at 50."""
    def _init():
        return Monitor(BloonsEnv(SimConfig(difficulty=difficulty, freeplay=freeplay),
                                 curriculum_p=curriculum_p, diversity_bonus=diversity_bonus,
                                 difficulty_choices=difficulty_choices))
    return _init


def build_model(vec_env, seed: int, gamma: float = 0.999, ent_coef: float = 0.01,
                learning_rate: float = 3e-4) -> MaskablePPO:
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
        learning_rate=learning_rate,  # Adam step size
        # --- return / advantage estimation ---
        gamma=gamma,         # discount over DECISION steps. Raised from 0.995: the
                             # agent plateaus at ~25 towers then hoards because the
                             # payoff of late-game towers (surviving rounds 42-50) is
                             # too distal to propagate. Now safe to raise since the
                             # anti-churn fix shortened episodes (~130 vs ~660 steps).
        gae_lambda=0.95,     # GAE bias/variance knob for the advantage (critic baseline)
        # --- PPO stability / exploration ---
        clip_range=0.2,      # the "proximal" trust region on the policy update
        ent_coef=ent_coef,   # entropy bonus (back to default: dart+bomb IS the
                             # winning strategy, so exploration was never the
                             # problem — under-building/hoarding is)
        vf_coef=0.5,         # weight of the value (critic) loss
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    # Training difficulty pool (comma-separated). >1 value = domain randomization.
    p.add_argument("--difficulties", default="easy,medium,hard")
    # Difficulty the eval callback (best_model selection) uses. hard = toughest test.
    p.add_argument("--eval-difficulty", default="hard", choices=["easy", "medium", "hard"])
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
    # Sweepable hyperparameters (exposed for the SLURM sweep; see agent/sweep.sbatch).
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    # Play past round 50 (procedural 51-149) instead of winning at 50.
    p.add_argument("--freeplay", action="store_true")
    args = p.parse_args()

    # Multiple envs still help PPO (decorrelated batch) even at equal throughput.
    VecEnv = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    diffs = tuple(d.strip() for d in args.difficulties.split(","))
    train_env = VecEnv([make_env(diffs[0], args.curriculum_p, args.diversity_bonus,
                                 diffs, args.freeplay)
                        for _ in range(args.n_envs)])
    # Separate eval env: fixed difficulty, no scaffolds/randomization, so best_model
    # is selected on honest full round-1 games at one difficulty.
    eval_env = DummyVecEnv([make_env(args.eval_difficulty, 0.0, 0.0, (), args.freeplay)])

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(save_path.parent),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = build_model(train_env, args.seed, gamma=args.gamma,
                        ent_coef=args.ent_coef, learning_rate=args.learning_rate)
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
    model.save(str(save_path))
    print(f"saved model -> {save_path}.zip")


if __name__ == "__main__":
    main()
