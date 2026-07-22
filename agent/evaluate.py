"""Evaluate a trained MaskablePPO agent over many games and report metrics.

    python agent/evaluate.py --model agent/models/best_model --episodes 50

Runs the deterministic policy on a fixed set of held-out seeds (distinct from
the random seeds training draws) and aggregates win-rate, furthest round
reached, lives remaining, and reward. This is the quantitative counterpart to
watch.py's single-game view.
"""

from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO

from btd.game import SimConfig
from envs import BloonsEnv


def run_episodes(model, env, seeds, deterministic) -> dict:
    rounds, lives, rewards, money, wins = [], [], [], [], 0
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done, total = False, 0.0
        info = {}
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            done = term or trunc
        rounds.append(info["round"])
        lives.append(info["lives"])
        rewards.append(total)
        money.append(info["money"])           # leftover money at game end
        wins += int(info["won"])
    return {
        "rounds": np.array(rounds),
        "lives": np.array(lives),
        "rewards": np.array(rewards),
        "money": np.array(money),
        "wins": wins,
        "n": len(seeds),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="agent/models/best_model")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    # Held-out seed base: far from anything training is likely to have drawn.
    p.add_argument("--seed-base", type=int, default=1_000_000)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--freeplay", action="store_true", help="play past round 50")
    args = p.parse_args()

    model = MaskablePPO.load(args.model)
    env = BloonsEnv(SimConfig(difficulty=args.difficulty, freeplay=args.freeplay))
    seeds = np.arange(args.seed_base, args.seed_base + args.episodes)

    res = run_episodes(model, env, seeds, deterministic=not args.stochastic)
    rounds = res["rounds"]

    print(f"model: {args.model}   difficulty: {args.difficulty}   episodes: {res['n']}")
    print("-" * 52)
    print(f"win rate         : {res['wins']}/{res['n']}  ({100*res['wins']/res['n']:.0f}%)")
    print(f"round reached    : mean {rounds.mean():.1f}  median {np.median(rounds):.0f}"
          f"  min {rounds.min()}  max {rounds.max()}")
    print(f"lives remaining  : mean {res['lives'].mean():.1f}")
    print(f"money at end     : mean ${res['money'].mean():.0f}  "
          f"(high => hoarding / under-building)")
    print(f"reward           : mean {res['rewards'].mean():.2f}"
          f"  (min {res['rewards'].min():.1f}  max {res['rewards'].max():.1f})")
    # Round-reached distribution, quartiles — quick sense of consistency.
    q = np.percentile(rounds, [25, 50, 75])
    print(f"round quartiles  : 25% {q[0]:.0f}   50% {q[1]:.0f}   75% {q[2]:.0f}")


if __name__ == "__main__":
    main()
