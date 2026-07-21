"""Replay a trained model headless and print its action log + a summary.

    python agent/inspect.py --model agent/models/run6/model --seed 42

Same decisions as watch.py, but text-only: shows exactly what the agent does
between rounds (the top-right event log from watch.py), plus counts that
quantify churn (sells vs places) and tower diversity.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from sb3_contrib import MaskablePPO

from btd.game import BloonsSim, SimConfig
from envs import actions as A
from envs.actions import Kind, cell_to_xy, decode
from envs.bloons_env import MAX_ECON_PER_ROUND
from envs.mask import build_action_mask, compute_cell_validity
from envs.observation import encode


def play(model, difficulty, seed, deterministic, log):
    sim = BloonsSim(SimConfig(difficulty=difficulty, seed=seed))
    cell_valid = compute_cell_validity(sim)
    econ_streak = 0
    counts = Counter()
    placed = Counter()
    sells = 0
    guard = 0
    while not sim.game_over and guard < 30000:
        guard += 1
        obs = encode(sim)
        if econ_streak >= MAX_ECON_PER_ROUND:
            mask = np.zeros(A.N_ACTIONS, dtype=bool)
            mask[A.START_ROUND] = True
        else:
            mask = build_action_mask(sim, cell_valid)
        act = decode(int(model.predict(obs, action_masks=mask, deterministic=deterministic)[0]))
        counts[act.kind.name] += 1
        if act.kind == Kind.START_ROUND:
            econ_streak = 0
            sim.start_round()
            log.append(f"> round {sim.round}  ({len(sim.towers)} towers, ${sim.money})")
            g = 0
            while sim.in_round and not sim.game_over and g < 10000:
                sim.step()
                g += 1
        else:
            econ_streak += 1
            if act.kind == Kind.PLACE:
                x, y = cell_to_xy(act.b)
                if sim.place_tower(act.tower_type, x, y) != -1:
                    placed[act.tower_type] += 1
                    log.append(f"    + {act.tower_type} @({x:.0f},{y:.0f})")
            elif act.kind == Kind.UPGRADE and act.a < len(sim.towers):
                t = sim.towers[act.a]
                entry = sim.available_upgrades(t.id)[act.b]
                if sim.upgrade_path(t.id, act.b) and entry:
                    log.append(f"    up #{t.id} -> {entry[0]}")
            elif act.kind == Kind.SELL and act.a < len(sim.towers):
                t = sim.towers[act.a]
                if sim.sell_tower(t.id):
                    sells += 1
                    log.append(f"    - sell {t.type} #{t.id}")
    return dict(round=sim.round, money=sim.money, won=sim.won,
                towers=len(sim.towers), counts=counts, placed=placed, sells=sells)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="agent/models/run6/model")
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-log", action="store_true", help="summary only")
    args = p.parse_args()

    model = MaskablePPO.load(args.model)
    log: list[str] = []
    r = play(model, args.difficulty, args.seed, not args.stochastic, log)

    if not args.no_log:
        print("\n".join(log))
        print("-" * 50)
    c = r["counts"]
    print(f"result: round {r['round']}, ${r['money']} left, {r['towers']} towers, won={r['won']}")
    print(f"actions: START={c['START_ROUND']}  PLACE={c['PLACE']}  "
          f"UPGRADE={c['UPGRADE']}  SELL={c['SELL']}")
    print(f"sells={r['sells']} vs places={sum(r['placed'].values())}  "
          f"(sells >> net towers => churn)")
    print(f"tower types placed: {dict(r['placed'])}")


if __name__ == "__main__":
    main()
