"""Observation encoder: BloonsSim state -> normalized Dict observation.
See RL_DESIGN.md "Observation" for the rationale behind every choice here.

Observed only between rounds, so there are no live bloons to encode; the
variable-length entity set is the tower list.
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from btd.constants import LIVES_BY_DIFF, STAGE_H, STAGE_W
from btd.game import BloonsSim

from . import actions as A

# Normalization scales.
MONEY_SCALE = 5000.0          # log(1+money) / log(1+MONEY_SCALE) -> ~[0,1]
THREAT_SCALE = 80.0           # bloons-per-rank in a round rarely exceeds this
WIN_ROUND = 50.0              # rounds are normalized against the win condition
N_RANKS = 10                  # bloon ranks 1..10

# Tower feature layout: one-hot type (N_TYPES) + x + y + p1_level + p2_level.
TOWER_FEATS = A.N_TYPES + 4


def make_observation_space() -> spaces.Dict:
    """The Gymnasium observation space matching `encode()`'s output."""
    return spaces.Dict(
        {
            "money": spaces.Box(0.0, 1.0, (1,), np.float32),
            "lives": spaces.Box(0.0, 1.0, (1,), np.float32),
            "round": spaces.Box(0.0, 1.0, (1,), np.float32),
            "cost_mult": spaces.Box(0.5, 1.5, (1,), np.float32),
            "next_round_counts": spaces.Box(0.0, 1.0, (N_RANKS,), np.float32),
            "towers": spaces.Box(0.0, 1.0, (A.MAX_TOWERS, TOWER_FEATS), np.float32),
            "tower_mask": spaces.Box(0.0, 1.0, (A.MAX_TOWERS,), np.float32),
        }
    )


def _next_round_counts(sim: BloonsSim) -> np.ndarray:
    """Per-rank counts of the round START_ROUND would play next, normalized."""
    counts = np.zeros(N_RANKS, dtype=np.float32)
    for rank in sim.round_table.get(sim.round + 1, []):
        counts[rank - 1] += 1.0
    return np.clip(counts / THREAT_SCALE, 0.0, 1.0)


def encode(sim: BloonsSim) -> dict[str, np.ndarray]:
    """Snapshot the sim as a normalized observation dict."""
    start_lives = LIVES_BY_DIFF[sim.config.difficulty]

    towers = np.zeros((A.MAX_TOWERS, TOWER_FEATS), dtype=np.float32)
    tower_mask = np.zeros(A.MAX_TOWERS, dtype=np.float32)
    for i, t in enumerate(sim.towers[: A.MAX_TOWERS]):
        row = towers[i]
        row[A.TOWER_TYPES.index(t.type)] = 1.0                    # one-hot type
        row[A.N_TYPES + 0] = t.x / STAGE_W                        # x in [0,1]
        row[A.N_TYPES + 1] = t.y / STAGE_H                        # y in [0,1]
        row[A.N_TYPES + 2] = (int(t.upgrade1) + int(t.upgrade2)) / 2.0  # path1 lvl
        row[A.N_TYPES + 3] = (int(t.upgrade3) + int(t.upgrade4)) / 2.0  # path2 lvl
        tower_mask[i] = 1.0

    return {
        # log-compressed: unbounded and multiplicatively meaningful.
        "money": np.array(
            [math.log1p(sim.money) / math.log1p(MONEY_SCALE)], dtype=np.float32
        ),
        "lives": np.array([sim.lives / start_lives], dtype=np.float32),
        "round": np.array([min(sim.round / WIN_ROUND, 1.0)], dtype=np.float32),
        "cost_mult": np.array([sim.cost_mult], dtype=np.float32),
        "next_round_counts": _next_round_counts(sim),
        "towers": towers,
        "tower_mask": tower_mask,
    }
