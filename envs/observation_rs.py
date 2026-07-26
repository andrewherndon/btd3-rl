"""Observation encoder — adapted for btd_rs Rust sim."""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from btd.constants import LIVES_BY_DIFF, STAGE_H, STAGE_W
from . import actions as A

MONEY_SCALE = 5000.0
THREAT_SCALE = 80.0
WIN_ROUND = 50.0
N_RANKS = 10
TOWER_FEATS = A.N_TYPES + 4
TOWER_TYPE_NAMES = A.TOWER_TYPES


def make_observation_space() -> spaces.Dict:
    return spaces.Dict({
        "money": spaces.Box(0.0, 1.0, (1,), np.float32),
        "lives": spaces.Box(0.0, 1.0, (1,), np.float32),
        "round": spaces.Box(0.0, 1.0, (1,), np.float32),
        "cost_mult": spaces.Box(0.5, 1.5, (1,), np.float32),
        "next_round_counts": spaces.Box(0.0, 1.0, (N_RANKS,), np.float32),
        "towers": spaces.Box(0.0, 1.0, (A.MAX_TOWERS, TOWER_FEATS), np.float32),
        "tower_mask": spaces.Box(0.0, 1.0, (A.MAX_TOWERS,), np.float32),
    })


def encode(sim) -> dict[str, np.ndarray]:
    """Encode the Rust sim state as a normalized observation dict."""
    start_lives = LIVES_BY_DIFF[sim.difficulty]

    towers = np.zeros((A.MAX_TOWERS, TOWER_FEATS), dtype=np.float32)
    tower_mask = np.zeros(A.MAX_TOWERS, dtype=np.float32)
    for i, t in enumerate(sim.get_towers()[:A.MAX_TOWERS]):
        row = towers[i]
        row[TOWER_TYPE_NAMES.index(t["type"])] = 1.0
        row[A.N_TYPES + 0] = t["x"] / STAGE_W
        row[A.N_TYPES + 1] = t["y"] / STAGE_H
        row[A.N_TYPES + 2] = (int(t["upgrade1"]) + int(t["upgrade2"])) / 2.0
        row[A.N_TYPES + 3] = (int(t["upgrade3"]) + int(t["upgrade4"])) / 2.0
        tower_mask[i] = 1.0

    # Next round counts via Rust sim's API.
    next_ranks = sim.get_next_round_bloons()
    counts = np.zeros(N_RANKS, dtype=np.float32)
    for rank in next_ranks:
        counts[rank - 1] += 1.0
    next_round_counts = np.clip(counts / THREAT_SCALE, 0.0, 1.0)

    return {
        "money": np.array([math.log1p(sim.money) / math.log1p(MONEY_SCALE)], dtype=np.float32),
        "lives": np.array([sim.lives / start_lives], dtype=np.float32),
        "round": np.array([
            min(sim.round / (sim.max_round if sim.freeplay else WIN_ROUND), 1.0)
        ], dtype=np.float32),
        "cost_mult": np.array([sim.cost_mult], dtype=np.float32),
        "next_round_counts": next_round_counts,
        "towers": towers,
        "tower_mask": tower_mask,
    }
