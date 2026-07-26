"""Action legality mask — adapted for btd_rs Rust sim."""

from __future__ import annotations

import numpy as np

from btd.constants import TOWER_STATS
from . import actions as A


def compute_cell_validity(sim) -> np.ndarray:
    """Static per-cell placement legality, shape (N_CELLS,), dtype bool."""
    valid = np.zeros(A.N_CELLS, dtype=bool)
    for cell in range(A.N_CELLS):
        x, y = A.cell_to_xy(cell)
        valid[cell] = sim.is_placement_valid(x, y)
    return valid


def build_action_mask(sim, cell_valid: np.ndarray) -> np.ndarray:
    """Legality of every flat action (bool[N_ACTIONS])."""
    mask = np.zeros(A.N_ACTIONS, dtype=bool)

    # START_ROUND
    mask[A.START_ROUND] = (not sim.game_over) and (not sim.in_round)

    # Cache tower list — get_towers() creates a new list of Python dicts
    # from Rust each call, so we call it ONCE.
    towers = sim.get_towers()
    n_towers = len(towers)

    # PLACE
    if n_towers < A.MAX_TOWERS:
        place = mask[A.PLACE_OFF:A.PLACE_END].reshape(A.N_TYPES, A.N_CELLS)
        for t, type_name in enumerate(A.TOWER_TYPES):
            price = sim.price(TOWER_STATS[type_name]["cost"])
            if price <= sim.money:
                place[t] = cell_valid
        # No same-cell stacking.
        for tw in towers:
            col = int(tw["x"] // A.CELL_SIZE)
            row = int(tw["y"] // A.CELL_SIZE)
            if 0 <= col < A.GRID_COLS and 0 <= row < A.GRID_ROWS:
                place[:, row * A.GRID_COLS + col] = False

    # UPGRADE
    for slot in range(min(n_towers, A.MAX_TOWERS)):
        tower = towers[slot]
        avail = sim.available_upgrades(tower["id"])
        for path in (1, 2):
            entry = avail.get(path)
            if entry is not None and entry[1] <= sim.money:
                mask[A.encode_upgrade(slot, path)] = True

    # SELL (anti-churn: must have survived at least one round)
    n = min(n_towers, A.MAX_TOWERS)
    for slot in range(n):
        if towers[slot]["placed_round"] < sim.round:
            mask[A.encode_sell(slot)] = True

    return mask
