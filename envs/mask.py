"""Action legality mask. See RL_DESIGN.md "Action": we mask illegal actions
(prob -> 0) rather than penalizing them. Legality is known exactly, so we
compute it directly instead of making the agent learn it.
"""

from __future__ import annotations

import numpy as np

from btd.constants import TOWER_STATS
from btd.game import BloonsSim

from . import actions as A


def compute_cell_validity(sim: BloonsSim) -> np.ndarray:
    """Static per-cell placement legality, shape (N_CELLS,), dtype bool.

    The path and grid never move, so this is computed once per episode (at
    reset) and reused every step — avoids 1200 point-to-path distances per
    decision.
    """
    valid = np.zeros(A.N_CELLS, dtype=bool)
    for cell in range(A.N_CELLS):
        x, y = A.cell_to_xy(cell)
        valid[cell] = sim.is_placement_valid(x, y)
    return valid


def build_action_mask(sim: BloonsSim, cell_valid: np.ndarray) -> np.ndarray:
    """Legality of every flat action given the sim's current state.

    Returns bool[N_ACTIONS]; True == legal == may be sampled.
    """
    mask = np.zeros(A.N_ACTIONS, dtype=bool)

    # START_ROUND: only at a decision point (between rounds, game live).
    mask[A.START_ROUND] = (not sim.game_over) and (not sim.in_round)

    # PLACE(type, cell): affordable type AND valid cell. The PLACE block is a
    # flat (N_TYPES x N_CELLS) grid; fill row-by-type with cell_valid gated on
    # affordability. This is where per-(type,cell) coupling is expressed.
    #
    # Safety net: forbid placing beyond MAX_TOWERS so the agent can never create
    # a tower that the observation can't represent or upgrade/sell can't address
    # (that blind spot broke the Markov property in early runs). MAX_TOWERS is
    # set high enough that this rarely binds and does not dictate strategy.
    if len(sim.towers) < A.MAX_TOWERS:
        place = mask[A.PLACE_OFF:A.PLACE_END].reshape(A.N_TYPES, A.N_CELLS)
        for t, type_name in enumerate(A.TOWER_TYPES):
            price = sim._price(TOWER_STATS[type_name]["cost"])
            if price <= sim.money:
                place[t] = cell_valid
        # No exact-overlap stacking: forbid placing on a cell already occupied by
        # a tower (matches the real game's rule that towers can't share a spot).
        for tw in sim.towers:
            col = int(tw.x // A.CELL_SIZE)
            row = int(tw.y // A.CELL_SIZE)
            if 0 <= col < A.GRID_COLS and 0 <= row < A.GRID_ROWS:
                place[:, row * A.GRID_COLS + col] = False

    # UPGRADE(slot, path): slot holds a tower, path has a next upgrade, afford.
    for slot, tower in enumerate(sim.towers[:A.MAX_TOWERS]):
        avail = sim.available_upgrades(tower.id)      # {1: (name, price)|None, 2: ...}
        for path in (1, 2):
            entry = avail[path]
            if entry is not None and entry[1] <= sim.money:
                mask[A.encode_upgrade(slot, path)] = True

    # SELL(slot): a tower can be sold once it has fought at least one round
    # (placed_round < current round). Forbidding sale of a tower placed THIS
    # shopping phase breaks the buy->sell->rebuy churn loop, with no reward change.
    n = min(len(sim.towers), A.MAX_TOWERS)
    for slot in range(n):
        if sim.towers[slot].placed_round < sim.round:
            mask[A.encode_sell(slot)] = True

    return mask
