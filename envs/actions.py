"""Action codec: the flat Discrete(N) action space and its translation to/from
semantic actions. See RL_DESIGN.md "Action" for the rationale.

Layout of the flat index space:

    0                       START_ROUND
    1     .. PLACE_END      PLACE(type, cell)   = 1 + type*N_CELLS + cell
    ...   .. UPGRADE_END    UPGRADE(slot, path) = UPGRADE_OFF + slot*2 + (path-1)
    ...   .. SELL_END       SELL(slot)          = SELL_OFF + slot

`slot` indexes the observation's tower table (row i <-> the i-th live tower),
so the agent acts on a tower by the same index it observes it at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# Ordered tower types. This order is the contract: it fixes the one-hot layout
# in the observation AND the `type` field of PLACE actions. Do not reorder
# without retraining. Mirrors the dict order in btd.constants.TOWER_STATS.
TOWER_TYPES: tuple[str, ...] = (
    "dart", "tack", "ice", "bomb", "spikeopult", "super", "boomerang", "beacon",
)
N_TYPES = len(TOWER_TYPES)

# Placement grid over the 640x480 stage.
CELL_SIZE = 16
GRID_COLS = 640 // CELL_SIZE          # 40
GRID_ROWS = 480 // CELL_SIZE          # 30
N_CELLS = GRID_COLS * GRID_ROWS       # 1200

# Cap on towers the env represents (obs rows and upgrade/sell slots). Raised
# 64 -> 256 (2026-07-27): the real game is uncapped and easy freeplay needs
# 100-200+ towers to push past round ~58 — a 64-tower defense's DPS plateaus
# exactly there, which was the true cause of the "round-58 wall" (see
# history/README.md). 256 covers near-real-game scale without imposing a
# tower-count strategy; a placement guard in mask.py enforces it as a safety net.
# NOTE: changing this changes obs/action shapes, so models are NOT cross-loadable
# across values (a fresh training run is required).
MAX_TOWERS = 256

# --- flat index block boundaries ------------------------------------------
START_ROUND = 0
PLACE_OFF = 1                                   # 1
PLACE_END = PLACE_OFF + N_TYPES * N_CELLS       # 9601 (exclusive)
UPGRADE_OFF = PLACE_END                         # 9601
UPGRADE_END = UPGRADE_OFF + MAX_TOWERS * 2      # 10113 (exclusive)
SELL_OFF = UPGRADE_END                          # 10113
SELL_END = SELL_OFF + MAX_TOWERS                # 10369 (exclusive)
N_ACTIONS = SELL_END                            # 10369


class Kind(IntEnum):
    START_ROUND = 0
    PLACE = 1
    UPGRADE = 2
    SELL = 3


@dataclass(frozen=True)
class Action:
    kind: Kind
    # PLACE: (type_idx, cell). UPGRADE: (slot, path). SELL: (slot, _).
    a: int = 0
    b: int = 0

    @property
    def tower_type(self) -> str:
        return TOWER_TYPES[self.a]


def cell_to_xy(cell: int) -> tuple[float, float]:
    """Cell index -> stage-pixel center of that grid cell."""
    col, row = cell % GRID_COLS, cell // GRID_COLS
    return (col + 0.5) * CELL_SIZE, (row + 0.5) * CELL_SIZE


def decode(index: int) -> Action:
    """Flat action index -> semantic Action."""
    if index == START_ROUND:
        return Action(Kind.START_ROUND)
    if index < PLACE_END:
        i = index - PLACE_OFF
        return Action(Kind.PLACE, a=i // N_CELLS, b=i % N_CELLS)
    if index < UPGRADE_END:
        i = index - UPGRADE_OFF
        return Action(Kind.UPGRADE, a=i // 2, b=(i % 2) + 1)   # path in {1,2}
    i = index - SELL_OFF
    return Action(Kind.SELL, a=i)


# --- encoders (inverse of decode; used by tests and scripted policies) -----

def encode_place(type_idx: int, cell: int) -> int:
    return PLACE_OFF + type_idx * N_CELLS + cell


def encode_upgrade(slot: int, path: int) -> int:
    return UPGRADE_OFF + slot * 2 + (path - 1)


def encode_sell(slot: int) -> int:
    return SELL_OFF + slot
