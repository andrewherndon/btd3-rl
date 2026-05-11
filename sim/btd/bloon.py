"""Bloon entity. Mirrors Bloon.as."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bloon:
    rank: int
    frame: float                    # advances by `speed` each tick
    maxspeed: float
    speed: float
    jitter_x: float
    jitter_y: float
    branch: int = 1                 # branch index for branched tracks (track 3 = 1)
    hits_remaining: int = 1         # only meaningful for MOAB/BFB
    radius: float = 12.0
    # Stage position cached each tick from path[round(frame)] + jitter.
    x: float = 0.0
    y: float = 0.0
    # Lifecycle.
    popped: bool = False
    escaped: bool = False
    hit_this_frame: bool = False    # bloon can absorb at most one bullet per tick (Bloon.Update returns after first hit)
    # Freeze state. `freeze_duration` is set per-freeze (min of the freezer's
    # freeze_len and the 100-frame AS hard cap).
    frozen: bool = False
    time_frozen: int = 0
    freeze_duration: int = 0

    @property
    def alive(self) -> bool:
        return not (self.popped or self.escaped)
