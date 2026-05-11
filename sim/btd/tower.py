"""Tower entity. Mirrors Tower.as."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import TOWER_STATS


@dataclass
class Tower:
    id: int
    type: str
    x: float
    y: float
    attack_rate: int
    attack_radius: float
    shoot_power: float
    pierce_max: int
    spent_on_me: int
    pop_count: int = 0
    time_since_last_shot: int = 0
    # Targeting mode. AS only ever uses "first"; "last" is supported for
    # completeness but no shipped tower switches to it.
    ai_mode: str = "first"

    @classmethod
    def from_type(cls, tower_id: int, type_: str, x: float, y: float) -> "Tower":
        stats = TOWER_STATS[type_]
        return cls(
            id=tower_id,
            type=type_,
            x=x,
            y=y,
            attack_rate=stats["attackRate"],
            attack_radius=stats["attackRadius"],
            shoot_power=stats["shootPower"],
            pierce_max=stats["pierceMax"],
            spent_on_me=stats["cost"],
        )
