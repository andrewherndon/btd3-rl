"""Bullet entity. Mirrors Bullet.as."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import BULLET_STATS


@dataclass
class Bullet:
    type: str
    x: float
    y: float
    vx: float
    vy: float
    pierce_max: int
    radius: float
    lifespan: int
    shooter_id: int
    pierce_count: int = 0
    time_alive: int = 0
    is_dead: bool = False

    @classmethod
    def from_type(
        cls,
        type_: str,
        x: float,
        y: float,
        vx: float,
        vy: float,
        pierce_max: int,
        shooter_id: int,
    ) -> "Bullet":
        stats = BULLET_STATS[type_]
        return cls(
            type=type_,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            pierce_max=pierce_max,
            radius=stats["radius"],
            lifespan=stats["lifespan"],
            shooter_id=shooter_id,
        )
