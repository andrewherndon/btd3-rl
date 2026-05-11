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
    icebreak: bool = False
    leadbreak: bool = False
    freeze_len: int = 0       # forwarded from ice tower; used by _try_freeze
    # Two-stage bullets (bomb): start with `radius`, detonate on first hit
    # (sets vx=vy=0, swaps to `explosion_radius`), continue colliding until
    # pierce_max or lifespan expires. 0 = no second stage.
    explosion_radius: float = 0.0
    hashit: bool = False
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
        icebreak: bool = False,
        leadbreak: bool = False,
        freeze_len: int = 0,
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
            icebreak=icebreak,
            leadbreak=leadbreak,
            freeze_len=freeze_len,
            explosion_radius=stats.get("explosion_radius", 0.0),
        )
