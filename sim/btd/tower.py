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
    icebreak: bool = False     # bullets can pop frozen bloons
    leadbreak: bool = False    # bullets can pop lead bloons
    is_spread: bool = False    # fires N shards in a fan (tack, ice); no aimed bullet
    is_attacker: bool = True   # False for beacons — skip fire logic, still tick counter
    freeze_len: int = 0        # frames a frozen bloon stays frozen (ice only)
    bullet_scale: float = 1.0  # multiplies bullet.radius and explosion_radius at fire time
    # Upgrade flags. upgrade1 + upgrade3 form path 1; upgrade2 + upgrade4 form
    # path 2. Path-locking lives in BloonsSim.upgrade_tower.
    upgrade1: bool = False
    upgrade2: bool = False
    upgrade3: bool = False
    upgrade4: bool = False
    # Visual / behavioural flags flipped by upgrades.
    transformed: bool = False  # blade tack, glaive boomerang, missile bomb, multishot spikeopult
    laser: bool = False        # super monkey laser (upgrade3) / plasma (upgrade4)
    # Set by the beacon-buff refresh each frame. Read by _acquire_target /
    # _tick_towers when deciding effective range / rate.
    beacon_radius_active: bool = False
    beacon_rate_active: bool = False
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
            icebreak=stats.get("icebreak", False),
            leadbreak=stats.get("leadbreak", False),
            is_spread=stats.get("is_spread", False),
            is_attacker=stats.get("is_attacker", True),
            freeze_len=stats.get("freeze_len", 0),
        )
