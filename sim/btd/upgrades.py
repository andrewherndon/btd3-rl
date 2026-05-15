"""Tower upgrade definitions. Ports BloonsTD.GetUpgrade() from BloonsTD.as.

Each tower has up to 4 upgrades. Two paths of two upgrades each:
  path 1 = upgrade1 (level 1) then upgrade3 (level 2)
  path 2 = upgrade2 (level 1) then upgrade4 (level 2)

Path-locking (from AS clickUpgradeBtn):
  - The upgrade1 button is blocked once upgrade2 is bought, which means you
    can never buy upgrade1 or upgrade3 after committing to path 2. Path 2 has
    no equivalent block in the other direction, so buying path 1 first and
    then progressing to path 2 is allowed.

Out of scope (omitted from the table; depend on monkey storm consumable which
the first RL iteration excludes):
  - beacon3 (storm unlock)
  - beacon4 (activate monkey storm)

Some upgrades have behavioural side-effects beyond stat changes; those are
documented on the spec and implemented inline in game.py:
  - bomb upgrade2: bombs spawn N frag bullets on detonation
  - ice  upgrade2: permafrost — frozen bloons emerge at half speed
  - ice  upgrade4: snap freeze — 39% chance to pop on freeze
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UpgradeSpec:
    cost: int
    # Stat deltas applied with +=.
    additive: dict[str, float] = field(default_factory=dict)
    # Stat values applied with =.
    absolute: dict[str, Any] = field(default_factory=dict)
    # Boolean flags flipped to True.
    flags: dict[str, bool] = field(default_factory=dict)
    # Transform upgrades (blade, glaive, missile, multishot) reset the firing
    # cooldown so the tower fires immediately after the upgrade.
    reset_tsls: bool = False


UPGRADES: dict[str, UpgradeSpec] = {
    # Dart Monkey — symmetric range / pierce on both paths.
    "dart1": UpgradeSpec(cost=90, additive={"attack_radius": 25}),
    "dart2": UpgradeSpec(cost=90, additive={"attack_radius": 25}),
    "dart3": UpgradeSpec(cost=140, additive={"pierce_max": 1}),
    "dart4": UpgradeSpec(cost=120, additive={"pierce_max": 1}),

    # Tack Shooter.
    "tack1": UpgradeSpec(cost=200, additive={"attack_rate": -15}),
    "tack2": UpgradeSpec(cost=180, additive={"attack_rate": -5},
                         flags={"transformed": True}, reset_tsls=True),
    "tack3": UpgradeSpec(cost=100, additive={"attack_radius": 10},
                         absolute={"bullet_scale": 1.3}),
    "tack4": UpgradeSpec(cost=100, additive={"attack_radius": 10},
                         absolute={"bullet_scale": 1.5}),

    # Boomerang.
    "boomerang1": UpgradeSpec(cost=270, additive={"pierce_max": 3}),
    "boomerang2": UpgradeSpec(cost=280, additive={"pierce_max": 3},
                              flags={"transformed": True}, reset_tsls=True),
    "boomerang3": UpgradeSpec(cost=150, flags={"icebreak": True}),
    "boomerang4": UpgradeSpec(cost=120, flags={"leadbreak": True}),

    # Bomb (Cannon). upgrade2 spawns frag bullets on detonation (handled in
    # game._on_hit).
    "bomb1": UpgradeSpec(cost=430, absolute={"bullet_scale": 1.5}),
    "bomb2": UpgradeSpec(cost=220),
    "bomb3": UpgradeSpec(cost=200, additive={"attack_radius": 20}),
    "bomb4": UpgradeSpec(cost=210, additive={"attack_rate": -8},
                         absolute={"shoot_power": 25.0},
                         flags={"transformed": True}, reset_tsls=True),

    # Ice Ball. upgrade2 permafrost / upgrade4 snap freeze are flag-only;
    # effects live in game._try_freeze and game._tick_bloons.
    "ice1": UpgradeSpec(cost=250, additive={"freeze_len": 20}),
    "ice2": UpgradeSpec(cost=250),
    "ice3": UpgradeSpec(cost=200, additive={"attack_radius": 15},
                        absolute={"bullet_scale": 1.0}),
    "ice4": UpgradeSpec(cost=290),

    # Super Monkey. Laser/plasma flip icebreak/leadbreak via the flags;
    # `laser=True` is kept as a marker (visualisation hook for future).
    "super1": UpgradeSpec(cost=1000, additive={"attack_radius": 50}),
    "super2": UpgradeSpec(cost=1400, additive={"attack_radius": 50}),
    "super3": UpgradeSpec(cost=3500, additive={"pierce_max": 1},
                          flags={"icebreak": True, "laser": True}),
    "super4": UpgradeSpec(cost=4000, additive={"pierce_max": 1},
                          absolute={"attack_rate": 1},
                          flags={"icebreak": True, "leadbreak": True, "laser": True}),

    # Spike-o-pult.
    "spikeopult1": UpgradeSpec(cost=250, additive={"attack_radius": 20}),
    "spikeopult2": UpgradeSpec(cost=825, absolute={"pierce_max": 20}),
    "spikeopult3": UpgradeSpec(cost=250, additive={"attack_rate": -8}),
    "spikeopult4": UpgradeSpec(cost=575, flags={"transformed": True, "is_spread": True},
                               reset_tsls=True),

    # Beacon. Drums (upgrade2) toggles the rate buff via beacon-refresh path.
    "beacon1": UpgradeSpec(cost=500, additive={"attack_radius": 30}),
    "beacon2": UpgradeSpec(cost=1500),
    # beacon3, beacon4 omitted (monkey storm — deferred).
}


def next_path_upgrade(tower_type: str, path: int,
                      u1: bool, u2: bool, u3: bool, u4: bool) -> str | None:
    """Return the upgrade name the given path button would buy next, or None
    if the path is exhausted or locked. Path 1 = upgrade1 then upgrade3,
    Path 2 = upgrade2 then upgrade4. Path 1 is locked once upgrade2 is bought."""
    if path == 1:
        if u2:
            return None
        if not u1:
            return f"{tower_type}1"
        if not u3:
            return f"{tower_type}3"
        return None
    if path == 2:
        if u4:
            return None
        if not u2:
            return f"{tower_type}2"
        name = f"{tower_type}4"
        # beacon4 is not a real upgrade (monkey-storm action); refuse.
        return name if name in UPGRADES else None
    raise ValueError(f"path must be 1 or 2, got {path}")
