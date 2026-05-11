"""Game balance constants. Mirrors the static fields of BloonsTD.as and the
per-type stat blocks in Tower.Init / Bullet.Init."""

from __future__ import annotations

from typing import Final

FPS: Final = 40
STAGE_W: Final = 640
STAGE_H: Final = 480

STARTING_MONEY: Final = 650
SELL_RATE: Final = 0.8

LIVES_BY_DIFF: Final = {"easy": 100, "medium": 75, "hard": 50}
COST_MULT_BY_DIFF: Final = {"easy": 0.85, "medium": 1.02, "hard": 1.08}

# Per-rank max path-frames-per-game-frame, from Bloon.Init switch.
BLOON_MAXSPEED: Final = {
    1: 1.0, 2: 1.4, 3: 1.8, 4: 3.2, 5: 1.8,
    6: 2.5, 7: 1.0, 8: 2.2, 9: 2.5, 10: 1.0,
}

# Names by rank, for debugging. Level hint 33 ("brown bloons ... ceramic, take
# several hits") confirms rank 9 = ceramic; level hint 36 ("beware the M.O.A.B
# its coming next level") + round-37 data (1x rank-10) confirms rank 10 = MOAB.
BLOON_NAMES: Final = {
    1: "red", 2: "blue", 3: "green", 4: "yellow",
    5: "black", 6: "white", 7: "lead", 8: "rainbow",
    9: "ceramic", 10: "MOAB",
}

# Lives lost on escape, from BloonsTD.Escaped.
BLOON_ESCAPE_DAMAGE: Final = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 9,
    6: 9, 7: 19, 8: 37, 9: 38, 10: 100,
}

# Hits to pop. Everything not listed is 1 hit.
BLOON_HITS: Final = {9: 8, 10: 130}

# Pop hierarchy from Bloon.RemoveMe: rank -> list of (child_rank, frame_offset).
# Frame offset is what the parent passes as currentFrame ± delta when calling NewBloon.
BLOON_CHILDREN: Final = {
    2: [(1, 0)],
    3: [(2, 0)],
    4: [(3, 0)],
    5: [(4, 5), (4, -5)],
    6: [(4, 5), (4, -5)],
    7: [(5, 4), (5, -4)],
    8: [(5, 5), (5, 1), (6, -1), (6, -5)],
    9: [(8, 6), (8, -6)],
    10: [(9, 5), (9, 2), (9, -2), (9, -5)],
}

# Placeholder collision radii in px. Refine later from extracted bounds of
# each rank's `inner` MovieClip.
BLOON_RADIUS: Final = {
    1: 12, 2: 13, 3: 14, 4: 13, 5: 14,
    6: 14, 7: 15, 8: 16, 9: 30, 10: 60,
}

# Tower stats, from Tower.Init. attackRate in frames between shots; attackRadius
# in px; shootPower in px/frame (bullet speed); pierceMax = bloons per bullet.
# `icebreak` = bullet can pop frozen bloons; `leadbreak` = bullet can pop leads.
TOWER_STATS: Final = {
    "dart": dict(attackRate=33, attackRadius=100, shootPower=23.0, pierceMax=1,
                 cost=250, name="Dart Monkey", icebreak=False, leadbreak=False),
    "bomb": dict(attackRate=54, attackRadius=120, shootPower=13.0, pierceMax=18,
                 cost=725, name="Cannon", icebreak=True, leadbreak=True),
}

# Bullet stats, from Bullet.Init. lifespan in frames. radius is sim-only
# (placeholder for hitTestObject -> circle replacement). `explosion_radius`
# is the post-detonation radius for two-stage bullets (bomb only). 0 means
# the bullet has no separate exploded stage.
BULLET_STATS: Final = {
    "dart": dict(lifespan=7, radius=4.0, explosion_radius=0.0),
    "bomb": dict(lifespan=18, radius=6.0, explosion_radius=30.0),
}

# Spawn-time jitter range (uniform integer 0..9 inclusive on each axis), from
# BloonsTD.NewBloon: `_loc10_.x = _loc11_ + random(10)`.
SPAWN_JITTER_RANGE: Final = 10

# Round-end timing, from BloonsTD.EnterFrame.
ROUND_END_GRACE_FRAMES: Final = 20      # frames after last bloon clears
ROUND_END_TIMEOUT_FRAMES: Final = 5 * FPS  # emergency cutoff (was 5000 ms)
