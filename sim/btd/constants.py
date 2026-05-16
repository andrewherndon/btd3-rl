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

# Collision radii in px. Calibrated from frame-1 bbox of each rank's `inner`
# MovieClip — see sim/extract_bloon_hitboxes.py and paths/bloon_hitboxes.json
# (raw bboxes + offsets). Values here are `max(half_w, half_h)` rounded to int,
# treating every rank as a circle. Note black is genuinely smaller in BTD3
# (~50% of red in linear extent — confirmed visually). MOAB is elongated
# (115x80 in true AABB), so a single radius mildly over-covers vertically and
# under-covers horizontally; an AABB-vs-circle refactor would be more faithful
# but the agent's strategy doesn't hinge on it.
BLOON_RADIUS: Final = {
    1: 19, 2: 21, 3: 22, 4: 24, 5: 10,
    6: 19, 7: 21, 8: 23, 9: 23, 10: 57,
}

# Tower stats, from Tower.Init. attackRate in frames between shots; attackRadius
# in px; shootPower in px/frame (bullet speed); pierceMax = bloons per bullet.
# `icebreak` = bullet can pop frozen bloons; `leadbreak` = bullet can pop leads.
TOWER_STATS: Final = {
    "dart": dict(attackRate=33, attackRadius=100, shootPower=23.0, pierceMax=1,
                 cost=250, name="Dart Monkey",
                 icebreak=False, leadbreak=False, is_spread=False),
    "tack": dict(attackRate=54, attackRadius=70, shootPower=15.0, pierceMax=8,
                 cost=360, name="Tack Shooter",
                 icebreak=False, leadbreak=False, is_spread=True),
    "ice": dict(attackRate=93, attackRadius=60, shootPower=6.0, pierceMax=50,
                cost=425, name="Ice Ball", freeze_len=50,
                icebreak=False, leadbreak=False, is_spread=True),
    "bomb": dict(attackRate=54, attackRadius=120, shootPower=13.0, pierceMax=18,
                 cost=725, name="Cannon",
                 icebreak=True, leadbreak=True, is_spread=False),
    "spikeopult": dict(attackRate=63, attackRadius=110, shootPower=10.0, pierceMax=6,
                       cost=600, name="Spike-o-pult",
                       icebreak=False, leadbreak=False, is_spread=False),
    "super": dict(attackRate=2, attackRadius=140, shootPower=20.0, pierceMax=1,
                  cost=4000, name="Super Monkey",
                  icebreak=False, leadbreak=False, is_spread=False),
    # Boomerang. AS has shootPower=0; the bullet itself doesn't move and the
    # arc lives in the Boomerang MovieClip's per-frame keyframes (symbol 437,
    # 25 frames, depth=3 hitbit). We extracted those into paths/boomerang_arc
    # and apply them with a per-shot rotation. attack_radius is informational
    # for the targeting filter; the arc itself defines the engagement zone.
    "boomerang": dict(attackRate=50, attackRadius=130, shootPower=0.0, pierceMax=2,
                      cost=515, name="Boomerang",
                      icebreak=False, leadbreak=False, is_spread=False),
    # Beacon. Doesn't fire — buffs nearby towers' range. The drums upgrade
    # (deferred) adds a rate buff. is_attacker=False short-circuits the fire
    # path; the beacon still "ticks" so debug HUDs read normally.
    "beacon": dict(attackRate=60, attackRadius=120, shootPower=0.0, pierceMax=0,
                   cost=1000, name="Monkey Beacon",
                   icebreak=False, leadbreak=False, is_spread=False,
                   is_attacker=False),
}

# Beacon buff multipliers. AS multiplies `arsq` (radius²) by 1.2, NOT radius
# itself — so the effective range increase is sqrt(1.2) ≈ 1.095x. The AS
# CalcRadius uses 1.2 on the visual radius too, which is inconsistent in the
# original game; we mirror the targeting math exactly.
BEACON_RANGE_FACTOR: Final = 1.2
BEACON_RATE_FACTOR: Final = 0.85

# Bullet stats, from Bullet.Init. lifespan in frames. radius is sim-only
# (placeholder for hitTestObject -> circle replacement). `explosion_radius`
# is the post-detonation radius for two-stage bullets (bomb only). 0 means
# the bullet has no separate exploded stage.
BULLET_STATS: Final = {
    "dart": dict(lifespan=7, radius=4.0, explosion_radius=0.0),
    "tack": dict(lifespan=5, radius=4.0, explosion_radius=0.0),
    "ice": dict(lifespan=10, radius=4.0, explosion_radius=0.0),
    # AS lifespan=50, but the Boomerang MovieClip arc is 25 frames; after that
    # the bullet would stop at frame 1 (invisible). We match the visible arc.
    "boomerang": dict(lifespan=24, radius=6.0, explosion_radius=0.0),
    "bomb": dict(lifespan=18, radius=6.0, explosion_radius=30.0),
    # Bomb-with-frag upgrade (bomb2): on detonation, spawn N frag shards.
    # AS Bullet.Init: lifespan=5; not leadbreak / not icebreak.
    "frag": dict(lifespan=5, radius=4.0, explosion_radius=0.0),
    "spikeopult": dict(lifespan=20, radius=6.0, explosion_radius=0.0),
    "super": dict(lifespan=20, radius=4.0, explosion_radius=0.0),
}

# Spread towers fire `SPREAD_SHARDS` projectiles in a uniform fan when any
# bloon is in range (target is null in AS Shoot for type=="tack"/"ice"/"spikey").
# Total per-shot pierce is shards * 1, matching the AS pattern of `pierceMax`
# children each capable of one bloon hit.
SPREAD_SHARDS: Final = 8

# Spawn-time jitter range (uniform integer 0..9 inclusive on each axis), from
# BloonsTD.NewBloon: `_loc10_.x = _loc11_ + random(10)`.
SPAWN_JITTER_RANGE: Final = 10

# Radius (px) around the path centerline where towers cannot be placed. AS
# uses explicit Pathhit rectangles laid out per-track in the SWF
# (`pathhitmc_15`, symbol 538, frame N for track N); we approximate with a
# uniform buffer around the bloon's extracted centerline. A bloon's collision
# radius is up to ~16 px (MOAB excluded — they're flying anyway). Tower-vs-
# tower overlap is permitted (matches the right-click speedrun glitch).
PATH_PLACEMENT_BUFFER: Final = 16.0

# Round-end timing, from BloonsTD.EnterFrame.
ROUND_END_GRACE_FRAMES: Final = 20      # frames after last bloon clears
ROUND_END_TIMEOUT_FRAMES: Final = 5 * FPS  # emergency cutoff (was 5000 ms)
