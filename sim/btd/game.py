"""BloonsSim — the top-level simulator. Mirrors the orchestration logic in
BloonsTD.as (Init, StartLevel, EnterFrame, NewBloon, ShootBullet, PoppedOne,
Escaped, EndLevel)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .bloon import Bloon
from .bullet import Bullet
from .rounds import build_levels
from .upgrades import UPGRADES, UpgradeSpec, next_path_upgrade
from .constants import (
    BEACON_RANGE_FACTOR,
    BEACON_RATE_FACTOR,
    BLOON_CHILDREN,
    BLOON_ESCAPE_DAMAGE,
    BLOON_HITS,
    BLOON_MAXSPEED,
    BLOON_RADIUS,
    COST_MULT_BY_DIFF,
    LIVES_BY_DIFF,
    PATH_PLACEMENT_BUFFER,
    ROUND_END_GRACE_FRAMES,
    ROUND_END_TIMEOUT_FRAMES,
    SELL_RATE,
    SPAWN_JITTER_RANGE,
    SPREAD_SHARDS,
    STARTING_MONEY,
    TOWER_STATS,
)
from .tower import Tower


@dataclass
class SimConfig:
    track: int = 3
    difficulty: str = "easy"          # "easy" | "medium" | "hard"
    seed: int = 0
    # If False (default, matches the shipped game), the game ends after round 50.
    # If True, rounds 51+ (procedurally generated) are also playable.
    freeplay: bool = False
    paths_dir: Path = Path(__file__).resolve().parents[1] / "paths"


class BloonsSim:
    """One self-contained game. Tick with .step(); inspect with .observe()."""

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.config = config or SimConfig()
        # Spawn independent RNG sub-streams from the seed. Each "consumer"
        # (main game RNG, level generator, ...) gets its own. This way adding
        # new RNG-consuming features later does not shift the main stream.
        seeds = np.random.SeedSequence(self.config.seed).spawn(2)
        self.rng = np.random.default_rng(seeds[0])
        rounds_rng = np.random.default_rng(seeds[1])

        # Path data, keyed by branch. Track 3 has one branch.
        self.paths: dict[int, np.ndarray] = self._load_paths(self.config.track)
        # Bullet arcs keyed by bullet type. None if the bullet integrates via
        # vx/vy. Boomerang is currently the only one; super-laser etc. could
        # join later if they have keyframed trajectories.
        self.bullet_arcs: dict[str, np.ndarray] = self._load_bullet_arcs()

        # All rounds, generated once. Order of the embedded list IS the
        # spawn order for that round (the AS BuildLevels appends batches).
        self.round_table: dict[int, list[int]] = build_levels(
            rounds_rng, self.config.difficulty
        )
        self.max_round: int = max(self.round_table.keys())

        # Money / lives / round.
        self.money: int = STARTING_MONEY
        self.lives: int = LIVES_BY_DIFF[self.config.difficulty]
        self.cost_mult: float = COST_MULT_BY_DIFF[self.config.difficulty]
        self.round: int = 0
        self.glob_speed_mod: float = 0.0

        # Entities.
        self.bloons: list[Bloon] = []
        self.towers: list[Tower] = []
        self.bullets: list[Bullet] = []
        self._next_tower_id: int = 0

        # Frame / round state.
        self.frame_count: int = 0
        self.in_round: bool = False
        self.spawn_queue: list[int] = []       # ranks left to spawn this round
        self.spawn_counter: int = 0            # frames since last spawn
        self.bloon_interval: int = 20          # set per round
        self.frames_since_last_bloon: int = 0
        self.end_round_count: int = 0
        self.game_over: bool = False
        self.won: bool = False
        # Per-round bookkeeping.
        self.bloons_popped_this_round: int = 0

    # ------------------------------------------------------------------ paths

    def _load_paths(self, track: int) -> dict[int, np.ndarray]:
        paths_dir = self.config.paths_dir
        # Single-branch tracks (1, 2, 3, 5, 7) live as track_<n>.npy.
        # Branched tracks (4, 6, 8) live as track_<n>_<branch>.npy. We only
        # ship track 3 today.
        primary = paths_dir / f"track_{track}.npy"
        if primary.exists():
            return {1: np.load(primary)}
        # Multi-branch case: try to glob the branches.
        found: dict[int, np.ndarray] = {}
        for branch in (1, 2, 3):
            p = paths_dir / f"track_{track}_{branch}.npy"
            if p.exists():
                found[branch] = np.load(p)
        if not found:
            raise FileNotFoundError(f"No path data for track {track} in {paths_dir}")
        return found

    def _path(self, branch: int) -> np.ndarray:
        return self.paths[branch]

    def _load_bullet_arcs(self) -> dict[str, np.ndarray]:
        """Optional per-frame arc data for bullets whose trajectory is
        keyframed rather than ballistic. Returns a dict keyed by bullet type."""
        arcs: dict[str, np.ndarray] = {}
        boomerang = self.config.paths_dir / "boomerang_arc.npy"
        if boomerang.exists():
            arcs["boomerang"] = np.load(boomerang)
        return arcs

    # ------------------------------------------------------------------- API

    def place_tower(self, type_: str, x: float, y: float) -> int:
        """Buy and place a tower. Returns the tower id, or -1 if invalid
        (insufficient money OR position too close to the path).

        Tower-vs-tower overlap is *not* checked — the right-click placement
        glitch in BTD3 lets towers stack arbitrarily close, and speedruns rely
        on it. We preserve that."""
        if type_ not in TOWER_STATS:
            raise ValueError(f"unknown tower type: {type_}")
        if not self.is_placement_valid(x, y):
            return -1
        price = self._price(TOWER_STATS[type_]["cost"])
        if price > self.money:
            return -1
        self.money -= price
        tid = self._next_tower_id
        self._next_tower_id += 1
        self.towers.append(Tower.from_type(tid, type_, x, y))
        return tid

    def is_placement_valid(self, x: float, y: float) -> bool:
        """A position is valid iff its distance to every path branch's
        centerline exceeds `PATH_PLACEMENT_BUFFER`. Lives off the renderer so
        play.py can color the placement preview."""
        return self.distance_to_path(x, y) > PATH_PLACEMENT_BUFFER

    def distance_to_path(self, x: float, y: float) -> float:
        """Minimum Euclidean distance from (x, y) to any segment of any path
        branch. Vectorised over each branch's segments."""
        point = np.array([x, y], dtype=np.float64)
        best = float("inf")
        for branch_path in self.paths.values():
            a = branch_path[:-1]
            b = branch_path[1:]
            seg = b - a
            rel = point - a
            seg_len_sq = (seg ** 2).sum(axis=1)
            seg_len_sq = np.where(seg_len_sq == 0.0, 1.0, seg_len_sq)
            t = (rel * seg).sum(axis=1) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            closest = a + t[:, None] * seg
            d = np.linalg.norm(point - closest, axis=1).min()
            if d < best:
                best = float(d)
        return best

    def sell_tower(self, tower_id: int) -> bool:
        for i, tower in enumerate(self.towers):
            if tower.id == tower_id:
                self.money += math.floor(SELL_RATE * tower.spent_on_me)
                del self.towers[i]
                return True
        return False

    # -- upgrade API ----------------------------------------------------------

    def upgrade_tower(self, tower_id: int, upgrade_name: str) -> bool:
        """Buy a specific upgrade for a tower. Returns True on success."""
        tower = self._tower_by_id(tower_id)
        if tower is None or upgrade_name not in UPGRADES:
            return False
        if not self._can_upgrade(tower, upgrade_name):
            return False
        spec = UPGRADES[upgrade_name]
        price = self._price(spec.cost)
        if price > self.money:
            return False
        self.money -= price
        tower.spent_on_me += price
        self._apply_upgrade(tower, upgrade_name, spec)
        return True

    def upgrade_path(self, tower_id: int, path: int) -> bool:
        """Buy the next upgrade on `path` (1 or 2) for the tower."""
        tower = self._tower_by_id(tower_id)
        if tower is None:
            return False
        name = next_path_upgrade(
            tower.type, path,
            tower.upgrade1, tower.upgrade2, tower.upgrade3, tower.upgrade4,
        )
        if name is None:
            return False
        return self.upgrade_tower(tower_id, name)

    def available_upgrades(self, tower_id: int) -> dict[int, tuple[str, int] | None]:
        """Returns `{1: (name, price) or None, 2: (name, price) or None}`."""
        tower = self._tower_by_id(tower_id)
        if tower is None:
            return {1: None, 2: None}
        out: dict[int, tuple[str, int] | None] = {}
        for path in (1, 2):
            name = next_path_upgrade(
                tower.type, path,
                tower.upgrade1, tower.upgrade2, tower.upgrade3, tower.upgrade4,
            )
            out[path] = (name, self._price(UPGRADES[name].cost)) if name else None
        return out

    def _can_upgrade(self, tower, upgrade_name: str) -> bool:
        # Flags 1+2 are path 1 (level 1 then level 2); flags 3+4 are path 2.
        # No cross-path locking; each path just requires its level-1 first.
        if not upgrade_name.startswith(tower.type):
            return False
        suffix = upgrade_name[len(tower.type):]
        if suffix == "1":
            return not tower.upgrade1
        if suffix == "2":
            return tower.upgrade1 and not tower.upgrade2
        if suffix == "3":
            return not tower.upgrade3
        if suffix == "4":
            return tower.upgrade3 and not tower.upgrade4
        return False

    def _apply_upgrade(self, tower, upgrade_name: str, spec: UpgradeSpec) -> None:
        suffix = upgrade_name[len(tower.type):]
        setattr(tower, f"upgrade{suffix}", True)
        for attr, delta in spec.additive.items():
            setattr(tower, attr, getattr(tower, attr) + delta)
        for attr, value in spec.absolute.items():
            setattr(tower, attr, value)
        for attr, value in spec.flags.items():
            setattr(tower, attr, value)
        if spec.reset_tsls:
            tower.time_since_last_shot = 0

    def _tower_by_id(self, tower_id: int):
        for t in self.towers:
            if t.id == tower_id:
                return t
        return None

    def start_round(self) -> bool:
        """Begin the next round. Returns False if a round is still active."""
        if self.in_round:
            return False
        if self.game_over:
            return False
        self.round += 1
        self.in_round = True
        self.spawn_queue = self._round_data(self.round)
        self.spawn_counter = 0
        self.bloon_interval = self._round_interval(self.round)
        self.frames_since_last_bloon = 0
        self.end_round_count = 0
        self.bloons_popped_this_round = 0
        if self.round > 50:
            base = (self.round - 50) / 15.0
            bonus = {"easy": 0.0, "medium": 0.1, "hard": 0.25}[self.config.difficulty]
            self.glob_speed_mod = base + bonus
        else:
            self.glob_speed_mod = 0.0
        return True

    def step(self) -> None:
        """Advance one game frame (1/40 s of in-game time)."""
        if self.game_over:
            return
        self.frame_count += 1

        if self.in_round:
            self._tick_spawns()

        self._tick_towers()
        self._tick_bullets()
        self._tick_bloons()
        self._tick_collisions()
        self._cleanup()
        self._tick_round_end()

    # -- debug helpers (interactive playtest, not gameplay API) ---------------

    def debug_add_money(self, amount: int) -> None:
        self.money = max(0, self.money + amount)

    def debug_add_lives(self, amount: int) -> None:
        self.lives = max(0, self.lives + amount)
        if self.lives > 0 and self.game_over and not self.won:
            self.game_over = False  # rescuable from a loss

    def debug_set_round(self, round_num: int) -> bool:
        """Set the *upcoming* round (i.e. what `start_round` will play next).
        Refuses if a round is in progress. Returns True on success."""
        if self.in_round:
            return False
        self.round = max(0, min(round_num - 1, self.max_round))
        return True

    def debug_clear_bloons(self) -> None:
        """Wipe all bloons without awarding money, spawning children, or
        leaking lives. Lets you abort a bad round mid-stream."""
        for b in self.bloons:
            b.popped = True

    def observe(self) -> dict:
        """Tiny human-readable snapshot. The Gym wrapper will produce a
        structured array form for training."""
        return {
            "frame": self.frame_count,
            "round": self.round,
            "in_round": self.in_round,
            "money": self.money,
            "lives": self.lives,
            "game_over": self.game_over,
            "won": self.won,
            "n_bloons": sum(b.alive for b in self.bloons),
            "n_bullets": sum(not b.is_dead for b in self.bullets),
            "n_towers": len(self.towers),
            "pops_this_round": self.bloons_popped_this_round,
        }

    # -------------------------------------------------------------- internals

    def _price(self, base_cost: int) -> int:
        # BloonsTD.GetPrice: round((cost * costmult) / 5) * 5.
        return int(round((base_cost * self.cost_mult) / 5.0)) * 5

    def _round_data(self, round_num: int) -> list[int]:
        # Return a copy so callers (the spawn queue) can pop without mutating
        # the source-of-truth table.
        return list(self.round_table.get(round_num, []))

    def _round_interval(self, round_num: int) -> int:
        # BloonsTD.StartLevel: 20 - round, clamped via ceil(7 - round/20).
        v = 20 - round_num
        if v < 7:
            v = math.ceil(7 - round_num / 20.0)
        return max(1, v)

    # -- spawn -----------------------------------------------------------------

    def _tick_spawns(self) -> None:
        if not self.spawn_queue:
            return
        self.spawn_counter += 1
        if self.spawn_counter > self.bloon_interval:
            self.spawn_counter = 0
            rank = self.spawn_queue.pop(0)
            self.spawn_bloon(rank=rank, branch=1, frame=0.0)
            self.frames_since_last_bloon = 0

    def spawn_bloon(
        self,
        rank: int,
        branch: int,
        frame: float,
        jitter: Optional[tuple[float, float]] = None,
    ) -> Bloon:
        if jitter is None:
            jitter = (
                float(self.rng.integers(0, SPAWN_JITTER_RANGE)),
                float(self.rng.integers(0, SPAWN_JITTER_RANGE)),
            )
        maxspeed = BLOON_MAXSPEED[rank] + self.glob_speed_mod
        bloon = Bloon(
            rank=rank,
            frame=max(0.0, frame),
            maxspeed=maxspeed,
            speed=maxspeed,
            jitter_x=jitter[0],
            jitter_y=jitter[1],
            branch=branch,
            hits_remaining=BLOON_HITS.get(rank, 1),
            radius=BLOON_RADIUS[rank],
        )
        self._refresh_position(bloon)
        self.bloons.append(bloon)
        return bloon

    def _refresh_position(self, b: Bloon) -> None:
        path = self._path(b.branch)
        idx = min(int(round(b.frame)), len(path) - 1)
        b.x = path[idx, 0] + b.jitter_x
        b.y = path[idx, 1] + b.jitter_y

    # -- tower / bullet / bloon ticks -----------------------------------------

    def _tick_towers(self) -> None:
        # Recompute beacon buffs from scratch every frame. O(beacons × non-
        # beacon towers); cheap in practice. Matches AS gameplay closely and
        # avoids the AS quirk where selling one beacon clears flags that
        # another beacon would re-apply only on its next refresh cycle.
        self._refresh_beacon_buffs()
        for t in self.towers:
            t.time_since_last_shot += 1
            if not t.is_attacker:
                continue
            effective_rate = t.attack_rate
            if t.beacon_rate_active:
                effective_rate = max(1, math.ceil(effective_rate * BEACON_RATE_FACTOR))
            if t.time_since_last_shot <= effective_rate:
                continue
            target = self._acquire_target(t)
            if target is None:
                # Spread towers still need a target-in-range gate. AS calls
                # GetTarget(), then nulls the target before ShootBullet — the
                # presence of a bloon in range is what triggers the volley.
                continue
            t.time_since_last_shot = 0
            if t.is_spread:
                self._shoot_spread(t)
            else:
                self._shoot(t, target)

    def _refresh_beacon_buffs(self) -> None:
        # Wipe all flags, then re-apply from each beacon. Non-beacons don't
        # propagate buffs; beacons never buff themselves (AS doBeaconUpdate
        # has `if(_loc2_.type != "beacon")`).
        for t in self.towers:
            if t.type == "beacon":
                continue
            t.beacon_radius_active = False
            t.beacon_rate_active = False
        for beacon in self.towers:
            if beacon.type != "beacon":
                continue
            ar_sq = beacon.attack_radius * beacon.attack_radius
            for t in self.towers:
                if t.type == "beacon":
                    continue
                dx = t.x - beacon.x
                dy = t.y - beacon.y
                if dx * dx + dy * dy < ar_sq:
                    t.beacon_radius_active = True
                    # Drums upgrade (deferred): also flip beacon_rate_active.

    def _acquire_target(self, t: Tower) -> Optional[Bloon]:
        # AS GetTarget: scans bloon list, dist² < range², picks by progress.
        # AImode "first" = highest progress; "last" = lowest progress.
        # Non-icebreak towers skip frozen bloons (AS GetTarget: `if(!icebreak)
        # if(_loc4_.frozen) continue`).
        # Beacon range buff multiplies arsq by 1.2 (NOT radius — AS quirk;
        # effective range gain is sqrt(1.2) ≈ 1.095x).
        ar_sq = t.attack_radius * t.attack_radius
        if t.beacon_radius_active:
            ar_sq *= BEACON_RANGE_FACTOR
        best: Optional[Bloon] = None
        best_progress = -1.0 if t.ai_mode == "first" else 2.0
        for b in self.bloons:
            if not b.alive:
                continue
            if b.frozen and not t.icebreak:
                continue
            dx = b.x - t.x
            dy = b.y - t.y
            if dx * dx + dy * dy >= ar_sq:
                continue
            progress = b.frame / max(1, len(self._path(b.branch)))
            if t.ai_mode == "first":
                if progress > best_progress:
                    best_progress = progress
                    best = b
            else:
                if progress < best_progress:
                    best_progress = progress
                    best = b
        return best

    def _shoot_spread(self, t: Tower) -> None:
        # AS spread bullet is a single MovieClip with N visual sub-projectiles;
        # we model it as N independent unit-pierce bullets fanning out evenly.
        # Total per-volley pierce = SPREAD_SHARDS for tack (matches AS
        # pierce_max=8). Ice has pierce_max=50 on the tower; each shard is
        # still unit-pierce and just freezes the bloon it hits.
        n = SPREAD_SHARDS
        for i in range(n):
            angle = (2.0 * math.pi * i) / n
            ux = math.cos(angle)
            uy = math.sin(angle)
            bullet = Bullet.from_type(
                type_=t.type,
                x=t.x + ux * 10.0,
                y=t.y + uy * 10.0,
                vx=ux * t.shoot_power,
                vy=uy * t.shoot_power,
                pierce_max=1,
                shooter_id=t.id,
                icebreak=t.icebreak,
                leadbreak=t.leadbreak,
                freeze_len=t.freeze_len,
                scale=t.bullet_scale,
            )
            self.bullets.append(bullet)

    def _shoot(self, t: Tower, target: Bloon) -> None:
        dx = target.x - t.x
        dy = target.y - t.y
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist

        if t.type == "boomerang" and "boomerang" in self.bullet_arcs:
            # Boomerang uses a keyframed arc (extracted from the SWF). Its
            # "forward" axis in the local frame is -y, so rotate by
            # atan2(ux, -uy) to align the arc with the shot direction.
            angle = math.atan2(ux, -uy)
            c = math.cos(angle)
            s = math.sin(angle)
            arc0 = self.bullet_arcs["boomerang"][0]
            x0 = t.x + arc0[0] * c - arc0[1] * s
            y0 = t.y + arc0[0] * s + arc0[1] * c
            bullet = Bullet.from_type(
                type_=t.type,
                x=x0,
                y=y0,
                vx=0.0,
                vy=0.0,
                pierce_max=t.pierce_max,
                shooter_id=t.id,
                icebreak=t.icebreak,
                leadbreak=t.leadbreak,
                scale=t.bullet_scale,
            )
            bullet.arc_anchor_x = t.x
            bullet.arc_anchor_y = t.y
            bullet.arc_angle = angle
            self.bullets.append(bullet)
            return

        # AS ShootBullet places the bullet 10 px out from the tower along the
        # shot vector; this matters because the bullet doesn't have to traverse
        # the tower body. Bullets inherit icebreak / leadbreak from the shooter.
        bullet = Bullet.from_type(
            type_=t.type,
            x=t.x + ux * 10.0,
            y=t.y + uy * 10.0,
            vx=ux * t.shoot_power,
            vy=uy * t.shoot_power,
            pierce_max=t.pierce_max,
            shooter_id=t.id,
            icebreak=t.icebreak,
            leadbreak=t.leadbreak,
            scale=t.bullet_scale,
        )
        self.bullets.append(bullet)

    def _tick_bullets(self) -> None:
        for b in self.bullets:
            if b.is_dead:
                continue
            b.time_alive += 1
            if b.time_alive > b.lifespan:
                b.is_dead = True
                continue
            arc = self.bullet_arcs.get(b.type)
            if arc is not None:
                idx = min(b.time_alive, len(arc) - 1)
                lx, ly = arc[idx]
                c = math.cos(b.arc_angle)
                s = math.sin(b.arc_angle)
                b.x = b.arc_anchor_x + lx * c - ly * s
                b.y = b.arc_anchor_y + lx * s + ly * c
            else:
                b.x += b.vx
                b.y += b.vy

    def _tick_bloons(self) -> None:
        for b in self.bloons:
            if not b.alive:
                continue
            b.hit_this_frame = False
            if b.frozen:
                b.time_frozen += 1
                if b.time_frozen > b.freeze_duration:
                    b.frozen = False
                    b.time_frozen = 0
                # Frozen bloons hold position; refresh in case jitter / branch
                # state changed elsewhere.
                self._refresh_position(b)
                continue
            b.frame += b.speed
            path_len = len(self._path(b.branch))
            if int(round(b.frame)) >= path_len:
                b.escaped = True
                self._on_escape(b)
                continue
            self._refresh_position(b)

    def _tick_collisions(self) -> None:
        # Iterate bullets, each can hit up to pierce_max bloons. A bloon
        # absorbs at most one bullet per frame (AS Bloon.Update returns after
        # first hit).
        for bullet in self.bullets:
            if bullet.is_dead:
                continue
            for bloon in self.bloons:
                if not bloon.alive or bloon.hit_this_frame:
                    continue
                if self._circle_hit(bullet, bloon):
                    bloon.hit_this_frame = True
                    bullet.pierce_count += 1
                    self._on_hit(bullet, bloon)
                    if bullet.pierce_count >= bullet.pierce_max:
                        bullet.is_dead = True
                        break

    @staticmethod
    def _circle_hit(bullet: Bullet, bloon: Bloon) -> bool:
        dx = bullet.x - bloon.x
        dy = bullet.y - bloon.y
        r = bullet.radius + bloon.radius
        return dx * dx + dy * dy <= r * r

    # -- pop / hit / escape ----------------------------------------------------

    def _on_hit(self, bullet: Bullet, bloon: Bloon) -> None:
        # Order mirrors the AS Bloon.Update collision branch:
        #   1. Lead clink (rank 7 + non-leadbreak + non-ice): big pierce penalty,
        #      no pop. AS adds 5 to pierceCount; _tick_collisions already added 1,
        #      so we add 4 more.
        #   2. Bomb two-stage: on the bomb's first hit, stop moving and switch
        #      to explosion radius. Continues to pop bloons via subsequent
        #      collision-loop iterations (matches AS's bloon-by-bloon
        #      explosion sweep, just collapsed into one frame).
        #   3. Frozen clink (frozen + non-icebreak + non-ice): no pop.
        #   4. Black bomb-immunity (rank 5 + bomb/pineapple): no pop. AS:
        #      `if((bomb or pineapple) && rank != 9 && rank != 10) if(rank == 5)
        #      popped=false; done=false; return`.
        # Otherwise pop (MOAB / ceramic decrement hits_remaining first).
        if bloon.rank == 7 and not bullet.leadbreak and bullet.type != "ice":
            bullet.pierce_count += 4
            return
        if bullet.type == "bomb" and not bullet.hashit:
            bullet.hashit = True
            bullet.vx = 0.0
            bullet.vy = 0.0
            if bullet.explosion_radius > 0.0:
                bullet.radius = bullet.explosion_radius
            # Bomb upgrade2 (frag): spawn shards from the detonation point.
            shooter = self._tower_by_id(bullet.shooter_id)
            if shooter is not None and shooter.upgrade2:
                self._spawn_frags(bullet.x, bullet.y, shooter)
        if bloon.frozen and not bullet.icebreak and bullet.type != "ice":
            return
        if bullet.type in ("bomb", "pineapple") and bloon.rank == 5:
            return
        # Ice doesn't pop — it freezes. Snap-freeze + permafrost are upgrades
        # and not in scope yet.
        if bullet.type == "ice":
            self._try_freeze(bloon, bullet)
            return
        bloon.hits_remaining -= 1
        if bloon.hits_remaining <= 0:
            self._pop(bloon, bullet.shooter_id)

    def _try_freeze(self, bloon: Bloon, bullet: Bullet) -> None:
        # AS Bloon.Update: only freezeMe if !frozen && rank != 6; and freezeMe
        # itself early-returns for rank 9, 10.
        if bloon.frozen:
            return
        if bloon.rank == 6:
            return
        if bloon.rank in (9, 10):
            return
        bloon.frozen = True
        bloon.time_frozen = 0
        bloon.freeze_duration = min(bullet.freeze_len, 100)
        bloon.freezer_id = bullet.shooter_id
        shooter = self._tower_by_id(bullet.shooter_id)
        if shooter is None:
            return
        # Permafrost (ice upgrade2): halve speed on freeze. AS gates with
        # `speed == maxspeed`, preventing stacking on already-permafrosted bloons.
        if shooter.upgrade2:
            if bloon.speed == bloon.maxspeed and bloon.rank != 10:
                bloon.speed /= 2.0
        # Snap freeze (ice upgrade4): 39% chance to instantly pop the bloon.
        # AS: `random(100) > 60` proc — i.e. roll in [61, 99] = 39 outcomes / 100.
        if shooter.upgrade4:
            if int(self.rng.integers(0, 100)) > 60:
                bloon.snap_frozen = True
                bloon.hits_remaining -= 1
                if bloon.hits_remaining <= 0:
                    self._pop(bloon, shooter.id)

    def _spawn_frags(self, x: float, y: float, shooter) -> None:
        # AS spawns frag bullets at the bomb's detonation point with vx/vy=0
        # (the visual fan is in the Frags MovieClip keyframes). We synthesise
        # an 8-way fan at a moderate speed so the AoE has a real footprint.
        speed = 10.0
        for i in range(SPREAD_SHARDS):
            angle = (2.0 * math.pi * i) / SPREAD_SHARDS
            ux = math.cos(angle)
            uy = math.sin(angle)
            bullet = Bullet.from_type(
                type_="frag",
                x=x + ux * 4.0,
                y=y + uy * 4.0,
                vx=ux * speed,
                vy=uy * speed,
                pierce_max=1,
                shooter_id=shooter.id,
                icebreak=False,
                leadbreak=False,
            )
            self.bullets.append(bullet)

    def _pop(self, bloon: Bloon, shooter_id: int) -> None:
        bloon.popped = True
        self.bloons_popped_this_round += 1
        self._award_pop_money()
        # Credit the tower.
        for t in self.towers:
            if t.id == shooter_id:
                t.pop_count += 1
                break
        # Spawn children at parent's progress ± offset, inheriting jitter & branch.
        for child_rank, frame_offset in BLOON_CHILDREN.get(bloon.rank, []):
            child = self.spawn_bloon(
                rank=child_rank,
                branch=bloon.branch,
                frame=bloon.frame + frame_offset,
                jitter=(bloon.jitter_x, bloon.jitter_y),
            )
            # Snap-freeze inheritance (AS NewBloon param7 + freezeMe(false)):
            # children of a snap-frozen parent are born frozen and inherit the
            # original freezer's permafrost effect. The chain stops here — the
            # children's `snap_frozen` stays False, so grand-children spawn
            # unfrozen unless they get a fresh snap-freeze hit.
            if bloon.snap_frozen and child.rank not in (6, 9, 10):
                child.frozen = True
                child.time_frozen = 0
                child.freeze_duration = bloon.freeze_duration
                child.freezer_id = bloon.freezer_id
                freezer = self._tower_by_id(bloon.freezer_id)
                if freezer is not None and freezer.upgrade2:
                    if child.speed == child.maxspeed and child.rank != 10:
                        child.speed /= 2.0

    def _award_pop_money(self) -> None:
        # BloonsTD.PoppedOne: 1$ pre-r51, 1/3 chance r51-59, 1/5 chance r60+.
        if self.round < 51:
            self.money += 1
        elif self.round < 60:
            if int(self.rng.integers(0, 3)) == 0:
                self.money += 1
        else:
            if int(self.rng.integers(0, 5)) == 0:
                self.money += 1

    def _on_escape(self, bloon: Bloon) -> None:
        if not self.in_round:
            return
        self.lives -= BLOON_ESCAPE_DAMAGE[bloon.rank]
        if self.lives <= 0:
            self.lives = 0
            self.game_over = True
            self.won = False

    # -- round end / cleanup ---------------------------------------------------

    def _tick_round_end(self) -> None:
        # AS BloonsTD.EnterFrame: the round ends only when the spawn queue is
        # drained AND no bloons remain alive. Two parallel conditions:
        #   - 20-frame grace (endRoundCount > 20) — the normal path.
        #   - 5-second timeout — emergency fallback if nothing has spawned and
        #     no bloons remain. Redundant in normal play; matches AS.
        if not self.in_round or self.spawn_queue:
            self.frames_since_last_bloon = 0
            return
        if any(b.alive for b in self.bloons):
            self.frames_since_last_bloon = 0
            self.end_round_count = 0
            return
        self.end_round_count += 1
        self.frames_since_last_bloon += 1
        if self.end_round_count > ROUND_END_GRACE_FRAMES:
            self._finish_round()
        elif self.frames_since_last_bloon > ROUND_END_TIMEOUT_FRAMES:
            self._finish_round()

    def _finish_round(self) -> None:
        self.in_round = False
        # Clear remaining bullets (AS EndLevel: ClearBullets, ClearBalloons).
        for b in self.bullets:
            b.is_dead = True
        win_round = self.max_round if self.config.freeplay else 50
        if self.round >= win_round:
            self.game_over = True
            self.won = True
            return
        self.money += 99 + self.round

    def _cleanup(self) -> None:
        if self.bullets:
            self.bullets = [b for b in self.bullets if not b.is_dead]
        if self.bloons:
            self.bloons = [b for b in self.bloons if b.alive]
