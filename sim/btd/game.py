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
from .constants import (
    BLOON_CHILDREN,
    BLOON_ESCAPE_DAMAGE,
    BLOON_HITS,
    BLOON_MAXSPEED,
    BLOON_RADIUS,
    COST_MULT_BY_DIFF,
    LIVES_BY_DIFF,
    ROUND_END_GRACE_FRAMES,
    ROUND_END_TIMEOUT_FRAMES,
    SPAWN_JITTER_RANGE,
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

    # ------------------------------------------------------------------- API

    def place_tower(self, type_: str, x: float, y: float) -> int:
        """Buy and place a tower. Returns the tower id, or -1 if not enough money.
        Legality check (path overlap, tower overlap) is the caller's job for now."""
        if type_ not in TOWER_STATS:
            raise ValueError(f"unknown tower type: {type_}")
        price = self._price(TOWER_STATS[type_]["cost"])
        if price > self.money:
            return -1
        self.money -= price
        tid = self._next_tower_id
        self._next_tower_id += 1
        self.towers.append(Tower.from_type(tid, type_, x, y))
        return tid

    def sell_tower(self, tower_id: int) -> bool:
        for i, tower in enumerate(self.towers):
            if tower.id == tower_id:
                self.money += math.floor(0.8 * tower.spent_on_me)
                del self.towers[i]
                return True
        return False

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
        for t in self.towers:
            t.time_since_last_shot += 1
            if t.time_since_last_shot <= t.attack_rate:
                continue
            target = self._acquire_target(t)
            if target is None:
                continue
            t.time_since_last_shot = 0
            self._shoot(t, target)

    def _acquire_target(self, t: Tower) -> Optional[Bloon]:
        # AS GetTarget: scans bloon list, dist² < range², picks by progress.
        # AImode "first" = highest progress; "last" = lowest progress.
        ar_sq = t.attack_radius * t.attack_radius
        best: Optional[Bloon] = None
        best_progress = -1.0 if t.ai_mode == "first" else 2.0
        for b in self.bloons:
            if not b.alive:
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

    def _shoot(self, t: Tower, target: Bloon) -> None:
        dx = target.x - t.x
        dy = target.y - t.y
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
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
            b.x += b.vx
            b.y += b.vy

    def _tick_bloons(self) -> None:
        for b in self.bloons:
            if not b.alive:
                continue
            b.hit_this_frame = False
            if b.frozen:
                # Ice not implemented yet; placeholder for symmetry.
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
        if bloon.frozen and not bullet.icebreak and bullet.type != "ice":
            return
        if bullet.type in ("bomb", "pineapple") and bloon.rank == 5:
            return
        bloon.hits_remaining -= 1
        if bloon.hits_remaining <= 0:
            self._pop(bloon, bullet.shooter_id)

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
            self.spawn_bloon(
                rank=child_rank,
                branch=bloon.branch,
                frame=bloon.frame + frame_offset,
                jitter=(bloon.jitter_x, bloon.jitter_y),
            )

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
