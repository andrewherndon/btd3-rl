# BTD3 Decompile — Findings & Sim Notes

Source: `Decompiled-Limited/scripts/` (ActionScript 3 from the BTD3 SWF). Art lives in `assets.swf` (not decompiled here).

## How the game is structured

- The whole game is one Flash MovieClip with timeline frames driving state: load (frame 4) → intro (7) → menu (10) → init (11) → main (12). `Init()` wires UI, calls `BuildLevels()`, then waits on `EnterFrame`.
- `BloonsTD.EnterFrame()` is the game loop. Runs at SWF framerate = **40 fps** (`frameRate="40.0"` in SWF header). Stage = 640×480 px (12800×9600 twips). Per tick:
  1. Spawn next bloon if `counter > bloonInterval`.
  2. `UpdateTowers()` → each tower ticks `timeSinceLastShot`, shoots if past `attackRate`.
  3. `UpdateBullets()` → integrate `x += vx; y += vy`, expire on `lifespan`.
  4. Each bloon's own `Update()` runs (registered as `ENTER_FRAME` listener) — advances path frame, tests collision against every bullet.
- Collision = Flash `hitTestObject` (axis-aligned bbox of the rendered sprite). Replace with circle-vs-circle in the sim.
- Bloon update order is the order bloons were added to `bloonholder` (insertion order). Tower `GetTarget` iterates the same order; ties in progress fraction break by spawn order.
- Round end: normally fires 20 frames after `numBloons` hits 0 (`endRoundCount > 20`). Emergency timeout: if no bloon spawned in last 5000 ms real time and none alive, end immediately.

## Bloons (`Bloon.as`)

- Rank 1–10. `maxspeed` per rank: 1, 1.4, 1.8, **3.2** (yellow), 1.8, 2.5, 1, 2.2, 2.5 (MOAB), 1 (BFB). Units = path-timeline-frames per game-frame.
- `Bloon.Init` does `maxspeed += game.globSpeedMod` once, so the round speed bonus is baked into each bloon at spawn time. Frozen speed = maxspeed; permafrost (`upgrade2`) halves it after thaw.
- Path is **baked into the MovieClip timeline of each `Bloon_<rank>_<track>` class.** `Update` does `frame += speed; gotoAndStop(round(frame))`; the rendered (x, y) is whatever the SWF keyframed for that frame. No explicit path array exists in code.
- Bloons are placed at stage position `(_loc11_, _loc12_) + (random(10), random(10))` — small jitter on spawn. See TRACK_OFFSETS table below.
- Children spawned by a popped bloon inherit the parent's path progress (`currentFrame/totalFrames`) and the parent's actual stage `x`, `y`. So a yellow popping at 60% of the path produces a red at the same 60% mark, not at the path start.
- Progress fraction (used by tower targeting) = `currentFrame / totalFrames`.
- Tracks 4, 6, 8 have branching paths — separate `Bloon_*_<track>_<branch>` classes (branches 1–2 or 1–3). `NewBloon` randomly assigns a branch when spawning the top-level bloon; children inherit it via the `side` field.
- Track 7 reuses track-4 MovieClip geometry (instantiates `Bloon_*_4` classes) with a different stage offset (240, 378). It is *not* branched — only branch 1 is reachable from a track-7 spawn.
- Pop hierarchy (`RemoveMe`):
  - rank 2–4 → one child of `rank-1`
  - rank 5, 6 → 2× rank-4
  - rank 7 (lead) → 2× rank-5
  - rank 8 (rainbow) → 2× rank-5 + 2× rank-6
  - rank 9 (MOAB) → 2× rank-8 (needs 8 hits before popping)
  - rank 10 (BFB) → 4× rank-9 (needs 130 hits before popping)
- Pop immunities (in `Pop` and `Update`):
  - rank 5 (black) immune to bomb / pineapple
  - rank 6 (white) immune to ice
  - rank 7 (lead) immune to non-`leadbreak` sharp bullets — clinks (`pierceCount += 5`)
  - frozen bloons immune to non-`icebreak` bullets — clinks
- Escape damage (`Escaped`): 1, 2, 3, 4, 9, 9, 19, 37, 38, 100 by rank.

## Towers (`Tower.as`)

- Static stats set in `Init()` by `type`: `attackRate` (frames between shots), `attackRadius` (px), `pierceMax`, `shootPower` (bullet speed), `freezeLen`, `pierceMax`, plus flags `isspread`, `icebreak`, `leadbreak`.
- Targeting: `GetTarget()` scans `bloonholder`, filters by `dist² < arsq` (skips frozen unless `icebreak`), picks by `AImode` — `"first"` = max progress, `"last"` = min progress. Only two modes exist; no "strong"/"close". `AImode` is set to `"first"` in `Tower()` and never reassigned anywhere — every tower in the shipped game targets first.
- Upgrades: 4 per tower, two paths (1+3 / 2+4). Resolved in `BloonsTD.GetUpgrade()` as a flat switch over `"<type><n>"`. Flips a boolean on the tower and mutates a stat (e.g. range +25, rate -15, `transformed=true`).
- Beacon: not an attacker. `doBeaconUpdate()` flips `beaconRadius` (→ ×1.2 range) and `beaconRate` (→ ×0.85 rate) on towers within its radius.
- Sell refund: `floor(SELL_RATE * spentonme)`, `SELL_RATE = 0.8`. `spentonme` accumulates base + upgrade costs.
- Difficulty cost multiplier: `GetPrice(c) = round((c * costmult) / 5) * 5`. `costmult` = 0.85 / 1.02 / 1.08 (easy/med/hard).

## Bullets (`Bullet.as`)

- One MovieClip per shot. `Init()` sets `lifespan` (frames) and `pierceMax` per type. Spread bullets (`tack`) use 8 child sub-projectiles indexed `tack1..tack8`.
- Bomb: on first hit → `vx = vy = 0`, plays explosion. If shooter has `upgrade2` (missile/frag), spawns `Frags` from impact point. The bomb stays alive for its full `lifespan` after detonation so its explosion bbox keeps hitting bloons until `pierceMax` is exhausted.
- Pineapples idle until they explode (the bloon-side check is `if(_loc3_.type == "pineapple") if(!_loc3_.exploded) continue`). The `exploded` flag is set by frame script on the pineapple MovieClip itself.
- Ice freeze: when frozen, bloon stops advancing and `timeFrozen++`. Thaws when `timeFrozen > min(freezer.freezeLen, 100)`. Default `freezeLen` = 50 (=1.25 s at 40 fps). Permafrost (`upgrade2`) halves speed permanently after thaw. Snap freeze (`upgrade4`): 40% chance to pop instantly on freeze for non-MOAB/BFB.
- `RoadSpikes`, `Glue`, `Pineapple` are placed by player click via `ShootBullet(false, false)` with `vx = vy = 0` and long `lifespan`.

## Rounds / economy

- `BuildLevels()` (BloonsTD.as:1529) hardcodes rounds 1–50 via `ABSTL(count, round, rank)`. Rounds 51+ procedural, random rank biased up by difficulty.
- `levelsArray[round-1]` = flat array of ranks; `bloonIndex` walks it on each spawn.
- `bloonInterval = max(20 - round, ceil(7 - round/20))`. Tracks 4/6/8 get ×1.3 interval (multi-path).
- Round 51+: `globSpeedMod = (round - 50) / 15` (reassigned each round, not accumulated; +0.1 medium, +0.25 hard). Added to every new bloon's `maxspeed` at `Bloon.Init`.
- Round reward: `99 + round`. Pop reward: 1$ per pop pre-round-51, 1/3 chance pre-round-60, 1/5 chance after.
- Starting money: 650. Lives: 100 / 75 / 50 by difficulty.
- Special: `MonkeyStorm` ($1000, unlocked by Beacon upgrade2) — kills all non-rank-10 bloons in its hitbox.

## Per-track stage offsets (from `BloonsTD.NewBloon`)

Bloon MovieClips are placed at these stage `(x, y)` positions; path coords inside the MovieClip are local to that origin.

| track | branch | offset (x, y) | bloon class       |
|-------|--------|---------------|-------------------|
| 1     | —      | (-54, 14)     | Bloon_<r>_1       |
| 2     | —      | (-48, -133)   | Bloon_<r>_2       |
| 3     | —      | (47, -174)    | Bloon_<r>_3       |
| 4     | 1      | (337, -164)   | Bloon_<r>_4       |
| 4     | 2      | (82, -178)    | Bloon_<r>_4_2 *   |
| 5     | —      | (-35, -15)    | Bloon_<r>_5       |
| 6     | 1      | (140, -156)   | Bloon_<r>_6_1     |
| 6     | 2      | (-65, 135)    | Bloon_<r>_6_2     |
| 6     | 3      | (-85, -135)   | Bloon_<r>_6_3     |
| 7     | —      | (240, 378)    | Bloon_<r>_4 (reuses track-4 geometry) |
| 8     | 1      | (250, -175)   | Bloon_<r>_8_1     |
| 8     | 2      | (-66, -66)    | Bloon_<r>_8_2     |
| 8     | 3      | (-72, 212)    | Bloon_<r>_8_3     |

\* Track 4 branch class names in the AS file use the pattern `Bloon_<r>_7_1` / `Bloon_<r>_7_2`; the in-code switch on `trackNum==4` is what maps a track-4 spawn to those classes. Branch randomization: track 4 = `random(2)+1`, tracks 6/8 = `random(3)+1`. Rank-10 (BFB) is forced to branch 2 on track 4 and branch 3 on track 6.

## Track inventory (extracted so far)

| track | sprite (rank-1) | frame count | duration @ speed 1 |
|-------|-----------------|-------------|--------------------|
| 3     | 274             | 940         | 23.5 s             |

## What's still needed from the SWF

1. **Path data for the other 7 tracks** — same extraction process, just other sprite IDs.
2. **`Pathhit` rectangles per track** (tower-placement blocking). Inside `pathhitmc_15` (symbol 538) and per-track `track*towertest_*` MCs (symbols 558/573/580/614 etc.).
3. **Tower `hitbit` bounds** (tower-vs-tower spacing). One rectangle per tower MovieClip.
4. **Bloon `inner` hitbox bounds** (collision radius for the sim). Per-rank — they get visibly bigger.
5. **Track background images** (optional, for renderer only).

## Symbol map highlights (`symbolClass/symbols.csv`)

- `294` Bloon_1_1, `284` Bloon_1_2, `274` Bloon_1_3, `264` Bloon_1_4, `254` Bloon_1_5 (rank-1 bloons; same MovieClip = same path for all ranks on that track).
- `236/226/225` Bloon_1_6_{1,2,3}; `207/197/156` Bloon_1_8_{1,2,3}; `148/104` Bloon_1_7_{1,2}.
- `473` Bloon (base class — empty MovieClip).
- `155` red bloon `inner` body MovieClip — the child placed by every track's bloon path MC.
- `482` Pathhit, `538` pathhitmc_15.
- `558/573/580/614` track1/2/4/7 placement test MCs.
- `384` DartMonkey, `332` TackTower, `314` CannonTower, `301` IceTower, `340` SuperMonkey, `376` MonkeyBeacon, `361` BoomerangMonkey.

## Sim design implications

- One env step = one game frame (40 fps). Vectorize across envs.
- Replace `hitTestObject` with circle-circle distance check. Bloon radius ≈ from `inner` bounds; bullet radius ≈ from `hitbit` bounds.
- Seed `Math.random()` replacements with a per-env RNG. Round 51+ generation, track 4/6/8 branch assignment, and tack-shooter dispersal all depend on it.
- Path representation: `paths[track] = np.array(shape=(num_frames, 2))`, lookup with `round(progress_frame)`. Branched tracks: `paths[track][branch]`.
- Tower placement: discretize the play area into a grid; legal cells = those outside all `Pathhit` rects and outside other towers' `hitbit` rects.
