# sim — BTD3 simulator

Python reimplementation of the BTD3 game loop intended to back an RL training
environment. The sim is faithful to the decompiled game logic (see
`../NOTES.md`) but replaces Flash-specific machinery (display-list rendering,
`hitTestObject`, `ENTER_FRAME` listeners) with pure data structures and
NumPy-friendly math.

## Layout

```
sim/
├── README.md            (this file)
├── extract_path.py      (SWF XML -> per-track path arrays)
├── visualize_path.py    (matplotlib plot of an extracted path)
├── smoke_test.py        (CLI sanity check; one round of track 3)
├── play.py              (interactive playtest entry point)
├── render.py            (pygame view onto a BloonsSim)
├── paths/
│   ├── track_N.npy      (shape (frames, 2), dtype float64, stage pixels)
│   ├── track_N.json     (same data + metadata)
│   └── track_N.png      (sanity-check plot)
└── btd/                 (sim core; nothing here imports pygame)
    ├── __init__.py
    ├── constants.py     (mirrors BloonsTD.as static fields)
    ├── bloon.py
    ├── tower.py
    ├── bullet.py
    └── game.py          (BloonsSim — orchestrator)
```

`btd/` is pure logic and depends only on numpy. `render.py` and `play.py`
add the pygame layer for visualization and interactive playtest. The
Gymnasium env wrapper, when it lands, will sit alongside `play.py` and
share the same `BloonsSim` instance.

## Data pipeline

```
Bloons_Tower_Defense_3.swf
        │  (JPEXS "Export SWF XML", one-time, manual)
        ▼
Decompiled-Limited/Bloons_Tower_Defense_3.xml
        │  extract_path.py --track N
        ▼
sim/paths/track_N.{npy,json}
        │  loaded once at sim init
        ▼
BloonsSim
```

The XML and the decompiled scripts are not committed (see top-level
`.gitignore`); they're treated as inputs that anyone can regenerate from the
original SWF.

## Conventions

- **Coordinate system**: stage pixels, origin top-left, +x right, +y down
  (Flash convention). Stage is 640x480.
- **Time unit**: one game frame = 1 / 40 s. Every duration in the code is
  in frames, not seconds. `attackRate=33` means 33 frames between shots.
- **Path coordinates**: stored in stage space, i.e. the per-track
  `_loc11_/_loc12_` offset from `BloonsTD.NewBloon` is baked in. Add only
  the per-bloon spawn jitter (uniform 0..9 px on each axis) at instantiation.
- **Path progress**: a bloon's position on its track is parameterised by a
  scalar `frame: float` advanced by `frame += speed` each tick. Stage
  position = `paths[track][round(frame)] + jitter`. When `round(frame)`
  exceeds the path length, the bloon has escaped.
- **RNG**: a per-environment `numpy.random.Generator` replaces every
  `Math.random()` call. Things that consume randomness: round-51+ rank
  selection, track-4/6/8 branch assignment, spawn jitter, tack-shooter
  sub-projectile dispersal, snap-freeze proc, post-round-50 pop rewards.
- **Collision**: circle-vs-circle on bloon center against bullet center.
  This diverges from Flash's `hitTestObject` (axis-aligned bbox of rendered
  sprite) but is much faster and gameplay-equivalent within a pixel or two.
  Radii are constants per bloon rank and per bullet type.
- **Floats are deterministic per seed**. Avoid wall-clock-derived state.
- **No globals**. Every piece of state lives on a `BloonsSim` instance so
  multiple sims can run in parallel processes without contention.

## Tools

### extract_path.py

Streams the JPEXS-exported SWF XML, locates the `DefineSpriteTag` for a
given track's rank-1 bloon, walks its `PlaceObject*` / `ShowFrameTag` tags,
and emits a `(num_frames, 2)` NumPy array of pixel coordinates.

```
python extract_path.py \
  --xml ../Decompiled-Limited/Bloons_Tower_Defense_3.xml \
  --symbols ../Decompiled-Limited/symbolClass/symbols.csv \
  --track 3 \
  --out-dir paths
```

Flags:
- `--branch {1,2,3}` for tracks 4/6/8 (default 1).
- `--local-coords` skips the per-track stage offset and saves the raw
  MovieClip-local coords. Default is to apply the offset so saved paths
  are directly usable in stage space.

The `TRACK_OFFSETS` table in this script is the source of truth for
per-track stage offsets, derived from `BloonsTD.NewBloon` `case <n>` blocks.

### visualize_path.py

Plots a saved path over a 640x480 stage outline. Pure sanity-check tool —
not used by the sim itself.

```
python visualize_path.py --npy paths/track_3.npy --out paths/track_3.png
```

## RL environment target

The simulator will be wrapped in a [Gymnasium](https://gymnasium.farama.org/)
`Env`. Gymnasium is the maintained successor to OpenAI Gym; same API surface
(`reset`, `step`, `observation_space`, `action_space`). Working assumptions
below — these will tighten as the sim solidifies.

- **One env step = one game frame** (initial choice; may bump to frameskip
  later for training throughput).
- **Episode** = one full game on a fixed track + difficulty. Terminates on
  `lives <= 0` (loss) or `round > 50` (win). Truncation only if a wall-clock
  step budget is hit.
- **Action space**: discrete, of the form
  `(NOOP | START_ROUND | PLACE tower_type at grid_cell | UPGRADE tower_id path | SELL tower_id)`.
  The placement grid will be coarse (e.g. 16 px cells -> 40x30 = 1200 cells)
  with an illegal-action mask derived from `Pathhit` blockers and existing
  towers.
- **Observation**: a structured dict (until we see what works) covering
  money, lives, round, time-in-round, a bloon table, and a tower table.
  Image-based observations are out of scope for the sim and the
  responsibility of the renderer.
- **Reward shaping** is deferred. For now reward = pops + money rewards;
  this is the simplest dense signal that aligns with surviving longer.

## Tooling beyond the gym

Two non-RL entry points matter for development and analysis:

- **Headless playback** of a sequence of actions against the sim, returning
  full per-frame state trace. Used for unit tests, replays, and feeding the
  renderer.
- **Renderer** (separate module, pygame planned). Consumes a state trace
  (or live `BloonsSim`) and draws a pygame window. Not in the hot training
  path. Used to spot-check agent behavior on rollouts.

Both rely on the same `BloonsSim` core; the gym `Env` is a thin wrapper.

## Status

- [x] Track-3 path extracted (940 frames, single path).
- [x] `BloonsSim` core: tick loop, bloons walking, towers acquiring & shooting,
      bullets flying, circle-vs-circle collision, pops with child spawning,
      escape damage, round end with grace + emergency timeout.
- [x] `smoke_test.py` runs round 1 of track 3 end-to-end (14 reds vs 2 darts;
      14 pops, 0 escaped, +$114, finishes at frame ~737).
- [x] Pygame renderer + interactive playtest (`play.py`): place dart towers
      with the mouse, SPACE starts the next round, ESC deselects, R resets,
      Q quits. Placeholder graphics — colored circles for bloons sized by
      rank, squares for towers, polyline for path.
- [x] Bomb tower (Cannon) + two-stage bullet (flies until first hit, then
      stops with a larger explosion radius). icebreak + leadbreak flags
      propagate from tower -> bullet.
- [x] Immunity framework: lead clink for non-leadbreak bullets, black bomb-
      immunity, frozen non-icebreak clink, ranks 9/10 multi-hit via
      `hits_remaining`. Verified with case checks.
- [ ] Tower types beyond dart and bomb: tack, ice, boomerang, spikeopult,
      super, beacon, monkey storm; placed items (spikes, glue, pineapple).
- [ ] Tower upgrades (4 per type, the `GetUpgrade` switch).
- [ ] Beacon range/rate buffs.
- [x] Full round table (`btd/rounds.py`): hardcoded rounds 1-50 from
      `BloonsTD.BuildLevels`, procedural rounds 51-149. `SimConfig.freeplay`
      gates play past round 50.
- [ ] Per-rank bloon hitbox radii from extracted `inner` bounds (currently
      placeholders in `constants.BLOON_RADIUS`).
- [ ] Tower placement legality from `Pathhit` rectangles.
- [ ] Gymnasium `Env` wrapper.
- [ ] Pygame renderer.
- [ ] Other tracks' paths.

## References

- `../NOTES.md` — decompile findings (game logic, balance constants).
- `../Decompiled-Limited/scripts/BloonsTD.as` — main game class (3218 lines).
- `../Decompiled-Limited/scripts/Bloon.as`, `Tower.as`, `Bullet.as` —
  load-bearing classes.
