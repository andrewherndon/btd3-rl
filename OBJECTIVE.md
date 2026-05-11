# Objective

Build a fast, deterministic Python simulation of *Bloons Tower Defense 3* so that an RL agent can later be trained against it. The user is learning RL and wants to do that part themselves; the work in this repo is the **simulation environment**, not the agent.

## Hard rule for the assistant

**Do not generate RL code, training scripts, agent architectures, or RL strategy advice unless the user explicitly asks.** That includes: gym wrappers can be designed/discussed at the API level when relevant, but agents, reward shaping recipes, training loops, network designs, and algorithm choice are off-limits unless asked. The deliverable is a faithful, efficient sim with good tooling around it.

## Inputs

- `Decompiled-Limited/` (gitignored) — JPEXS output from `Bloons_Tower_Defense_3.swf`. Contains the ActionScript source (`scripts/BloonsTD.as` is the core), `symbolClass/symbols.csv` (character-id ↔ class-name map), and the full `Bloons_Tower_Defense_3.xml` (every SWF tag — used to extract path geometry).
- Original game is Flash AS3, runs at 40 fps, 640×480 stage. Bloon paths are baked into per-track MovieClip timelines as keyframed transforms — they don't exist as arrays in the AS code.

## Current scope (working)

- Track 3 only (other tracks deferred; the API doesn't bake in this limitation).
- Tower types: dart, bomb. Bullet types: dart, bomb (two-stage).
- Immunities: lead (rank 7), black (rank 5), frozen, MOAB / ceramic multi-hit.
- Full round table 1-50 hardcoded from `BuildLevels`, 51-149 procedural. `freeplay` flag gates 51+.
- Pygame renderer with placeholder graphics + interactive playtest (`sim/play.py`).

## Where information lives

- `NOTES.md` (this dir) — game-specific facts learned from the decompile: rank stats, immunities, round/economy formulas, per-track stage offsets, symbol IDs, sources for things-still-needed-from-the-SWF.
- `sim/README.md` — implementation notes: directory layout, data pipeline, coordinate conventions, tool docs, RL-env *target API* (Gymnasium dict obs / discrete actions, not implemented yet), status checklist of what's done vs. open.
- `sim/btd/` — the sim core. Pure Python + numpy, no pygame dependency.
- `sim/render.py`, `sim/play.py` — pygame layer for visualization / manual playtesting.
- Git history — short one-line commits, each is a logical chunk (gitignore, notes, path extraction, sim core, renderer, round table, bomb tower).

If you (the assistant) are starting fresh: read this file, then `NOTES.md`, then `sim/README.md`. That's the full picture.
