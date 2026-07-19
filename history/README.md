# BTD3 RL — project history

Log of the RL effort: a MaskablePPO agent trained against the BTD3 sim to reach
round 50 (track 3, difficulty easy). Full design rationale lives in
`../RL_DESIGN.md`. This file is the "what happened and why" trail.

Archived model checkpoints are under `run*/best_model.zip`. Each was trained
under a specific code state — to replay one, `git checkout` the commit noted in
its `info.txt` (obs/action shapes change across runs, so a model only loads
against its own code).

## Design (one-liner each; see RL_DESIGN.md for the reasoning)

- **MDP**: event-driven / decision-level. Agent acts only between rounds;
  `START_ROUND` fast-forwards a whole round. No mid-round micro.
- **Observation**: economy + next-round threat preview + padded tower table
  (+ mask). log-compressed money, one-hot tower types.
- **Action**: flat `Discrete(9661)` (`START_ROUND | PLACE | UPGRADE | SELL`) with
  a legality mask (mask, don't penalize illegal moves).
- **Reward**: +1 / round cleared, −0.1 / life lost, +10 win. Money NEVER
  rewarded (instrumental — let the value function price it).
- **Algorithm**: MaskablePPO (sb3-contrib), MultiInputPolicy.

## Compute findings (M4, 10 cores)

- ~500 sim-steps/s single core. The workload is env-bound (tiny net, pure-Python
  sim), not GPU-bound.
- `SubprocVecEnv` gave only ~1.15× (measured), not the expected ~5×: the main
  process serializes N Dict-obs + N 9661-masks every step → Amdahl-capped IPC.
- Conclusion: **don't outsource** — GPU sits idle, and a many-core CPU box hits
  the same serialization wall. The only real speed lever is numpy-vectorizing the
  sim hot loop (`_tick_collisions`, etc.).

## Runs

Eval = 30 held-out seeds (base 1_000_000), deterministic policy, easy.

| run | steps | dates | round reached (mean/med/min/max) | win% | reward | code |
|---|---|---|---|---|---|---|
| run1_1M | 1M | 2026-07-18 | 34.3 / 33 / 33 / 40 | 0% | 22.3 | 8b5b753 |
| run2_18M | ~18M | 07-18→07-19 | 37.1 / 37 / 37 / 38 | 0% | 25.1 | 8b5b753 |

## Findings & observed behaviors

- **Not sample-limited.** 18× more steps moved the wall only 34→37;
  `explained_variance 0.985` (critic converged). It's a design/capability wall,
  not a compute one.
- **MOAB wall.** run2 dies at round 37 = the first MOAB (rank 10, 130 hits) with
  near-zero variance (min 37). Cheap-tower spam can't burst a MOAB; it needs
  concentrated damage (bombs / upgrades / dense chokepoint).
- **Buy/sell money churn.** At rounds ~30+, the agent repeatedly buys and sells
  the cheapest dart, bleeding money to the ~80% sell-back, then starts the round.
  Inflates episode length (`ep_len_mean ~412`). Cause: money is (correctly) not
  rewarded, so churn is reward-neutral → no gradient against it; likely amplified
  by the tower blind spot below.
- **MAX_TOWERS blind spot (representation bug).** `MAX_TOWERS=20`, but the agent
  placed 31–33 towers. Towers past 20 fire in the sim (visible in the renderer)
  but are absent from the observation and unaddressable by upgrade/sell — so the
  agent is blind to ~1/3 of its own board. Violates the "full tower state =
  Markov" design.

## Decisions

- **Separate the bug from the strategy.** Blind spot = bug (fix it). A hard tower
  limit = a strategy choice (not ours to make — a viable human strategy is 30+
  tack shooters). So: raise `MAX_TOWERS` well above what binds (→ 64), removing
  the blind spot, and keep a placement guard only as a safety net. This changes
  obs/action shapes → a fresh run; run1/run2 are archived here as the pre-change
  baseline.
- **Churn fix deferred.** Try the representation fix first (it may remove the
  churn). If it persists, add a tiny per-decision step cost (efficiency nudge) —
  never a money penalty (that re-imposes strategy and risks cowardly policies).
