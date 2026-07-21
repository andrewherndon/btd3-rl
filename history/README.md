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

## The churn -> hoarding -> exploration arc (2026-07-19)

After `MAX_TOWERS=64` (run3, 1M): churn/over-stacking persisted. Chased it with a
blanket per-economic-action cost and learned it's a **blunt instrument that can't
win**:

| change | result |
|---|---|
| no penalty (run3, MAX_TOWERS=64) | churn + 15-tower exact-stacks at the best cell |
| econ cost 0.01 (run4, 1M) | churn gone, but **hoards ~$5049**, round-reached **28.6** (regressed from 34) |
| econ cost 0.005 (run5, 5M) | **churn returns**, round ~33 |

The insight (from watching + the money metric): **churn and hoarding are the same
symptom** — the agent has *no learned productive use for late-game money* because
it never discovered how to beat a MOAB (rank 10, 130 hits, first at round 37). No
penalty -> it wastes the surplus (churn); with penalty -> it sits on it (hoard).
Both starve it of the firepower to break the **round-37 MOAB wall**.

Root cause = a **hard-exploration + long-horizon-credit-assignment** problem
(shape of Montezuma's Revenge): the agent is stuck in a cheap-dart-spam local
optimum, rarely reaches MOABs (so gets ~no experience fighting them), and won't
risk expensive/different towers (bomb, super) whose payoff is distant.

Decisions:
- **Reverted the econ penalty** (symptom-whacking; every value regressed us).
- **Added a curriculum** (`envs/bloons_env.py`): a fraction of *training* episodes
  start mid-game at a hard round (default rounds 20-38) with income-scaled money
  and a fresh board, so the agent gets *dense* MOAB-era practice. Eval stays on
  full round-1 games (`curriculum_p=0`) so metrics remain honest.
- Note on reward shaping: a sell-only penalty was considered but declined — it
  still shapes strategy (biases against selling), and the real fix is upstream
  (let the agent *discover* MOAB counters via curriculum), not more reward hacks.

## It was never the MOAB — it's leads + a tower monoculture (2026-07-20)

Built `agent/trace.py` (headless action-log replay) to watch the agent directly.
On a full game (run6, 1M) it: reached round 33, took PLACE=155 / SELL=127 /
UPGRADE=62 -> only ~28 towers on board, and placed **only darts (153 place
actions) + 2 bombs, nothing else**. So: a **dart monoculture** (rarely a tack),
and the big place-count is **churn** (buy/sell the same spot), not board size.

Then tested defenses directly against the round table:
- **Round 37's MOAB is trivial** — "30 darts + 8 bombs" or "6 supers + 6 bombs"
  lose **0 lives**. Our "MOAB wall" framing was a misread.
- The real walls: **leads** (rank 7, rounds 36/39/41) which **only bombs pop**
  (leadbreak), and **dense rounds** (e.g. 60 blacks at r33) that a money-starved
  dart-only defense can't out-DPS. (Off-by-one in the first test hid this: the
  agent dies ~33 to firepower/lead limits, not to the MOAB.)

So the game IS winnable with the available towers; the blocker is a **tower-type
exploration collapse** — darts give immediate early reward, the policy collapses
to darts before trying bombs/supers, and then it structurally can't pass leads.
Churn is downstream (no learned use for money).

Tier-1 fix (this iteration):
- Curriculum range widened to rounds **20-42** so it covers the lead rounds where
  dart-only *must* fail (forcing gradient pressure toward alternatives).
- `ent_coef` **0.01 -> 0.03** to slow the collapse so bombs/supers get sampled.
- Success metric shifts from round-reached to **tower diversity** (check via
  `trace.py`: do bombs/supers actually appear?).
- Tier-2 if still monoculture: directed exploration (intrinsic diversity bonus —
  defensible since leads *hard-require* bombs) or a demonstration warm-start.

Tools: `agent/trace.py` replays a model headless and prints the action log +
counts (sells vs places = churn; tower-type distribution = diversity).

Update — Tier-1 failed, went to Tier-2 (diversity bonus). run7 (1M, curriculum +
ent_coef 0.03) did NOT break the monoculture: `best_model` @seed0 = 156 dart /
4 bomb places, 143 churn-sells, round 29; `model` (final) = 7 dart / 3 bomb,
hoards $3.3k, round 26. (Reminder: `best_model` = best-eval checkpoint, `model` =
final — different policies; here they show the two symptoms, churn vs hoard.)
So added **Tier-2 directed exploration**: a **diversity bonus** (+0.3 the first
time each tower type is placed in an episode; `--diversity-bonus`, training only,
eval 0.0). Rewards breadth (try each type once), not spam, so the agent samples
bombs/supers and can discover that leads require bombs. Success = tower-type
spread in `trace.py`, not round-reached.
