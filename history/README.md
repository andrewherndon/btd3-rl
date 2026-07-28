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

## The real problem is under-building, not diversity (2026-07-20, later)

run8 (diversity bonus, 1M) **regressed**: mean round 25.9, hoards ~$4400, still
only dart+1bomb. Every stacked intervention had made it worse than the untouched
baseline (34). Stepped back and tested the actual question with the sim: **can a
dart+bomb defense win the whole game?**

    30 darts +  8 bombs (upgraded) -> round 50, WON, lives 100
    25 darts + 25 bombs (upgraded) -> round 50, WON, lives 100

**Yes — dart+bomb, built out and upgraded, clears round 50 losslessly.** So the
"plain" strategy the agent found is *correct*: dart covers blacks (bomb-immune),
bomb covers leads (need leadbreak); together they cover every immunity. The whole
gap is that the agent builds **9-17 towers and hoards ~$4400** when the winning
defense is **~38 upgraded towers**. It's a pure **under-building / investment**
problem — not exploration, diversity, or tool choice.

Refocus:
- Reverted the scaffolds (diversity bonus -> 0, ent_coef -> 0.01). Exploration
  was never the issue.
- Reconfigured the curriculum to be **winnable**: start at MODERATE rounds (8-22,
  winnable from fresh) and **play forward**, so building out is rewarded (spend ->
  survive -> reward). The old fresh-board-at-hard-round starts were unwinnable and
  taught hoarding.
- Success metric: **money-at-end drops** and **tower-count rises** (anti-hoarding),
  pushing toward round 50. Escalation if under-building persists: self-snapshot
  curriculum (restart from the agent's own mid-game states, which carry the
  hoarded money it can then learn to spend) and/or higher gamma (0.995 -> 0.999)
  for the long-horizon "buy now, survive later" credit assignment.

### Winnable curriculum helped, but churn came back — structural fix

run9 (winnable curriculum): round 31.9 (up from 25.9), money-at-end $2169 (down
from $4400) — so the winnable curriculum DID teach more investment. But churn
returned hard (best_model: 306 sells) since nothing penalized it.

Key realization about *why* the agent can't "figure out" churn is bad: it's
**reward-neutral** (money isn't rewarded), so churn scores the same as not-churn.
Worse, it's **self-perpetuating** — churn inflates episodes to ~660 steps, which
(at gamma 0.995) discounts the distal "you wasted money -> you died" signal to
nothing, so the agent can't learn churn is bad *because it churns*. It's a
signal-poverty trap that needs an external break.

Chose the **structural** fix over a sell penalty (we'd tried a *blanket*
per-action cost -> hoarding; never a sell-only cost, but the user is wary of
reward shaping): **forbid selling a tower placed in the current shopping phase.**
A mask rule, zero reward change (no hoarding-rebound risk), and provably safe
since winning strategies sell zero towers. It breaks the intra-phase
buy->sell->rebuy loop; legit repositioning (selling older towers) is unaffected.
Bonus: killing churn shortens episodes, which should *un-poison* credit
assignment for the under-building we're also fighting.

### Best result yet, and the 25-tower plateau (run11, 1M)

The anti-churn ban had a train/inference mismatch (it lived in the env, but
watch/trace rebuild masks via `build_action_mask`), so watch still showed churn.
Fixed by centralizing BOTH rules in `build_action_mask`: the resell ban (via a new
`Tower.placed_round`, sellable only once `placed_round < current round`) and a
**same-cell placement block** (no exact-overlap stacking). Now every mask consumer
is identical.

run11 (winnable curriculum + centralized fixes, 1M) is the **best yet**: eval
round mean 35.6 / median 37 / max 39 — *past the MOAB* — building 25 towers (was
9-17), churn mostly dead (19 sells, milder "sell-old-rebuy"), using 6 bombs + a
tack, darts spaced out. The winnable curriculum clearly taught investment.

But it **plateaus at exactly 25 towers by round 25, then hoards** ($253 -> $5365
across rounds 25-38 with zero new towers) and dies at 38 sitting on $6k. It has
the money for the remaining ~13 towers (winning defense ~38) but won't spend it:
a **distal-reward problem** — the payoff of late-game towers (surviving 42-50) is
too far to propagate through the discount. Fix (one variable): **gamma 0.995 ->
0.999**, now safe because the anti-churn fix shortened episodes (~130 vs ~660
steps) so the long horizon won't blow up. Escalation if it still plateaus:
self-snapshot curriculum from the agent's own hoarding states (round 33, 25
towers, $5k) to directly teach "spend the hoard to survive".

## SOLVED (run13, 2026-07-21)

Gamma did it. run13 (winnable curriculum + centralized anti-churn/same-cell +
**gamma 0.999**), still mid-run at ~3M steps:

    win rate: 30/30 (100%)   round reached: 50.0 (all seeds)   lives left: 98.1 avg
    composition: 26 dart, 19 bomb, 5 tack, 1 spike, 3 super  (~47 towers)

**The agent solves BTD3 track 3 (easy) with a 100% win rate, near-flawless
(98/100 lives).** The winning strategy is exactly what the sim test predicted:
a dart+bomb core (darts for blacks, bombs for leads), built out to ~40+ towers,
with a few supers late that it doesn't even need.

What actually cracked it, in order:
1. **Anti-churn (structural same-phase-resell ban)** — killed the reward-neutral
   buy/sell loop that was inflating episodes to ~660 steps and poisoning credit
   assignment. This was the enabler; nothing else worked until episodes were short.
2. **Winnable curriculum** (start at moderate rounds 8-22, play forward) — taught
   investment (9 -> 25 towers) by making "spend -> survive" a reachable lesson.
3. **gamma 0.995 -> 0.999** — the final lever. With episodes now short, the higher
   discount let the round-50 payoff propagate back to early "buy more towers"
   decisions, breaking the 25-tower hoarding plateau -> it builds the full ~40.

Dead ends that taught us the most: the "MOAB wall" was a misread (round-37 MOAB is
trivial; the real walls were leads + under-building); every *reward*-shaping
anti-churn attempt (blanket/sell penalties) regressed us via hoarding; and
"diversity" was a non-problem (dart+bomb wins). The lesson throughout: measure the
actual behavior (trace.py + sim tests), don't tune on hunches.

## Overfitting + domain randomization (2026-07-21)

Zero-shot test of the easy-trained run13 on other difficulties (which change only
lives 100->50 and cost_mult 0.85->1.08; rounds 1-50 are the same bloons):

    run13 (easy-trained) on HARD:   0/30 wins, dies round 5
    run13 (easy-trained) on MEDIUM: 0/30 wins, dies round 7

Textbook **overfitting**: `cost_mult=1.08` is out-of-distribution (it only ever saw
0.85), so the policy collapses into a degenerate 2-tower churn loop and never
builds. It learned "easy BTD", not "BTD". The obs *includes* cost_mult/lives, but
the policy never trained on those values.

Fix = **domain randomization**: `BloonsEnv(difficulty_choices=...)` samples the
difficulty per episode (train.py `--difficulties easy,medium,hard`), so one policy
trains across the whole economy and learns to adapt (prices/lives are in its obs).
Eval stays fixed (`--eval-difficulty`, default hard) for honest best_model
selection. Small change, and the elegant cure for the brittleness.

## Rust sim backend + HPC cluster (2026-07-25)

Two infrastructure changes to train faster and in parallel.

**Rust sim** (`sim-rs/`, module `btd_rs`, PyO3/maturin; DeepSeek-authored). Wired
in via `envs/bloons_env_rs.py` (+ `observation_rs.py`, `mask_rs.py`) and a
`--backend {python,rust}` flag on `train.py`. Verified legitimate:
- **Behavioral parity.** run13 (trained entirely on the *Python* sim) evals
  *identically* on the Rust env: 30/30 easy, 0/30 medium/hard, near-identical
  round-reached and lives (99.9 vs 100.0 — the gap is only RNG jitter). A
  Python-trained policy transferring with matching win-rate means obs/mask/sim
  dynamics match where it counts.
- **RNG is NOT bit-exact with numpy** (the crate docstring overclaimed). numpy
  seeds PCG64 via `SeedSequence`, `.spawn()` substreams, Lemire-rejection
  `integers()`; the Rust uses `seed_from_u64` + `next_u64() % max`. Irrelevant:
  for rounds 1-50 the only RNG consumer is ±0-9px spawn jitter (noise either way).
- **Speedup is ~2.3×, not the "26×" sim-only claim.** Cluster: Python ~30 → Rust
  ~70-84 steps/s. Once the sim is cheap the **PPO update dominates** (the
  bottleneck moved), so removing sim cost only buys ~2×. Supersedes the earlier
  "numpy-vectorize the hot loop" plan — the real lever was Rust, but even that is
  capped by the network update on slow CPUs.

**HPC cluster.** 3× x86_64 nodes (i5-6500, 4 cores, 3500 MB schedulable RAM) + an
ARM Pi controller/NFS head; env + repo on `/clusterfs` (shared).
- **Scheduler packs by CORE, not memory** (`AllocMem=0` — memory isn't a
  consumable resource), so `--mem` is cosmetic and the real limit is cores. Left
  alone it crammed 4 jobs onto node01 (284 MB free, near-OOM) while node03 idled.
- Fix: **`--cpus-per-task=2`** → 2 jobs/node → even **2/2/2** spread (healthy RAM
  everywhere) + `OMP_NUM_THREADS=2` gives the PPO update a 2nd thread → **~84
  steps/s** (~1.2×). RAM-bound ceiling = **6 concurrent**.
- Built `btd_rs` via `maturin build -i <python>` + `pip install` (not
  `maturin develop`, which needs an *activated* env — unavailable under `srun`).

## Hyperparameter sweep — 12×1.5M, Rust (2026-07-26)

Grid: `gamma {0.997, 0.999} × ent_coef {0.005, 0.01, 0.02} × lr {1e-4, 3e-4}`,
seed 0, domain-randomized (easy/med/hard), eval-difficulty hard, 1.5M each, ~10 h.

Eval = 30 held-out seeds, **easy**, `best_model` (round-reached mean):

| idx | gamma | ent | lr | round |
|---|---|---|---|---|
| 001 | 0.997 | 0.005 | 3e-4 | **41.1** |
| 007 | 0.999 | 0.005 | 3e-4 | **39.5** |
| 008 | 0.999 | 0.01 | 1e-4 | 37.8 |
| 003 | 0.997 | 0.01 | 3e-4 | 37.6 |
| 002/005/009 | — | — | — | 26.0 (collapsed) |

**All 12 went 0/30 on easy** (run13 wins 100%) → **undertrained**: 1.5M is too
short (run13 needed ~3M), compounded by domain randomization + a hard-selected
checkpoint. So the table ranks *learning speed*, not converged quality.

Marginal effects (avg round over the other axes):

| axis | best value | vs rest |
|---|---|---|
| **ent_coef** | **0.005 → 36.2** | vs 0.01=31.9 / 0.02=31.5 — clear |
| lr | 3e-4 → 34.0 | vs 1e-4=32.3 — mild |
| gamma | 0.999 → 33.9 | vs 0.997=32.4 — ~tie |

**`ent_coef=0.005` is the one real signal** (+4.5 rounds) — confirms exploration
was never the bottleneck (dart+bomb is the known strategy; lower entropy → more
exploitation → faster progress). The three "stuck at 26" collapses (`min=max=26`)
span unrelated hyperparameters → **seed-0 noise**, not bad configs (why
single-seed rankings can't be trusted). Decision: carry **`ent 0.005 / lr 3e-4`**
forward (gamma unresolved, keep both), **confirm at 3-5M × seeds {0,1,2}** — the
sweep gives a direction, not a winner.

## Freeplay: the round-37 wall returns, and warm-start (2026-07-25→26)

Goal: play past round 50 (procedural 51-149) instead of winning at 50
(`--freeplay`). A fresh-trained 10M easy-freeplay run **regressed** hard:

| run | dies | money left | bombs | sells vs places |
|---|---|---|---|---|
| freeplay from scratch (10M) | ~37 | $5,000 | 3 | 2-9 vs 24-32 (churn minor) |
| + **wide curriculum 8-120** (10M) | 34-37 | $577-1,359 | 3-4 | **49-86** vs 76-111 (churn 10×'d) |
| **run13 warm-start, untuned** | **55-56** | $14-17k | — | none |

**Root cause:** enabling freeplay moved the `+10 WIN_BONUS` from a *reachable*
round 50 to an *unreachable* round 149, deleting the terminal pull that (in run13)
taught the agent to spend its hoard and break the MOAB era. Left with only
`+1/round`, it settles into a local optimum: coast to the first MOAB on saved
cash, die at 37. Curriculum (8-22) sits *below* the wall, so no practice there.

**Wide-curriculum experiment (8-120) backfired** (tested out of curiosity): high
seeds hand money but **no towers** (`debug_set_round` only moves the counter), so
round-100 starts are unwinnable-from-scratch → it flails, and thrashing gets baked
in (churn 10×'d). Curriculum has a **feasibility ceiling ~round 45**; above it,
seeds are noise that poisons the policy. Also surfaced a **latent bug**: the
log-normalized `money` obs wasn't clipped, so deep-round hoards overflow the
`[0,1]` Box — fixed with `min(…, 1.0)` in both encoders.

**Path forward = warm-start.** run13 already beats MOABs, so loading it into a
freeplay env reaches **round ~56 with zero fine-tuning** (48-49 towers, no churn);
it just hoards ~$15k past round 50 because it was trained to *win* at 50, not push
further — and that hoard-past-50 is exactly the learnable signal a fine-tune
attacks. Added `--init-from` (warm-start + lower LR) and `--curriculum-min/max`
flags. The real deep-frontier fix if needed later: **state-snapshot curriculum**
(seed round + money + *tower loadout* from good play) — the only way curriculum
reaches rounds a from-scratch build can't survive.

Code (on `rust-sim`, some uncommitted): `--backend`, `--init-from`,
`--curriculum-min/max` flags; money-obs clip; HPC `install.sh`/`bench` Rust build
+ `--backend` plumbing; sweep at 12×1.5M rust.

## The round-58 freeplay "wall" was MAX_TOWERS=64 (2026-07-27)

The 6-config 5M easy-freeplay wave (control / gamma9995 / scratch / milestone50 /
milestone10 / chain) all converged to **round 55-58, 0/30 wins** with near-zero
variance across gamma, milestone bonuses, warm-start, and from-scratch. That
flatness screamed "structural, not RL-tunable." Chasing *why* took **two wrong
turns** before the real cause — a self-imposed artifact — surfaced. Logged in full
because the failures are the lesson.

### What the cross-agent replay analysis showed (this part was right)

Replayed all six `best_model`s deterministically over 5 seeds, snapshotting
per-round tower count / money / composition / churn. Real patterns:
- **Composition converged bomb-heavy** everywhere (21-27 bombs, dart second, tack
  minor) regardless of the lever.
- **Two opposite failure modes, same wall.** *Hoarders* (control $2.8k, scratch up
  to **$17k**, milestone50) died ~57-58 sitting on cash with ~56 towers; *spenders*
  (gamma9995 $82, chain $62) spent out to **63-64 towers** and died at the *same*
  57-58.
- **Churn didn't predict the wall** — churniest (chain, sell:place 0.39) and
  cleanest (milestone50, 0.22) both landed at 57.8.

The spenders hitting 64 towers and dying alongside the 52-tower hoarders looked
decisive: not hoarding, not the slot count (they reached 64 and still died), not
tack-spam (bomb-heavy won). `rounds.py` confirmed the procedural threat ramps
*smoothly* (batches `R-43`, ceramics `R-42`/batch, +`(R-50)/15` speed) — **no
round-58 spike**, unlike the old discrete round-37 first-MOAB. Working frame at
this point: a smooth threat curve crossing a flat DPS ceiling. (The
DPS-saturation-plateau math was correct — the *height* of the plateau was the part
I got wrong, see below.)

### Wrong turn #1 — "it's composition lock-in" (infinite-money probe)

Hand-built spread, fully-upgraded **64-tower** boards with **unlimited money + full
lives**, played 51→66:

| build | result |
|---|---|
| super_max (64 maxed supers) | **survives r66, −0 lives** |
| agent_learned mix | 63-66 |
| bomb_max | dies 52-53 (blacks are bomb-immune) |
| tack_max | **dies r51, −100** |

Concluded: **a 64-tower build DOES beat r60 (super-heavy), so 58 is composition
lock-in, not a ceiling — RL is stuck in a local optimum.** Half-right (super-heavy
is strong), but the `tack_max` r51 death was a red flag I waved off instead of
chasing.

### Wrong turn #2 — "solved at 58, dart+bomb is budget-optimal" (budget probe)

Measured the **exact** lifetime budget of a clearing player (free-build a
survivable defense, reset to starting cash, play 1→N spending nothing → money held
= accrued income; build-path-independent because clearing pops ~all bloons; +$1/pop
pre-51 then 1/3, plus the 99+round bonus): **$41k entering r51, $49.6k entering
r57.** Funded realistic **64-tower** builds at $49.6k:

| build | dies (mean) |
|---|---|
| dart+bomb (fully maxed, 196 upgrades) | **57.7** |
| tack-bulk / learned / spike / boomerang | 55-56 |
| super-heavy / tack+super | **51** (budget can't upgrade them) |

Every pricier tower stole the upgrade budget; dart+bomb was the cheapest
immunity-covering pair to fully max, and it matched the agents' 57-58 *exactly*.
**Concluded: freeplay-easy is SOLVED at ~58, dart+bomb is the global budget
optimum, tacks are the worst tower, the wall is economic — declare solved, stop
training.** Confidently, comprehensively wrong.

### The correction — real-game screenshots blew it up

User produced two real BTD3 screenshots: **round 74 easy with ~100+ tack shooters**
(+ a few leadbreak rockets), and **round 121 (track 5) with ~200 supers/tacks/ice**.
The game is trivially pushable past 58. That forced the experiment I'd skipped:
**remove the tower cap.** `MAX_TOWERS=64` is an *env/mask* limit — the sim's
`place_tower` has **no count cap**. Uncapped (150 towers, spread):

| composition | @64 cap | uncapped |
|---|---|---|
| dart+bomb | 58 | **68** |
| super | 66 | **70+** |
| tack-bulk + bomb/super | 55 | **72+, zero leaks** |

And the `tack_max` r51 death, finally diagnosed: **pure tacks leak *only leads*** (6
of them; `leadbreak=False`, and a lead's escape costs 19 lives → 6 ≈ game over).
Tacks pop everything else perfectly — strong cheap AoE, **not** the worst tower. The
"tacks worst" ranking was a *double* artifact: the 64-cap **and** testing tacks with
no leadbreak support. With a few bombs, uncapped tack-bulk reaches 72+, exactly like
the screenshots.

### Root cause + the two sim gaps

The round-58 wall was **never RL**: it was **`MAX_TOWERS=64`**, an env representation
cap the agent literally cannot exceed, pinning it to ~¼ of the real game's tower
count. Every 5M sweep config was fighting a slot limit, not a policy. Also surfaced
(user's hunch, correct): **beacon drums speed-buff is unimplemented** — `game.py:470`
is a bare comment, so `beacon_rate_active` never flips and `BEACON_RATE_FACTOR=0.85`
is dead code; beacons give only a ~9.5% range nudge, no DPS multiplier.

The lesson is run13's, self-inflicted: **measure the actual thing.** I measured an
*artifact* (capped at 64, pure compositions) and over-concluded **twice** —
"composition lock-in," then "solved at 58." Real-world evidence, not another probe,
was the corrective; the `tack_max`-dies-r51 anomaly was the tell I ignored both
times. The saturation/DPS-plateau reasoning was sound — I just never noticed the
plateau *height* was set by a cap I'd imposed on myself.

Decisions:
- **Raise `MAX_TOWERS` 64 → 256** (near real-game scale). Grows the obs tower-table
  4× and adds ~576 actions (+6%; PLACE's 9,600 cells dominate so the space barely
  moves), lengthens episodes, and — changing obs/action shapes — forces a **fresh
  training run** (no warm-start from 64-slot models).
- **Implement beacon drums** (wire `beacon2` → `beacon_rate_active`), restoring the
  dead rate buff. In-scope; only monkey-storm beacon3/4 stay deferred.
- Re-train easy freeplay under the raised cap (+ beacons); expect the agent to push
  toward the real game's 70+ instead of stalling at 58.

Investigation tooling (scratchpad, not committed): `analyze_fp.py` (cross-agent
per-round snapshots), `probe_ceiling.py` (infinite-money builds), `probe_budget.py`
(exact-income measurement + realistic builds), `probe_uncapped.py` + `probe_tack.py`
(uncapped spam + leaked-rank diagnosis).

## Money-bug, snapshot curriculum, escalating reward (2026-07-28)

Analysis-heavy session (no training). Three findings + an overnight run.

**The curriculum's money formula undercounts income ~6×.** `_accumulated_money`
(the mid-game curriculum's starting cash, in both env files) sums ONLY round-end
bonuses (`99+round`) and ignores **per-pop income** (+$1/pop pre-r51) — which is
the *dominant* source. At round 51 it hands ~$6.9k vs the true ~$41k (measured with
pops). So high-round fresh curriculum starts are **cash-starved** (~6-8 affordable
towers → unwinnable): the "feasibility ceiling ~r45" is mostly this bug, NOT a
structural limit. Fixing the formula would make higher empty starts winnable *in
principle* — but the agent would then have to learn an unnatural "instant 60-tower
build in one phase," which is why the snapshot curriculum (hand it the loadout) is
cleaner. Logged as memory `curriculum-money-bug`.

**Snapshot curriculum = the real frontier lever (designed, not built).** Bank full
game states from good play (round + money + lives + **tower loadout with upgrades**)
and restart a fraction of training episodes from them, so the agent practices
*extending* a real defense instead of building from scratch at a high round. It's
the single-player substitute for self-play's automatic curriculum: iterate
(capture → train → re-capture from the improved policy) and the frontier crawls
forward. Injects **no strategy** — states are self-generated — so the agent still
discovers the tactics. Needs a sim state dump/load (both backends), a snapshot
bank, and an env `snapshot_p` reset path. Deferred (too big to build + run
unsupervised in one night). Family: Go-Explore, reverse-curriculum generation,
backplay. Honest ceiling: diminishing returns ~r65-75, past which success depends
on the *whole trajectory* (the foundation), which no local start-state trick fixes —
that needs AlphaZero-style search + a learned value function.

**Escalating per-round reward (implemented).** New `--frontier-bonus b`: adds
`b*(round-50)` to the clear reward past round 50 (train-only; eval stays honest at
0), so deeper frontier rounds are worth progressively more. Replaces the dead
freeplay terminal (WIN_BONUS at unreachable r149) with a dense, reachable pull.
Fixes **credit assignment, NOT exploration** — makes depth desirable, not reachable
— so it's *complementary* to the snapshot curriculum, not a substitute. Caveats: it
fights the discount (needs growth to outpace γ^t — fine at γ=0.999), can induce
risk-seeking, and is non-potential-based shaping (can shift the optimum, here toward
depth, which is aligned). Threaded through both env backends + `train.py` like the
milestone flags.

**Overnight run** (`agent/models/fp_frontier256/`, Mac, 10M, rust): fresh
from-scratch (the 64→256 cap changed obs/action shapes, so NO warm-start from run13
etc.), easy freeplay, winnable curriculum 8-22, `ent 0.005 / lr 3e-4 / gamma 0.999`
(run13 + sweep), `--frontier-bonus 0.3`, beacons live. Tests whether cap-raise +
frontier-pull move the old ~58 wall. Expect possible undertraining (fresh 256-cap
freeplay in 10M is ambitious — run13 needed ~3M just for base round-50 at the 64
cap); best_model saved on honest eval.
