"""BloonsEnv — Gymnasium wrapper over BloonsSim.

Event-driven / decision-level MDP (see RL_DESIGN.md): the agent acts only
between rounds. Economic actions (place/upgrade/sell) advance zero frames;
START_ROUND is the temporally-extended action that runs a whole round and
returns the reward.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from btd.constants import STARTING_MONEY
from btd.game import BloonsSim, SimConfig

from . import actions as A
from .actions import Kind, cell_to_xy, decode
from .mask import build_action_mask, compute_cell_validity
from .observation import encode, make_observation_space

# Reward shaping (see RL_DESIGN.md "Reward"). Tunable knobs.
ROUND_CLEAR_BONUS = 1.0      # dense progress: survived one more round
LIFE_PENALTY = 0.1           # per life lost in a round
WIN_BONUS = 10.0             # reached the win condition (round 50)
LOSS_PENALTY = -1.0          # lives hit 0
# Per-economic-action cost, OFF by default (0.0). A blanket cost failed to
# separate "stop waste" from "keep building" (0.01 -> hoarding, 0.005 -> churn
# returns). Churn/hoarding turned out to be *symptoms* of the agent having no
# learned use for late-game money (it can't beat MOABs), so we fix that upstream
# with a curriculum, not this knob. Kept tunable, disabled.
ECON_ACTION_COST = 0.0

# Curriculum: fraction of TRAINING episodes that start mid-game (scaled money,
# fresh board) then PLAY FORWARD. Start rounds are MODERATE and winnable-from-
# fresh (8-22): the agent builds up naturally and reaches the hard rounds with a
# real defense. (Starting fresh AT a hard round like 36 was unwinnable in one
# shopping phase -> it taught helplessness/hoarding, the opposite of what we
# want.) A winnable start teaches "spend -> field more towers -> survive ->
# reward", the fix for the under-building we actually have. 0.0 = always round 1;
# eval uses 0.0 so metrics stay on honest games.
CURRICULUM_P = 0.0
CURRICULUM_MIN_ROUND = 8
CURRICULUM_MAX_ROUND = 22

# One-time reward the FIRST time each tower type is placed in an episode. A
# directed-exploration scaffold: the policy collapsed to darts and never sampled
# bombs/supers, so it never discovered that leads REQUIRE bombs. Rewards breadth
# (try each type once), not spam. OFF by default (eval uses 0.0); train.py turns
# it on. Expected to fade once real reward reinforces the towers hard rounds need.
DIVERSITY_BONUS = 0.0

# Truncation backstops (not part of the MDP; guard against pathological loops).
MAX_ECON_PER_ROUND = 60      # forced START_ROUND after this many buys/sells
MAX_STEPS = 8000             # hard episode step cap


class BloonsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        max_econ_per_round: int = MAX_ECON_PER_ROUND,
        max_steps: int = MAX_STEPS,
        econ_action_cost: float = ECON_ACTION_COST,
        curriculum_p: float = CURRICULUM_P,
        curriculum_min: int = CURRICULUM_MIN_ROUND,
        curriculum_max: int = CURRICULUM_MAX_ROUND,
        diversity_bonus: float = DIVERSITY_BONUS,
        difficulty_choices: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        # Template config; the per-episode seed is filled in at reset() for
        # domain randomization.
        self._cfg_template = config or SimConfig()
        self.max_econ_per_round = max_econ_per_round
        self.max_steps = max_steps
        self.econ_action_cost = econ_action_cost
        self.curriculum_p = curriculum_p
        self.curriculum_min = curriculum_min
        self.curriculum_max = curriculum_max
        self.diversity_bonus = diversity_bonus
        # If non-empty, each episode samples a difficulty from this pool (domain
        # randomization) so one policy learns to adapt to the economy (prices /
        # lives, which are in the obs). Empty = fixed to the template difficulty.
        self.difficulty_choices = difficulty_choices

        self.observation_space = make_observation_space()
        self.action_space = spaces.Discrete(A.N_ACTIONS)

        self.sim: BloonsSim
        self.cell_valid: np.ndarray
        self._econ_streak = 0
        self._steps = 0
        self._types_placed: set[str] = set()   # tower types placed this episode

    # ---------------------------------------------------------------- gym API

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # Explicit seed -> reproducible (eval). Otherwise draw a fresh one each
        # episode from the env RNG -> domain randomization over jitter/round-gen.
        sim_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        difficulty = self._cfg_template.difficulty
        if self.difficulty_choices:
            difficulty = str(self.np_random.choice(self.difficulty_choices))
        self.sim = BloonsSim(replace(self._cfg_template, seed=sim_seed, difficulty=difficulty))
        self.cell_valid = compute_cell_validity(self.sim)
        # Curriculum start: sometimes begin mid-game at a hard round with scaled
        # money and a fresh board, so the agent gets dense practice at rounds it
        # rarely reaches from round 1. Off (curriculum_p=0) for eval.
        if self.curriculum_p > 0.0 and self.np_random.random() < self.curriculum_p:
            r = int(self.np_random.integers(self.curriculum_min, self.curriculum_max + 1))
            self.sim.debug_set_round(r)                 # next start_round plays round r
            self.sim.money = self._accumulated_money(r)
        self._econ_streak = 0
        self._steps = 0
        self._types_placed.clear()
        return encode(self.sim), self._info()

    @staticmethod
    def _accumulated_money(round_num: int) -> int:
        """Money a never-spending player would hold entering round `round_num`:
        starting cash + all prior round-end bonuses (99 + round). Generous by
        design so the curriculum agent can afford to experiment with counters."""
        bonus = sum(99 + k for k in range(1, round_num))
        return STARTING_MONEY + bonus

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self._steps += 1
        act = decode(int(action))
        reward = 0.0
        terminated = False

        if act.kind == Kind.START_ROUND:
            reward, terminated = self._play_round()
            self._econ_streak = 0
        else:
            # Economic action: pay the small per-action cost (anti-churn), then
            # apply it. START_ROUND is never taxed — we want to encourage progress.
            reward = -self.econ_action_cost
            if act.kind == Kind.PLACE:
                x, y = cell_to_xy(act.b)
                if self.sim.place_tower(act.tower_type, x, y) != -1:
                    # Directed-exploration scaffold: bonus for the first use of
                    # each tower type this episode (breadth, not spam).
                    if act.tower_type not in self._types_placed:
                        self._types_placed.add(act.tower_type)
                        reward += self.diversity_bonus
            elif act.kind == Kind.UPGRADE:
                if act.a < len(self.sim.towers):
                    self.sim.upgrade_path(self.sim.towers[act.a].id, act.b)
            elif act.kind == Kind.SELL:
                if act.a < len(self.sim.towers):
                    self.sim.sell_tower(self.sim.towers[act.a].id)
            self._econ_streak += 1

        truncated = self._steps >= self.max_steps
        return encode(self.sim), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """Legality mask for MaskablePPO. Forces START_ROUND once the agent has
        shopped too long, so an episode can't stall in a buy/sell loop."""
        if self._econ_streak >= self.max_econ_per_round:
            m = np.zeros(A.N_ACTIONS, dtype=bool)
            m[A.START_ROUND] = True
            return m
        return build_action_mask(self.sim, self.cell_valid)

    # -------------------------------------------------------------- internals

    def _play_round(self) -> tuple[float, bool]:
        """Run one full round to completion; return (reward, terminated)."""
        lives_before = self.sim.lives
        if not self.sim.start_round():
            return 0.0, self.sim.game_over
        while self.sim.in_round and not self.sim.game_over:
            self.sim.step()

        lives_lost = lives_before - self.sim.lives
        reward = -LIFE_PENALTY * lives_lost
        if self.sim.game_over and not self.sim.won:
            return reward + LOSS_PENALTY, True          # lost
        reward += ROUND_CLEAR_BONUS                     # survived the round
        if self.sim.won:
            return reward + WIN_BONUS, True             # won the game
        return reward, False

    def _info(self) -> dict[str, Any]:
        return {
            "round": self.sim.round,
            "money": self.sim.money,
            "lives": self.sim.lives,
            "n_towers": len(self.sim.towers),
            "won": self.sim.won,
        }
