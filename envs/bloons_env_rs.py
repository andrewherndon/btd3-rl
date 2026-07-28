"""BloonsEnv — Gymnasium wrapper using the Rust btd_rs simulator.

Same event-driven MDP as bloons_env.py, backed by the Rust sim for
~26× faster simulation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from btd.constants import STARTING_MONEY
from btd_rs import BloonsSim, SimConfig as RsConfig

from . import actions as A
from .actions import Kind, cell_to_xy, decode
from .mask_rs import build_action_mask, compute_cell_validity
from .observation_rs import encode, make_observation_space

# Reward shaping (same as Python env).
ROUND_CLEAR_BONUS = 1.0
LIFE_PENALTY = 0.1
WIN_BONUS = 10.0
LOSS_PENALTY = -1.0
ECON_ACTION_COST = 0.0

CURRICULUM_P = 0.0
CURRICULUM_MIN_ROUND = 8
CURRICULUM_MAX_ROUND = 22

DIVERSITY_BONUS = 0.0
MAX_ECON_PER_ROUND = 60
MAX_STEPS = 8000

RS_PATHS_DEFAULT = "sim-rs/paths"


class BloonsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[Any] = None,
        max_econ_per_round: int = MAX_ECON_PER_ROUND,
        max_steps: int = MAX_STEPS,
        econ_action_cost: float = ECON_ACTION_COST,
        curriculum_p: float = CURRICULUM_P,
        curriculum_min: int = CURRICULUM_MIN_ROUND,
        curriculum_max: int = CURRICULUM_MAX_ROUND,
        diversity_bonus: float = DIVERSITY_BONUS,
        difficulty_choices: tuple[str, ...] = (),
        milestone_bonus: float = 0.0,
        milestone_every: int = 0,
        frontier_bonus: float = 0.0,
    ) -> None:
        super().__init__()
        self._cfg_template = config or None  # We'll construct RsConfig from this
        # Extract common fields from the Python SimConfig if provided.
        if config is not None:
            self._track = config.track
            self._difficulty = config.difficulty
            self._freeplay = config.freeplay
        else:
            self._track = 3
            self._difficulty = "easy"
            self._freeplay = False
        self.max_econ_per_round = max_econ_per_round
        self.max_steps = max_steps
        self.econ_action_cost = econ_action_cost
        self.curriculum_p = curriculum_p
        self.curriculum_min = curriculum_min
        self.curriculum_max = curriculum_max
        self.diversity_bonus = diversity_bonus
        self.difficulty_choices = difficulty_choices
        self.milestone_bonus = milestone_bonus
        self.milestone_every = milestone_every
        self.frontier_bonus = frontier_bonus

        self.observation_space = make_observation_space()
        self.action_space = spaces.Discrete(A.N_ACTIONS)

        self.sim: BloonsSim = None  # type: ignore
        self.cell_valid: np.ndarray
        self._econ_streak = 0
        self._steps = 0
        self._types_placed: set[str] = set()

    def _make_rs_config(self, seed: int, difficulty: str) -> RsConfig:
        return RsConfig(
            track=self._track,
            difficulty=difficulty,
            seed=seed,
            freeplay=self._freeplay,
            paths_dir=RS_PATHS_DEFAULT,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        sim_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        difficulty = self._difficulty
        if self.difficulty_choices:
            difficulty = str(self.np_random.choice(self.difficulty_choices))
        self.sim = BloonsSim(self._make_rs_config(sim_seed, difficulty))
        self.cell_valid = compute_cell_validity(self.sim)

        if self.curriculum_p > 0.0 and self.np_random.random() < self.curriculum_p:
            r = int(self.np_random.integers(self.curriculum_min, self.curriculum_max + 1))
            self.sim.debug_set_round(r)
            self.sim.debug_add_money(self._accumulated_money(r))

        self._econ_streak = 0
        self._steps = 0
        self._types_placed.clear()
        return encode(self.sim), self._info()

    @staticmethod
    def _accumulated_money(round_num: int) -> int:
        return STARTING_MONEY + sum(99 + k for k in range(1, round_num))

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self._steps += 1
        act = decode(int(action))
        reward = 0.0
        terminated = False

        if act.kind == Kind.START_ROUND:
            reward, terminated = self._play_round()
            self._econ_streak = 0
        else:
            reward = -self.econ_action_cost
            if act.kind == Kind.PLACE:
                x, y = cell_to_xy(act.b)
                if self.sim.place_tower(act.tower_type, x, y) != -1:
                    if act.tower_type not in self._types_placed:
                        self._types_placed.add(act.tower_type)
                        reward += self.diversity_bonus
            elif act.kind == Kind.UPGRADE:
                towers = self.sim.get_towers()
                if act.a < len(towers):
                    self.sim.upgrade_path(towers[act.a]["id"], act.b)
            elif act.kind == Kind.SELL:
                towers = self.sim.get_towers()
                if act.a < len(towers):
                    self.sim.sell_tower(towers[act.a]["id"])
            self._econ_streak += 1

        truncated = self._steps >= self.max_steps
        return encode(self.sim), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        if self._econ_streak >= self.max_econ_per_round:
            m = np.zeros(A.N_ACTIONS, dtype=bool)
            m[A.START_ROUND] = True
            return m
        return build_action_mask(self.sim, self.cell_valid)

    def _play_round(self) -> tuple[float, bool]:
        lives_before = self.sim.lives
        if not self.sim.start_round():
            return 0.0, self.sim.game_over
        while self.sim.in_round and not self.sim.game_over:
            self.sim.step()

        lives_lost = lives_before - self.sim.lives
        reward = -LIFE_PENALTY * lives_lost
        if self.sim.game_over and not self.sim.won:
            return reward + LOSS_PENALTY, True
        reward += ROUND_CLEAR_BONUS
        reward += self._milestone_reward()
        reward += self._frontier_reward()
        if self.sim.won:
            return reward + WIN_BONUS, True
        return reward, False

    def _milestone_reward(self) -> float:
        """Freeplay stepping-stone bonus: restores a reachable pull past round 50.
        milestone_every==0 -> once at round 50; >0 -> every N rounds."""
        if not self._freeplay or self.milestone_bonus <= 0.0:
            return 0.0
        r = self.sim.round
        if (self.milestone_every and r % self.milestone_every == 0) or \
           (not self.milestone_every and r == 50):
            return self.milestone_bonus
        return 0.0

    def _frontier_reward(self) -> float:
        """Escalating freeplay pull: +frontier_bonus per round survived past 50.
        Replaces the dead terminal (WIN_BONUS at round 149) with a dense,
        reachable gradient toward depth. Train-only (eval uses 0)."""
        if not self._freeplay or self.frontier_bonus <= 0.0:
            return 0.0
        return self.frontier_bonus * max(0, self.sim.round - 50)

    def _info(self) -> dict[str, Any]:
        return {
            "round": self.sim.round,
            "money": self.sim.money,
            "lives": self.sim.lives,
            "n_towers": self.sim.n_towers,
            "won": self.sim.won,
        }
