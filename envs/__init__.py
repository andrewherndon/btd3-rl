"""Gymnasium environment wrapping BloonsSim.

Thin adapter layer: translates the sim's Python API into the observation /
action / reward contract described in RL_DESIGN.md. Depends on `btd` and
`gymnasium` only; nothing here flows back into the sim core.
"""

from .bloons_env import BloonsEnv

__all__ = ["BloonsEnv"]
