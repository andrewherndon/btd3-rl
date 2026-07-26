"""Thorough benchmark: Python sim env vs Rust sim env.

Profiles each phase: raw env throughput, per-component timing, training.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import replace

warnings.filterwarnings("ignore")

import numpy as np
import torch
torch.set_num_threads(1)

from btd.game import SimConfig
from envs.bloons_env import BloonsEnv as PyEnv
from envs.bloons_env_rs import BloonsEnv as RsEnv
from envs import actions as A
from envs.mask import build_action_mask as py_mask, compute_cell_validity as py_cv
from envs.mask_rs import build_action_mask as rs_mask, compute_cell_validity as rs_cv
from envs.observation import encode as py_encode
from envs.observation_rs import encode as rs_encode


def play_episodes(EnvCls, n_eps=20, seed=0):
    """Play episodes with only START_ROUND actions. Returns (total_time, ep_count)."""
    t0 = time.perf_counter()
    eps = 0
    for _ in range(n_eps):
        env = EnvCls(SimConfig())
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(0)  # START_ROUND
            done = terminated or truncated
            if env.sim.game_over:
                done = True
        eps += 1
    dt = time.perf_counter() - t0
    return dt, eps


print("=" * 60)
print("  BTD3 Env Backend Comparison")
print("=" * 60)

# ---- Phase 1: Raw env throughput ----
print("\n[Phase 1] Raw env throughput (START_ROUND only, no NN)")
for label, EnvCls in [("Python", PyEnv), ("Rust  ", RsEnv)]:
    times = []
    for _ in range(3):
        dt, eps = play_episodes(EnvCls, n_eps=30)
        times.append(eps / dt)
    avg = sum(times) / len(times)
    print(f"  {label}: {avg:.1f} eps/s")

# ---- Phase 2: Per-component timing (mid-game state) ----
print("\n[Phase 2] Per-component timing (round 20-ish state)")
for label, EnvCls, mask_fn, cv_fn, enc_fn in [
    ("Python", PyEnv, py_mask, py_cv, py_encode),
    ("Rust  ", RsEnv, rs_mask, rs_cv, rs_encode),
]:
    env = EnvCls(SimConfig())
    env.reset(seed=0)
    # Play to round 20
    for _ in range(8):
        env.step(0)
        if env.sim.game_over:
            break
    cell_valid = cv_fn(env.sim)

    # Time mask
    mask_us = []
    for _ in range(500):
        t0 = time.perf_counter_ns()
        m = mask_fn(env.sim, cell_valid)
        t1 = time.perf_counter_ns()
        mask_us.append((t1 - t0) / 1000)
    avg_mask = sum(mask_us) / len(mask_us)

    # Time observation
    obs_us = []
    for _ in range(500):
        t0 = time.perf_counter_ns()
        o = enc_fn(env.sim)
        t1 = time.perf_counter_ns()
        obs_us.append((t1 - t0) / 1000)
    avg_obs = sum(obs_us) / len(obs_us)

    # Time round play (step 0 = START_ROUND)
    round_ms = []
    for _ in range(10):
        env2 = EnvCls(SimConfig())
        env2.reset(seed=0)
        t0 = time.perf_counter_ns()
        env2.step(0)  # play round 1
        t1 = time.perf_counter_ns()
        round_ms.append((t1 - t0) / 1e6)
    avg_round = sum(round_ms) / len(round_ms)

    print(f"  {label}:")
    print(f"    mask build:   {avg_mask:>7.0f} μs")
    print(f"    obs encode:   {avg_obs:>7.0f} μs")
    print(f"    play round 1: {avg_round:>7.1f} ms")

# ---- Phase 3: Training throughput ----
print("\n[Phase 3] Training throughput (MaskablePPO)")
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

N_ENVS = 4
TIMESTEPS = 100_000

for label, EnvCls in [("Python", PyEnv), ("Rust  ", RsEnv)]:
    def _factory(c=EnvCls):
        return Monitor(c(SimConfig()))

    venv = DummyVecEnv([_factory for _ in range(N_ENVS)])
    model = MaskablePPO(
        "MultiInputPolicy", venv,
        n_steps=2048, batch_size=256,
        verbose=0, seed=0, n_epochs=10,
    )
    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    dt = time.time() - t0
    fps = TIMESTEPS / dt
    # Model components for step-time breakdown
    n_updates = TIMESTEPS // (2048 * N_ENVS)
    train_time = model.logger.name_to_value.get("time/train", dt)
    rollout_time = model.logger.name_to_value.get("time/rollouts", dt)
    print(f"  {label}: {fps:>7.0f} steps/s  ({TIMESTEPS} steps in {dt:.1f}s)")
    del model, venv

# ---- Phase 4: Sim-only throughput (for reference) ----
print("\n[Phase 4] Sim-only throughput (frames/sec, measured inside play_round)")
from btd.game import BloonsSim as PySim
from btd_rs import BloonsSim as RsSim
from btd_rs import SimConfig as RsConfig

for label, SimCls, cfg in [
    ("Python", PySim, SimConfig()),
    ("Rust  ", RsSim, RsConfig(difficulty="easy", paths_dir="sim-rs/paths")),
]:
    # Play 5 towers through to round 50, measure frames/time
    total_frames = 0
    t0 = time.perf_counter()
    for _ in range(20):
        sim = SimCls(cfg) if label == "Python" else SimCls(cfg)
        if label == "Rust":
            sim.place_tower("dart", 350, 100)
            sim.place_tower("dart", 200, 250)
            sim.place_tower("dart", 400, 300)
            sim.place_tower("dart", 150, 300)
            sim.place_tower("dart", 300, 350)
        else:
            sim.place_tower("dart", 350, 100)
            sim.place_tower("dart", 200, 250)
            sim.place_tower("dart", 400, 300)
            sim.place_tower("dart", 150, 300)
            sim.place_tower("dart", 300, 350)
        for r in range(50):
            if sim.game_over:
                break
            sim.start_round()
            while sim.in_round and not sim.game_over:
                sim.step()
                total_frames += 1
    dt = time.perf_counter() - t0
    fps = total_frames / dt
    print(f"  {label}: {fps/1e3:>8.0f} K fps  ({total_frames/1e3:.0f}K frames in {dt:.2f}s)")

print()
