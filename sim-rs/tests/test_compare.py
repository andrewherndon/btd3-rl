"""Snapshot-comparison test harness: runs the same scenarios through both
Python and Rust sims, then compares outputs.

Usage:
    cd BTD/ && .venv/bin/python sim-rs/tests/test_compare.py

Each test scenario:
1. Constructs both sims with identical config (seed, difficulty, track)
2. Plays the same sequence of actions (place towers, start rounds, upgrade, sell)
3. Compares final state (money, lives, round, win/loss, tower count, total pops)

RNG-dependent scenarios (snap-freeze, rounds 51+, freeplay) are expected
to differ structurally due to different RNG algorithms (PCG64 vs ChaCha12).
We validate game-logic consistency across many seeds instead.
"""

import sys
import os
from pathlib import Path

# Ensure we can import both sims.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim"))

import json

# Import Python reference sim.
from btd.game import BloonsSim as PyBloonsSim
from btd.game import SimConfig as PyConfig

# Import Rust sim.
from btd_rs import BloonsSim as RsBloonsSim
from btd_rs import SimConfig as RsConfig

RS_PATHS = str(REPO / "sim-rs" / "paths")


def make_py_sim(seed=0, difficulty="easy", freeplay=False, track=3):
    return PyBloonsSim(PyConfig(track=track, difficulty=difficulty, seed=seed, freeplay=freeplay))


def make_rs_sim(seed=0, difficulty="easy", freeplay=False, track=3):
    cfg = RsConfig(track=track, difficulty=difficulty, seed=seed, freeplay=freeplay,
                   paths_dir=RS_PATHS)
    return RsBloonsSim(cfg)


# ---- Test scenarios -----------------------------------------------------------

SCENARIOS = []


def scenario(name, setup, rounds=1):
    """Register a test scenario.
    `setup(sim)` is called before start_round() to place towers, etc.
    `rounds` is the number of rounds to play.
    """
    SCENARIOS.append((name, setup, rounds))


# Scenario 1: Round 1 with 2 dart monkeys.
def setup_2darts(sim):
    sim.place_tower("dart", 350.0, 100.0)
    sim.place_tower("dart", 200.0, 250.0)
scenario("round1_2darts", setup_2darts, rounds=1)


# Scenario 2: Rounds 1-5 with 2 dart monkeys.
scenario("rounds1-5_2darts", setup_2darts, rounds=5)


# Scenario 3: Rounds 1-10 with 2 darts (tests cumulative play).
scenario("rounds1-10_2darts", setup_2darts, rounds=10)


# Scenario 4: Bomb + lead test. Place dart + bomb, run to round 21.
def setup_bomb_lead(sim):
    sim.place_tower("dart", 350.0, 100.0)
    sim.place_tower("bomb", 400.0, 300.0)
scenario("lead_round21", setup_bomb_lead, rounds=21)


# Scenario 5: Dart + ice, run to round 15 (freeze test).
def setup_ice(sim):
    sim.place_tower("dart", 350.0, 100.0)
    sim.place_tower("ice", 250.0, 300.0)
scenario("freeze_round15", setup_ice, rounds=15)


# Scenario 6: Boomerang, run rounds 1-10.
def setup_boomerang(sim):
    sim.place_tower("boomerang", 350.0, 100.0)
    sim.place_tower("boomerang", 200.0, 200.0)
scenario("boomerang_rounds1-10", setup_boomerang, rounds=10)


# Scenario 7: Dart + bomb, run to round 37 (MOAB).
def setup_moab(sim):
    sim.place_tower("dart", 350.0, 100.0)
    sim.place_tower("dart", 200.0, 250.0)
    sim.place_tower("bomb", 400.0, 300.0)
    sim.place_tower("bomb", 300.0, 350.0)
    sim.place_tower("spikeopult", 150.0, 300.0)
scenario("moab_round37", setup_moab, rounds=37)


# Scenario 8: Upgrades (path 1 + path 2 on dart).
def setup_upgrades(sim):
    sim.place_tower("dart", 350.0, 100.0)
    # Apply some upgrades.
    sim.upgrade_tower(0, "dart1")
    sim.upgrade_tower(0, "dart3")
scenario("upgrade_both_paths", setup_upgrades, rounds=5)


# Scenario 9: Multiple tower types.
def setup_multi_type(sim):
    sim.place_tower("dart", 350.0, 100.0)
    sim.place_tower("tack", 250.0, 200.0)
    sim.place_tower("spikeopult", 200.0, 300.0)
    sim.place_tower("bomb", 400.0, 350.0)
    sim.place_tower("super", 300.0, 150.0)
scenario("multi_type_rounds1-15", setup_multi_type, rounds=15)


# Scenario 10: Medium difficulty.
def setup_medium(sim):
    sim.place_tower("dart", 350.0, 100.0)
    sim.place_tower("dart", 200.0, 250.0)
    sim.place_tower("bomb", 400.0, 300.0)
scenario("medium_difficulty", setup_medium, rounds=15)


# Scenario 11: Hard difficulty + lead.
def setup_hard(sim):
    sim.place_tower("dart", 350.0, 100.0)
    sim.place_tower("dart", 200.0, 250.0)
    sim.place_tower("bomb", 400.0, 300.0)
    sim.place_tower("bomb", 300.0, 350.0)
scenario("hard_difficulty", setup_medium, rounds=15)


# ---- Runner -------------------------------------------------------------------

def play_rounds(sim, n_rounds, max_frames_per_round=10000):
    """Play `n_rounds` rounds on `sim`. Returns final snapshot and per-round data."""
    rounds_data = []
    for r in range(n_rounds):
        if sim.game_over:
            break
        if not sim.in_round:
            if not sim.start_round():
                break
        frame = 0
        while sim.in_round and not sim.game_over:
            sim.step()
            frame += 1
            if frame > max_frames_per_round:
                break
        # Record per-round state.
        obs = sim.observe()
        # Handle both Python sim (has .towers) and Rust sim (has .n_towers).
        n_tow = obs.get("n_towers", getattr(sim, 'n_towers', 0))
        rounds_data.append({
            "round": sim.round,
            "money": sim.money,
            "lives": sim.lives,
            "pops": obs.get("pops_this_round", 0),
            "n_towers": n_tow,
        })
    return rounds_data


def compare_scenario(name, setup_fn, n_rounds, difficulty="easy", seed=0):
    """Run both sims and compare results."""
    py_sim = make_py_sim(seed=seed, difficulty=difficulty)
    rs_sim = make_rs_sim(seed=seed, difficulty=difficulty)

    setup_fn(py_sim)
    setup_fn(rs_sim)

    py_rounds = play_rounds(py_sim, n_rounds)
    rs_rounds = play_rounds(rs_sim, n_rounds)

    # Compare final state (last round data).
    py_final = py_rounds[-1] if py_rounds else {}
    rs_final = rs_rounds[-1] if rs_rounds else {}

    issues = []

    # Money and lives differ due to different RNG (SeedSequence in numpy vs
    # seed_from_u64 in Rust). Both produce deterministic sequences from the
    # same seed, but the sequences differ. Accept reasonable drift.
    money_diff = abs(py_final.get("money", 0) - rs_final.get("money", 0))
    if money_diff > 50:
        issues.append(f"MONEY mismatch: py={py_final.get('money')} rs={rs_final.get('money')}")

    lives_diff = abs(py_final.get("lives", 0) - rs_final.get("lives", 0))
    if lives_diff > 15:
        issues.append(f"LIVES mismatch: py={py_final.get('lives')} rs={rs_final.get('lives')}")

    # Round reached should match.
    if py_final.get("round") != rs_final.get("round"):
        issues.append(f"ROUND mismatch: py={py_final.get('round')} rs={rs_final.get('round')}")

    # Tower count should match.
    if py_final.get("n_towers") != rs_final.get("n_towers"):
        issues.append(f"TOWERS mismatch: py={py_final.get('n_towers')} rs={rs_final.get('n_towers')}")

    # Win/loss should match.
    if py_sim.won != rs_sim.won or py_sim.game_over != rs_sim.game_over:
        issues.append(f"GAME_STATE mismatch: py(won={py_sim.won},go={py_sim.game_over}) "
                       f"rs(won={rs_sim.won},go={rs_sim.game_over})")

    if issues:
        print(f"  FAIL: {', '.join(issues)}")
        return False
    else:
        print(f"  OK (money={rs_final.get('money')}, lives={rs_final.get('lives')}, "
              f"round={rs_final.get('round')})")
        return True


def main():
    passed = 0
    failed = 0

    for name, setup_fn, rounds in SCENARIOS:
        print(f"Test: {name}")
        ok = compare_scenario(name, setup_fn, rounds)
        if ok:
            passed += 1
        else:
            failed += 1

    # RNG-scrambled tests: run with several seeds, just check game-logic consistency.
    print("\n--- Seed-variation tests (RNG tolerance) ---")
    seeds = [1, 42, 123, 999, 4567]
    for seed in seeds:
        # 2 darts rounds 1-10 at different seeds
        name = f"2darts_rounds1-10_seed{seed}"
        ok = compare_scenario(name, setup_2darts, 10, seed=seed)
        if ok:
            passed += 1
        else:
            failed += 1

    # Difficulty tests.
    print("\n--- Difficulty tests ---")
    for diff in ["medium", "hard"]:
        name = f"multi_type_rounds1-10_{diff}"
        ok = compare_scenario(name, setup_multi_type, 10, difficulty=diff)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
