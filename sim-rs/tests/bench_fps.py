"""Benchmark: Rust sim vs Python sim, raw simulation throughput.

Usage:
    cd BTD/ && .venv/bin/python sim-rs/tests/bench_fps.py
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim"))

from btd.game import BloonsSim as PyBloonsSim
from btd.game import SimConfig as PyConfig
from btd_rs import BloonsSim as RsBloonsSim
from btd_rs import SimConfig as RsConfig

RS_PATHS = str(REPO / "sim-rs" / "paths")


def make_py(seed=0, freeplay=False):
    return PyBloonsSim(PyConfig(seed=seed, freeplay=freeplay))

def make_rs(seed=0, freeplay=False):
    cfg = RsConfig(seed=seed, freeplay=freeplay, paths_dir=RS_PATHS)
    return RsBloonsSim(cfg)


def play_rnds(sim, n):
    """Play up to `n` rounds; return (frames, rounds_done, timed_out)."""
    f = 0
    done = 0
    for _ in range(n):
        if sim.game_over:
            break
        sim.start_round()
        while sim.in_round and not sim.game_over:
            sim.step()
            f += 1
        done += 1
    return f, done


def bench_one(make_sim, towers, rounds, trials=3):
    """Time how long it takes to play `rounds` worth of sim. Returns (fps, frames, secs, r_done)."""
    best = (0, 0, 0, 0)
    for _ in range(trials):
        sim = make_sim()
        for t, x, y in towers:
            sim.place_tower(t, float(x), float(y))
        t0 = time.perf_counter()
        f, r_done = play_rnds(sim, rounds)
        dt = time.perf_counter() - t0
        fps = f / max(dt, 1e-9)
        if fps > best[0]:
            best = (fps, f, dt, r_done)
    return best


# Tower configs.
LIGHT = [("dart", 350, 100)]
MID   = [("dart", 350, 100), ("dart", 200, 250), ("dart", 400, 300),
         ("dart", 150, 300), ("dart", 300, 350)]
HEAVY = [("dart", 350, 100), ("bomb", 400, 300), ("dart", 200, 250),
         ("bomb", 300, 350), ("tack", 250, 150), ("spikeopult", 150, 300),
         ("ice", 450, 200), ("boomerang", 180, 180), ("super", 320, 200),
         ("bomb", 380, 250)]


def main():
    print("=" * 62)
    print("  BTD3 Simulator — Rust vs Python FPS Benchmark")
    print("=" * 62)

    print("\n  All scenarios: easy difficulty")
    print("  ─────────────────────────────────────────────────────────────")

    # 1 dart (dies ~round 7; use only 10 rounds)
    for label, config, rnds in [
        ("1 dart tower    (dies ~r7)",      LIGHT, 10),
        ("5 dart towers  (survives 50)",    MID,   50),
        ("10 mixed       (survives 50)",    HEAVY, 50),
    ]:
        py_fps, py_f, py_t, _ = bench_one(make_py, config, rnds)
        rs_fps, rs_f, rs_t, _ = bench_one(make_rs, config, rnds)
        spd = rs_fps / py_fps
        print(f"\n  {label}")
        print(f"    Python  {py_fps/1000:>7.0f} K fps  ({py_f:>5} frames in {py_t*1000:>5.0f} ms)")
        print(f"    Rust    {rs_fps/1000:>7.0f} K fps  ({rs_f:>5} frames in {rs_t*1000:>5.0f} ms)")
        print(f"    └─ speedup: {spd:.1f}x")

    # Freeplay
    print(f"\n  freeplay 5 dart (procedural rounds 51+, 80 rounds)")
    py_fps, py_f, py_t, _ = bench_one(lambda s=0: make_py(s, freeplay=True), MID, 80)
    rs_fps, rs_f, rs_t, _ = bench_one(lambda s=0: make_rs(s, freeplay=True), MID, 80)
    spd = rs_fps / py_fps
    print(f"    Python  {py_fps/1000:>7.0f} K fps  ({py_f:>5} frames in {py_t*1000:>5.0f} ms)")
    print(f"    Rust    {rs_fps/1000:>7.0f} K fps  ({rs_f:>5} frames in {rs_t*1000:>5.0f} ms)")
    print(f"    └─ speedup: {spd:.1f}x")

    print()
    print("  Done.")


if __name__ == "__main__":
    main()
