"""Manual smoke test for the sim. Places one dart, runs round 1, prints state.

  python smoke_test.py
"""

from __future__ import annotations

from btd import BloonsSim
from btd.game import SimConfig


def main() -> None:
    sim = BloonsSim(SimConfig(track=3, difficulty="easy", seed=0))

    # Hand-pick a chokepoint near the upper-middle of track 3.
    sim.place_tower("dart", x=350.0, y=100.0)
    sim.place_tower("dart", x=200.0, y=250.0)

    print(f"start: {sim.observe()}")
    assert sim.start_round()
    print(f"round started: queue={len(sim.spawn_queue)} interval={sim.bloon_interval}")

    frame = 0
    while not sim.game_over and sim.in_round:
        sim.step()
        frame += 1
        if frame % 50 == 0:
            print(f"frame {frame}: {sim.observe()}")
        if frame > 5000:
            print("hit step cap; aborting")
            break

    print(f"end:   {sim.observe()}")


if __name__ == "__main__":
    main()
