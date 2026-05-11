"""Interactive playtest entry point.

  python play.py

Controls:
  1            select dart        ($250)
  2            select bomb        ($725)
  3            select tack        ($360)
  4            select spike-o-pult ($600)
  5            select super       ($4000)
  6            select ice         ($425)
  7            select boomerang   ($515)
  Left click   place selected tower (or select existing)
  SPACE        start next round
  ESC          deselect tower-type
  R            reset game
  Q            quit

Debug:
  P            pause / resume
  M            +$1000
  L            +50 lives
  ]            next round +1   (between rounds only)
  [            next round -1   (between rounds only)
  X            clear all bloons (no escape damage)
"""

from __future__ import annotations

import argparse

import pygame

from btd import BloonsSim
from btd.game import SimConfig
from render import PygameRenderer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=3)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=40)
    args = ap.parse_args()

    sim = BloonsSim(SimConfig(track=args.track, difficulty=args.difficulty, seed=args.seed))
    view = PygameRenderer(sim, scale=args.scale)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                view.mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q,):
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    view.selected_tower_type = None
                    view.selected_tower_id = None
                elif event.key == pygame.K_1:
                    view.selected_tower_type = "dart"
                    view.selected_tower_id = None
                elif event.key == pygame.K_2:
                    view.selected_tower_type = "bomb"
                    view.selected_tower_id = None
                elif event.key == pygame.K_3:
                    view.selected_tower_type = "tack"
                    view.selected_tower_id = None
                elif event.key == pygame.K_4:
                    view.selected_tower_type = "spikeopult"
                    view.selected_tower_id = None
                elif event.key == pygame.K_5:
                    view.selected_tower_type = "super"
                    view.selected_tower_id = None
                elif event.key == pygame.K_6:
                    view.selected_tower_type = "ice"
                    view.selected_tower_id = None
                elif event.key == pygame.K_7:
                    view.selected_tower_type = "boomerang"
                    view.selected_tower_id = None
                # --- debug controls ---
                elif event.key == pygame.K_p:
                    view.paused = not view.paused
                elif event.key == pygame.K_m:
                    sim.debug_add_money(1000)
                elif event.key == pygame.K_l:
                    sim.debug_add_lives(50)
                elif event.key == pygame.K_RIGHTBRACKET:
                    sim.debug_set_round(sim.round + 2)  # +1 over the "next" round
                elif event.key == pygame.K_LEFTBRACKET:
                    sim.debug_set_round(sim.round)  # one fewer than the current "next"
                elif event.key == pygame.K_x:
                    sim.debug_clear_bloons()
                elif event.key == pygame.K_SPACE:
                    sim.start_round()
                elif event.key == pygame.K_r:
                    sim = BloonsSim(
                        SimConfig(track=args.track, difficulty=args.difficulty, seed=args.seed)
                    )
                    view.sim = sim
                    view.selected_tower_type = None
                    view.selected_tower_id = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                sx, sy = event.pos
                if not view.in_stage_area(sx, sy):
                    continue
                stage_x, stage_y = view.screen_to_stage(sx, sy)
                if view.selected_tower_type is not None:
                    tid = sim.place_tower(view.selected_tower_type, stage_x, stage_y)
                    if tid != -1:
                        view.selected_tower_id = tid
                else:
                    # Select an existing tower by proximity.
                    view.selected_tower_id = _hit_tower(sim, stage_x, stage_y)

        if not view.paused:
            sim.step()
        view.draw()
        view.tick(args.fps)

    view.quit()


def _hit_tower(sim: BloonsSim, x: float, y: float, max_dist: float = 16.0) -> int | None:
    best = None
    best_d2 = max_dist * max_dist
    for t in sim.towers:
        d2 = (t.x - x) ** 2 + (t.y - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = t.id
    return best


if __name__ == "__main__":
    main()
