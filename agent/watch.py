"""Watch a trained MaskablePPO agent play one game, rendered at human speed.

    python agent/watch.py --model agent/models/best_model

Checkpoint-replay viewer: training is headless and uncapped; this loads a saved
policy and plays ONE game with the pygame renderer, throttled so a round runs at
~40fps. It drives the sim directly (rather than BloonsEnv.step) so the battle
frames are visible instead of being fast-forwarded inside a START_ROUND.

Live controls:  [ / ]  slow down / speed up      P  pause      Q / Esc  quit

Speed changes only the render pacing, never the sim math — the sim is
fixed-timestep (one sim.step() == one 1/40s game frame, no wall-clock dt), so
any speed replays the same deterministic game. High speeds skip *draws*, never
*steps*, so determinism holds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# render.py lives under sim/ and isn't an installed package (it's pygame-only);
# add sim/ to the path so we can import the renderer without touching the core.
SIM_DIR = Path(__file__).resolve().parents[1] / "sim"
sys.path.insert(0, str(SIM_DIR))

import pygame  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402

from btd.game import BloonsSim, SimConfig  # noqa: E402
from render import PygameRenderer  # noqa: E402

from envs import actions as A  # noqa: E402
from envs.actions import Kind, cell_to_xy, decode  # noqa: E402
from envs.bloons_env import MAX_ECON_PER_ROUND  # noqa: E402
from envs.mask import build_action_mask, compute_cell_validity  # noqa: E402
from envs.observation import encode  # noqa: E402

BASE_FPS = 40.0        # game frames per second at 1x
DRAW_CAP = 60.0        # above this game-fps we skip draws instead of drawing all
SPEED_MIN, SPEED_MAX = 0.25, 32.0


def pacing(speed: float) -> tuple[int, int]:
    """(sim steps per drawn frame, draw fps) for a given speed multiplier.

    speed <= 0 -> uncapped (batch many steps per draw, no throttle; for headless
    runs). Otherwise pick steps/draw so the *game* runs at BASE_FPS*speed while
    the *draw* rate stays <= DRAW_CAP. All steps always execute (exact calc);
    only draws are skipped at high speed."""
    if speed <= 0:
        return 10_000, 0
    game_fps = BASE_FPS * speed
    steps = max(1, round(game_fps / DRAW_CAP))
    draw_fps = max(1, round(game_fps / steps))
    return steps, draw_fps


def choose_action(model, sim, cell_valid, econ_streak, deterministic) -> int:
    """Same decision logic as BloonsEnv: encode obs, build the legality mask
    (forcing START_ROUND if the agent has shopped too long), predict."""
    obs = encode(sim)
    if econ_streak >= MAX_ECON_PER_ROUND:
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        mask[A.START_ROUND] = True
    else:
        mask = build_action_mask(sim, cell_valid)
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    return int(action)


def apply_economic(sim: BloonsSim, act) -> None:
    if act.kind == Kind.PLACE:
        x, y = cell_to_xy(act.b)
        sim.place_tower(act.tower_type, x, y)
    elif act.kind == Kind.UPGRADE and act.a < len(sim.towers):
        sim.upgrade_path(sim.towers[act.a].id, act.b)
    elif act.kind == Kind.SELL and act.a < len(sim.towers):
        sim.sell_tower(sim.towers[act.a].id)


def _set_caption(speed: float) -> None:
    pygame.display.set_caption(f"BTD3 — agent replay — {speed:g}x")


def pump_events(state: dict) -> None:
    """Handle quit / pause / speed; toggles live in `state` so it works
    mid-round too."""
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            state["running"] = False
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_q, pygame.K_ESCAPE):
                state["running"] = False
            elif e.key == pygame.K_p:
                state["paused"] = not state["paused"]
            elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
                state["speed"] = min(SPEED_MAX, state["speed"] * 1.5)
                _set_caption(state["speed"])
            elif e.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                state["speed"] = max(SPEED_MIN, state["speed"] / 1.5)
                _set_caption(state["speed"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="agent/models/best_model")
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--speed", type=float, default=1.0, help="initial speed; 0 = uncapped")
    p.add_argument("--shop-fps", type=int, default=5, help="shopping pacing at 1x")
    p.add_argument("--stochastic", action="store_true", help="sample instead of argmax")
    p.add_argument("--max-rounds", type=int, default=0, help="stop after N rounds (0=full game)")
    p.add_argument("--auto-close", action="store_true", help="don't wait after game ends")
    args = p.parse_args()

    model = MaskablePPO.load(args.model)
    sim = BloonsSim(SimConfig(difficulty=args.difficulty, seed=args.seed or 0))
    cell_valid = compute_cell_validity(sim)
    renderer = PygameRenderer(sim, scale=args.scale, caption="BTD3 — agent replay")
    _set_caption(args.speed)

    def tick(fps: int) -> None:
        if fps and fps > 0:
            renderer.tick(fps)

    state = {"running": True, "paused": False, "speed": args.speed}
    econ_streak = 0
    deterministic = not args.stochastic

    while state["running"] and not sim.game_over:
        pump_events(state)
        renderer.paused = state["paused"]
        if state["paused"]:
            renderer.draw()
            tick(30)
            continue

        act = decode(choose_action(model, sim, cell_valid, econ_streak, deterministic))

        if act.kind == Kind.START_ROUND:
            econ_streak = 0
            lives_before = sim.lives
            sim.start_round()
            print(f"round {sim.round}: {len(sim.towers)} towers, ${sim.money}")
            while sim.in_round and not sim.game_over and state["running"]:
                pump_events(state)
                renderer.paused = state["paused"]
                steps_per_draw, draw_fps = pacing(0 if state["paused"] else state["speed"])
                if not state["paused"]:
                    for _ in range(steps_per_draw):
                        sim.step()
                        if not sim.in_round or sim.game_over:
                            break
                renderer.draw()
                tick(draw_fps if not state["paused"] else 30)
            print(f"  -> round {sim.round}: lives {sim.lives} "
                  f"(-{lives_before - sim.lives}), ${sim.money}")
            if args.max_rounds and sim.round >= args.max_rounds:
                break
        else:
            apply_economic(sim, act)
            econ_streak += 1
            renderer.draw()
            tick(min(int(args.shop_fps * max(state["speed"], 1)), 60))

    renderer.draw()
    result = "WON" if sim.won else ("LOST" if sim.game_over else "stopped")
    print(f"game over: {result} at round {sim.round}, lives {sim.lives}")
    if not args.auto_close:
        while state["running"]:
            pump_events(state)
            renderer.draw()
            tick(30)
    renderer.quit()


if __name__ == "__main__":
    main()
