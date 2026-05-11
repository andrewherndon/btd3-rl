"""Pygame renderer for BloonsSim.

Placeholder graphics: bloons are colored circles sized by rank, towers are
squares with a range ring on selection, bullets are small white dots, the
path is a polyline. The HUD shows money / lives / round / state.

Decoupled from the sim's tick loop — the caller drives both. Used by play.py
(interactive) and could be used by a headless rollout recorder later.
"""

from __future__ import annotations

from typing import Optional

import pygame

from btd.constants import BLOON_RADIUS, STAGE_H, STAGE_W
from btd.game import BloonsSim


# Bloon body colors, by rank. BTD canon.
BLOON_COLORS: dict[int, tuple[int, int, int]] = {
    1: (220, 30, 30),     # red
    2: (50, 110, 230),    # blue
    3: (40, 180, 70),     # green
    4: (240, 220, 50),    # yellow
    5: (35, 35, 40),      # black
    6: (245, 245, 245),   # white
    7: (115, 115, 130),   # lead (placeholder; lead is shiny in canon)
    8: (200, 70, 200),    # rainbow placeholder
    9: (140, 50, 40),     # MOAB
    10: (90, 50, 40),     # BFB
}

TOWER_COLORS: dict[str, tuple[int, int, int]] = {
    "dart": (175, 130, 80),
    "tack": (235, 140, 50),
    "ice": (170, 215, 240),
    "bomb": (60, 60, 60),
    "boomerang": (200, 140, 60),
    "spikeopult": (130, 90, 50),
    "super": (220, 80, 30),
    "beacon": (200, 200, 60),
}

PATH_COLOR = (190, 165, 110)
PATH_WIDTH = 14
GRASS_COLOR = (110, 160, 90)
HUD_BG = (245, 245, 230)
HUD_BORDER = (60, 60, 60)
HUD_TEXT = (30, 30, 30)
BULLET_COLOR = (240, 240, 240)
RANGE_RING = (255, 255, 255, 60)  # not used directly (no alpha on plain draw)


class PygameRenderer:
    """Draw a `BloonsSim` to a pygame window.

    `scale` enlarges the 640x480 stage. `hud_width` is screen pixels reserved
    for the HUD strip on the right (in addition to the stage)."""

    def __init__(
        self,
        sim: BloonsSim,
        scale: float = 2.0,
        hud_width: int = 250,
        caption: str = "BTD3 Sim",
    ) -> None:
        pygame.init()
        pygame.display.set_caption(caption)
        self.sim = sim
        self.scale = scale
        self.hud_w = hud_width
        self.stage_px_w = int(STAGE_W * scale)
        self.stage_px_h = int(STAGE_H * scale)
        self.screen = pygame.display.set_mode(
            (self.stage_px_w + hud_width, self.stage_px_h)
        )
        self.font = pygame.font.SysFont("menlo", int(14 * (scale / 2)))
        self.big_font = pygame.font.SysFont("menlo", int(20 * (scale / 2)), bold=True)
        self.clock = pygame.time.Clock()
        # Highlight target for tower placement preview, if any.
        self.selected_tower_type: Optional[str] = None
        self.selected_tower_id: Optional[int] = None
        self.mouse_pos: tuple[int, int] = (0, 0)
        # Interactive playtest state; driven by play.py.
        self.paused: bool = False

    # -- coordinate transforms ------------------------------------------------

    def stage_to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (int(x * self.scale), int(y * self.scale))

    def screen_to_stage(self, sx: int, sy: int) -> tuple[float, float]:
        return (sx / self.scale, sy / self.scale)

    def in_stage_area(self, sx: int, sy: int) -> bool:
        return 0 <= sx < self.stage_px_w and 0 <= sy < self.stage_px_h

    # -- main draw entry ------------------------------------------------------

    def draw(self) -> None:
        self.screen.fill(GRASS_COLOR)
        self._draw_path()
        self._draw_towers()
        self._draw_bullets()
        self._draw_bloons()
        self._draw_placement_preview()
        self._draw_hud()
        pygame.display.flip()

    # -- pieces ---------------------------------------------------------------

    def _draw_path(self) -> None:
        path = self.sim.paths[1]
        # Clip to on-screen segment; the path enters / exits off the stage,
        # so we use it as-is and let pygame clip.
        pts = [self.stage_to_screen(p[0], p[1]) for p in path]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, PATH_COLOR, False, pts, PATH_WIDTH)

    def _draw_towers(self) -> None:
        for t in self.sim.towers:
            cx, cy = self.stage_to_screen(t.x, t.y)
            color = TOWER_COLORS.get(t.type, (180, 180, 180))
            size = max(10, int(10 * self.scale))
            rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (30, 30, 30), rect, 1)
            if self.selected_tower_id == t.id:
                self._draw_range_ring(cx, cy, t.attack_radius)

    def _draw_range_ring(self, cx: int, cy: int, radius_stage: float) -> None:
        r = int(radius_stage * self.scale)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), r, 1)

    def _draw_bullets(self) -> None:
        for b in self.sim.bullets:
            if b.is_dead:
                continue
            cx, cy = self.stage_to_screen(b.x, b.y)
            pygame.draw.circle(
                self.screen, BULLET_COLOR, (cx, cy), max(2, int(b.radius * self.scale * 0.6))
            )

    def _draw_bloons(self) -> None:
        for b in self.sim.bloons:
            if not b.alive:
                continue
            cx, cy = self.stage_to_screen(b.x, b.y)
            r = max(3, int(BLOON_RADIUS[b.rank] * self.scale))
            color = BLOON_COLORS.get(b.rank, (255, 0, 255))
            pygame.draw.circle(self.screen, color, (cx, cy), r)
            pygame.draw.circle(self.screen, (20, 20, 20), (cx, cy), r, 1)
            # White bloon needs outline for visibility against grass.
            if b.rank == 6:
                pygame.draw.circle(self.screen, (40, 40, 40), (cx, cy), r, 2)
            # Frozen bloons get a thick cyan halo.
            if b.frozen:
                pygame.draw.circle(self.screen, (140, 220, 255), (cx, cy), r + 3, 2)

    def _draw_placement_preview(self) -> None:
        if self.selected_tower_type is None:
            return
        sx, sy = self.mouse_pos
        if not self.in_stage_area(sx, sy):
            return
        color = TOWER_COLORS.get(self.selected_tower_type, (200, 200, 200))
        size = max(10, int(10 * self.scale))
        rect = pygame.Rect(sx - size // 2, sy - size // 2, size, size)
        ghost = pygame.Surface((size, size), pygame.SRCALPHA)
        ghost.fill((*color, 130))
        self.screen.blit(ghost, rect.topleft)
        # Range preview.
        from btd.constants import TOWER_STATS
        r = int(TOWER_STATS[self.selected_tower_type]["attackRadius"] * self.scale)
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), r, 1)

    def _draw_hud(self) -> None:
        x0 = self.stage_px_w
        hud_rect = pygame.Rect(x0, 0, self.hud_w, self.stage_px_h)
        pygame.draw.rect(self.screen, HUD_BG, hud_rect)
        pygame.draw.rect(self.screen, HUD_BORDER, hud_rect, 2)
        s = self.sim
        next_round = s.round + 1 if not s.in_round else s.round
        round_label = f"Round {s.round}" + (" *" if s.in_round else "")
        lines: list[tuple[str, pygame.font.Font]] = [
            (round_label, self.big_font),
        ]
        if self.paused:
            lines.append(("[PAUSED]", self.big_font))
        lines += [
            (f"Money:  ${s.money}", self.font),
            (f"Lives:  {s.lives}", self.font),
            (f"Frame:  {s.frame_count}", self.font),
            ("", self.font),
            (f"Bloons:  {sum(b.alive for b in s.bloons)}", self.font),
            (f"Towers:  {len(s.towers)}", self.font),
            (f"Bullets: {sum(not b.is_dead for b in s.bullets)}", self.font),
            ("", self.font),
            (f"Sel:     {self.selected_tower_type or '-'}", self.font),
            (f"Next rd: {next_round}", self.font),
            ("", self.font),
            ("--- play ---", self.font),
            ("[SPACE] start round", self.font),
            ("[1] dart       $250", self.font),
            ("[2] bomb       $725", self.font),
            ("[3] tack       $360", self.font),
            ("[4] spikeopult $600", self.font),
            ("[5] super     $4000", self.font),
            ("[6] ice        $425", self.font),
            ("[ESC] deselect", self.font),
            ("[Q]   quit  [R] reset", self.font),
            ("", self.font),
            ("--- debug ---", self.font),
            ("[P]   pause/resume", self.font),
            ("[M]   +$1000", self.font),
            ("[L]   +50 lives", self.font),
            ("[]]   next round +1", self.font),
            ("[[]   next round -1", self.font),
            ("[X]   clear bloons", self.font),
        ]
        if s.game_over:
            lines.insert(1, ("WIN" if s.won else "GAME OVER", self.big_font))
        y = 12
        for text, font in lines:
            if text:
                surf = font.render(text, True, HUD_TEXT)
                self.screen.blit(surf, (x0 + 12, y))
            y += font.get_height() + 2

    # -- pacing ---------------------------------------------------------------

    def tick(self, fps: int = 40) -> None:
        self.clock.tick(fps)

    def quit(self) -> None:
        pygame.quit()
