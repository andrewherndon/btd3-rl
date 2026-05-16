"""Render a debug image of each bloon's extracted hitbox so you can eyeball
sanity. For each rank, draws:
  - The extracted bbox as a filled rectangle in the rank's canonical color.
  - The proposed collision shape (circle for ranks 1-8, AABB for ranks 9-10),
    outlined in lime.
  - The inner sprite's origin (where `bloon.x, bloon.y` in the sim is) as +.
  - The bbox center as a dot.

All panels share the same axis range so sizes are directly comparable. Y is
inverted to match the game's top-left = (0, 0) convention.

  python visualize_hitboxes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Same palette as render.py.
BLOON_COLORS_RGB = {
    1: (220, 30, 30),     2: (50, 110, 230),    3: (40, 180, 70),
    4: (240, 220, 50),    5: (35, 35, 40),      6: (245, 245, 245),
    7: (115, 115, 130),   8: (200, 70, 200),    9: (140, 50, 40),
    10: (90, 50, 40),
}

# Bigger bloons use AABB collision; smaller ones use a circumscribed circle.
AABB_RANKS = {9, 10}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=Path("paths/bloon_hitboxes.json"))
    ap.add_argument("--out", type=Path, default=Path("paths/bloon_hitboxes_debug.png"))
    args = ap.parse_args()

    data = json.loads(args.json.read_text())

    fig, axes = plt.subplots(2, 5, figsize=(22, 10), dpi=110)
    # Shared axis range: large enough to fit MOAB, padded a bit.
    xlim = (-95, 55)
    ylim = (35, -70)  # inverted — Flash y is down

    for rank in range(1, 11):
        info = data[str(rank)]
        ax = axes[(rank - 1) // 5][(rank - 1) % 5]
        half_w, half_h = info["half_w"], info["half_h"]
        cx, cy = info["cx"], info["cy"]

        rgb = tuple(c / 255.0 for c in BLOON_COLORS_RGB[rank])

        # Filled bbox in the rank's color, with a subtle gray border.
        bbox_rect = mpatches.Rectangle(
            (cx - half_w, cy - half_h), 2 * half_w, 2 * half_h,
            facecolor=rgb, edgecolor="dimgray", linewidth=0.8, alpha=0.5,
        )
        ax.add_patch(bbox_rect)

        # Collision shape overlay (lime outline). Circle for small bloons,
        # AABB for big ones.
        if rank in AABB_RANKS:
            coll_patch = mpatches.Rectangle(
                (cx - half_w, cy - half_h), 2 * half_w, 2 * half_h,
                facecolor="none", edgecolor="#1aff1a", linewidth=2.5,
            )
            coll_label = f"AABB 2·({half_w:.1f},{half_h:.1f})"
        else:
            radius = (half_w ** 2 + half_h ** 2) ** 0.5  # circumscribed
            coll_patch = mpatches.Circle(
                (cx, cy), radius,
                facecolor="none", edgecolor="#1aff1a", linewidth=2.5,
            )
            coll_label = f"circle r={radius:.1f}"
        ax.add_patch(coll_patch)

        # Origin (sim's bloon.x, bloon.y reference) and bbox center.
        ax.plot(0, 0, "k+", markersize=16, markeredgewidth=2.0)
        ax.plot(cx, cy, "ko", markersize=4)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)  # inverted
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.3, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.3, alpha=0.5)
        ax.set_title(
            f"rank {rank}: {info['name']}\n"
            f"bbox {2*half_w:.1f} × {2*half_h:.1f} px"
            f"   center ({cx:+.1f}, {cy:+.1f})\n"
            f"{coll_label}",
            fontsize=10,
        )

    # Legend on the figure.
    legend_handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="dimgray", label="extracted bbox"),
        mpatches.Patch(facecolor="none", edgecolor="#1aff1a", label="proposed collision shape"),
        plt.Line2D([0], [0], color="black", marker="+", linestyle="None",
                   markersize=12, markeredgewidth=2, label="sim origin (bloon.x, bloon.y)"),
        plt.Line2D([0], [0], color="black", marker="o", linestyle="None",
                   markersize=5, label="bbox center"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Bloon hitboxes (frame 1 of each inner sprite, in the inner's local frame)",
        fontsize=14, y=0.995,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
