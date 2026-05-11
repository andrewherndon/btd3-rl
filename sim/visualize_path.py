"""Plot an extracted bloon path over the 640x480 stage and save a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np

STAGE_W, STAGE_H = 640, 480


def plot_path(path: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Stage rectangle.
    ax.add_patch(
        plt.Rectangle((0, 0), STAGE_W, STAGE_H, fill=False, edgecolor="black", lw=1)
    )

    # Path as a line plus a thin scatter of every 10th frame, with start/end markers.
    ax.plot(path[:, 0], path[:, 1], "-", lw=1.5, color="tab:blue", label="path")
    ax.scatter(path[::10, 0], path[::10, 1], s=4, color="tab:blue", alpha=0.4)
    ax.scatter(path[0, 0], path[0, 1], s=80, color="green", zorder=5, label="start")
    ax.scatter(path[-1, 0], path[-1, 1], s=80, color="red", zorder=5, label="end")

    ax.set_xlim(-50, STAGE_W + 50)
    ax.set_ylim(STAGE_H + 150, -150)  # Flash y is top-down: invert
    ax.set_aspect("equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    path = np.load(args.npy)
    title = args.title or args.npy.stem
    plot_path(path, args.out, title)


if __name__ == "__main__":
    main()
